"""Trusted, same-environment Pipeline worker transport (internal API).

Workers reconstruct definitions through the original entrypoint. Never load these
pickle archives from an untrusted source. Only the main thread publishes events.
"""

import gzip
import json
import logging
import os
import pickle
import subprocess
import sys
import tarfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from queue import Empty, Full, Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import MappingProxyType
from typing import Any, BinaryIO

import matplotlib.pyplot as plt

from .env import get_interactivity
from .pipeable import (
    _add_tar_binary_entry,
    _save_image_bytes_to_dir,
    _split_evenly_in_order,
    _write_parallel_streaming_result_entries,
)
from .pipeable_io import (
    ExitCodes,
    exit_with_error_streaming,
    redirect_stdout_to_stderr,
)
from .pipeline import (
    DataFrameInputSnapshot,
    Inputs,
    Lane,
    MapExecutionContext,
    MapSegment,
    Stage,
    _validate_stages,
    execute_map_segment,
    execute_pipeline,
    execute_sequential,
)

logger = logging.getLogger(__name__)

# data_name, image_name, encoded image, image_type
ImageSink = Callable[[str, str, bytes, str | None], None]


class TransferError(RuntimeError):
    pass


def dumps(value: Any, label: str) -> bytes:
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise TransferError(
            f"cannot pickle {label}: {type(exc).__name__}: {exc}"
        ) from exc


def canonical_manifest(stages: Sequence[Stage]) -> str:
    def identity(func):
        module = func.__module__
        if module == "__main__":
            module = "<entrypoint>"
        return [module, func.__qualname__]

    return json.dumps(
        {
            "version": 1,
            "stages": [
                {"kind": stage.kind, "callable": identity(stage.func)}
                for stage in stages
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class FileInputSnapshot:
    def __init__(self, lanes: tuple[Lane, ...], paths: Mapping[int, Path]):
        self.lanes = lanes
        self.paths = MappingProxyType(dict(paths))

    def restore(self, lane_id: int) -> Inputs:
        return pickle.loads(self.paths[lane_id].read_bytes())

    def restore_all(self) -> Mapping[str, Inputs]:
        return MappingProxyType(
            {lane.data_name: self.restore(lane.lane_id) for lane in self.lanes}
        )


@contextmanager
def file_snapshot(inputs: Inputs, lanes: tuple[Lane, ...]):
    with TemporaryDirectory(prefix="analysisrun-pipeline-") as directory:
        source = DataFrameInputSnapshot(inputs, lanes)
        paths = {}
        for lane in lanes:
            path = Path(directory).resolve() / f"{lane.lane_id}.pkl"
            path.write_bytes(
                dumps(
                    source.restore(lane.lane_id),
                    f"input lane_id={lane.lane_id} data_name={lane.data_name!r}",
                )
            )
            paths[lane.lane_id] = path
        del source
        yield FileInputSnapshot(lanes, paths)


def worker_limit() -> int:
    setting = os.environ.get("ANALYSISRUN_MAX_WORKERS")
    if setting is None:
        return os.process_cpu_count() or 1
    if not setting.isascii() or not setting.isdecimal() or int(setting) <= 0:
        raise ValueError("ANALYSISRUN_MAX_WORKERS must be a positive integer")
    return int(setting)


class _CountingWriter:
    """Give TarFile a logical position without seeking or record buffering.

    TarFile's uncompressed write mode only needs write/tell/flush. Entries reach
    the pipe immediately, through public file operations, including small values.
    """

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.position = 0

    def write(self, data):
        written = self.stream.write(data)
        self.position += written
        return written

    def tell(self):
        return self.position

    def flush(self):
        self.stream.flush()


@contextmanager
def writing_tar(stream: BinaryIO):
    writer = _CountingWriter(stream)
    with tarfile.open(fileobj=writer, mode="w") as archive:
        try:
            yield archive
        finally:
            # TarFile.__exit__ omits end markers on exceptions. Explicitly close
            # first: our protocol requires end markers even after SystemExit.
            archive.close()
            writer.flush()


def _add(archive, name, data, image_type=None):
    _add_tar_binary_entry(archive, name, data, image_type=image_type)


class _TrackedReader:
    def __init__(self, stream):
        self.stream = stream
        self.size = 0
        self.tail = b""

    def read(self, size=-1):
        data = self.stream.read(size)
        self.size += len(data)
        self.tail = (self.tail + data)[-1024:]
        return data


def _read_entries(stream, on_entry):
    reader = _TrackedReader(stream)
    with tarfile.open(fileobj=reader, mode="r|", bufsize=512) as archive:
        for member in archive:
            if not member.isfile():
                raise TransferError(f"unexpected non-file entry: {member.name}")
            on_entry(member, archive.extractfile(member).read())
        end_offset = archive.offset
    while data := reader.read(65536):
        if data.strip(b"\0"):
            raise TransferError("corrupt TAR: data after end marker")
    if (
        reader.size < end_offset + 1024
        or reader.size % 512
        or reader.tail != bytes(1024)
    ):
        raise TransferError("corrupt TAR: missing end markers")


def _validate_request(request, stages):
    _validate_stages(stages)
    if request["protocol_version"] != 1:
        raise TransferError("protocol_version mismatch")
    if request["manifest"] != canonical_manifest(stages):
        raise TransferError("pipeline manifest mismatch")
    start, end = request["stage_start"], request["stage_end"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(stages)
        or any(stage.kind != "map" for stage in stages[start:end])
    ):
        raise TransferError("invalid map segment range")
    ids, names = set(), set()
    for lane in request["lanes"]:
        lane_id = lane["lane_id"]
        if (
            type(lane_id) is not int
            or lane_id < 0
            or lane_id in ids
            or not isinstance(lane["data_name"], str)
            or lane["data_name"] in names
            or not isinstance(lane["sample_name"], str)
            or not Path(lane["input_path"]).is_absolute()
        ):
            raise TransferError("invalid lane set")
        ids.add(lane_id)
        names.add(lane["data_name"])
    return MapSegment(start, end, tuple(stages[start:end]))


class _WorkerOutput:
    def __init__(self, archive, stream, lane_id):
        self.archive, self.stream, self.lane_id = archive, stream, lane_id

    def __call__(self, fig, name, image_type, **kwargs):
        buffer = BytesIO()
        fig.savefig(buffer, **kwargs)
        _add(
            self.archive,
            f"lanes/{self.lane_id}/images/{name}",
            buffer.getvalue(),
            image_type,
        )
        self.stream.flush()
        fig.clear()
        plt.close(fig)


def run_worker(stages: Sequence[Stage], stdin: BinaryIO, stdout: BinaryIO) -> int:
    """Read the complete request before writing any response bytes."""
    entries = {}
    lane = None
    stage_index = None

    def collect(member, data):
        if member.name in entries:
            raise TransferError(f"duplicate request entry: {member.name}")
        entries[member.name] = data

    # Even request failures produce a structured response, after stdin is drained.
    try:
        _read_entries(stdin, collect)
        request = json.loads(entries["request.json"])
        segment = _validate_request(request, stages)
        expected = {"request.json", "params.pkl", "synthesized.pkl"} | {
            f"lanes/{item['lane_id']}/value.pkl" for item in request["lanes"]
        }
        if entries.keys() != expected:
            raise TransferError("unexpected or missing request entries")
    except Exception as exc:
        logger.exception("worker request failed")
        while stdin.read(65536):
            pass
        with writing_tar(stdout) as archive:
            _write_error(archive, exc, None, None)
        return 1

    with writing_tar(stdout) as archive:
        try:
            for lane in request["lanes"]:
                stage_index = None
                lane_id = lane["lane_id"]

                def before_stage(index):
                    nonlocal stage_index
                    stage_index = index

                value = execute_map_segment(
                    segment,
                    lane=Lane(lane_id, lane["data_name"], lane["sample_name"]),
                    restore_inputs=lambda path=lane["input_path"]: pickle.loads(
                        Path(path).read_bytes()
                    ),
                    restore_params=lambda: pickle.loads(entries["params.pkl"]),
                    restore_synthesized=lambda: pickle.loads(
                        entries["synthesized.pkl"]
                    ),
                    value=pickle.loads(entries[f"lanes/{lane_id}/value.pkl"]),
                    output=_WorkerOutput(archive, stdout, lane_id),
                    before_stage=before_stage,
                )
                _add(
                    archive,
                    f"lanes/{lane_id}/value.pkl",
                    dumps(
                        value,
                        f"value lane_id={lane_id} data_name={lane['data_name']!r} "
                        f"stage_index={stage_index}",
                    ),
                )
                stdout.flush()
        except Exception as exc:
            logger.exception("worker map failed")
            _write_error(archive, exc, lane, stage_index)
            return 1
    return 0


def _write_error(archive, exc, lane, stage_index):
    error = {
        "protocol_version": 1,
        "lane_id": lane["lane_id"] if lane else None,
        "data_name": lane["data_name"] if lane else None,
        "stage_index": stage_index,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    _add(archive, "error.json", json.dumps(error, ensure_ascii=False).encode())


def receive_response(stream, request, emit):
    expected = {lane["lane_id"]: lane for lane in request["lanes"]}
    seen = set()

    def receive(member, data):
        if member.name == "error.json":
            error = json.loads(data)
            if error["protocol_version"] != 1:
                raise TransferError("response protocol_version mismatch")
            raise TransferError(json.dumps(error, ensure_ascii=False))
        parts = member.name.split("/")
        if len(parts) < 3 or parts[0] != "lanes":
            raise TransferError(f"unexpected response entry: {member.name}")
        lane_id = int(parts[1])
        if lane_id not in expected:
            raise TransferError(f"unknown lane_id={lane_id}")
        if parts[2:] == ["value.pkl"]:
            if lane_id in seen:
                raise TransferError(f"duplicate value.pkl lane_id={lane_id}")
            try:
                value = pickle.loads(data)
            except Exception as exc:
                raise TransferError(
                    f"cannot unpickle value.pkl lane_id={lane_id} "
                    f"data_name={expected[lane_id]['data_name']!r} "
                    f"stage_start={request['stage_start']} "
                    f"stage_end={request['stage_end']}: {type(exc).__name__}: {exc}"
                ) from exc
            seen.add(lane_id)
            emit("value", lane_id, value)
        elif parts[2] == "images" and len(parts) > 3 and "/".join(parts[3:]):
            emit(
                "image",
                lane_id,
                ("/".join(parts[3:]), data, member.pax_headers.get("image_type")),
            )
        else:
            raise TransferError(f"unexpected response entry: {member.name}")

    _read_entries(stream, receive)
    if seen != expected.keys():
        raise TransferError(f"missing value.pkl lanes={sorted(expected.keys() - seen)}")


def _stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    proc.wait()


class MultiprocessMapExecutor:
    def __init__(
        self,
        stages: Sequence[Stage],
        entrypoint: Path,
        on_image: ImageSink,
        *,
        max_workers: int | None = None,
        queue_size: int = 16,
    ):
        self.manifest = canonical_manifest(stages)
        self.entrypoint = Path(entrypoint).resolve()
        self.on_image = on_image
        self.max_workers = worker_limit() if max_workers is None else max_workers
        if type(self.max_workers) is not int or self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if type(queue_size) is not int or queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")
        self.queue_size = queue_size

    def execute(self, segment, context: MapExecutionContext, on_value):
        if not isinstance(context.snapshot, FileInputSnapshot):
            raise TypeError("multiprocess execution requires FileInputSnapshot")
        chunks = _split_evenly_in_order(list(context.lanes), self.max_workers)
        if not chunks:
            return
        # Serialize shared state once, before spawning. No live values are read by
        # management threads while the main thread is accepting worker results.
        params = dumps(context.params, f"params stage_start={segment.start}")
        synthesized = dumps(
            context.synthesized, f"synthesized stage_start={segment.start}"
        )
        requests = []
        for chunk in chunks:
            lanes = [
                {
                    "lane_id": lane.lane_id,
                    "data_name": lane.data_name,
                    "sample_name": lane.sample_name,
                    "input_path": str(context.snapshot.paths[lane.lane_id]),
                }
                for lane in chunk
            ]
            request = {
                "protocol_version": 1,
                "manifest": self.manifest,
                "stage_start": segment.start,
                "stage_end": segment.end,
                "lanes": lanes,
            }
            values = {
                lane.lane_id: dumps(
                    context.values[lane.data_name],
                    f"value lane_id={lane.lane_id} data_name={lane.data_name!r} "
                    f"stage_start={segment.start}",
                )
                for lane in chunk
            }
            requests.append((request, values))

        events = Queue(maxsize=self.queue_size)
        stopped = Event()
        processes = []

        def emit(*event):
            while not stopped.is_set():
                try:
                    events.put(event, timeout=0.05)
                    return
                except Full:
                    pass

        def manage(proc, request, values):
            diagnostics = bytearray()

            def drain():
                while data := proc.stderr.read(65536):
                    diagnostics.extend(data)

            drainer = Thread(target=drain)
            drainer.start()
            error = None
            try:
                with proc.stdin, writing_tar(proc.stdin) as archive:
                    _add(archive, "request.json", json.dumps(request).encode())
                    _add(archive, "params.pkl", params)
                    _add(archive, "synthesized.pkl", synthesized)
                    for lane_id, data in values.items():
                        _add(archive, f"lanes/{lane_id}/value.pkl", data)
                receive_response(proc.stdout, request, emit)
                if proc.wait() != 0:
                    raise TransferError(f"worker exited with code {proc.returncode}")
            except Exception as exc:
                logger.exception("worker communication failed")
                error = exc
                _stop_process(proc)
            finally:
                proc.wait()
                drainer.join()
                proc.stdout.close()
                proc.stderr.close()
            if error:
                wrapped = TransferError(
                    f"worker failure: {error}; exit={proc.returncode}; "
                    f"stderr={diagnostics.decode(errors='replace')}"
                )
                wrapped.__cause__ = error
                emit("error", wrapped)
            else:
                emit(
                    "done",
                )

        pool = ThreadPoolExecutor(max_workers=len(chunks))
        futures = []
        try:
            env = os.environ | {"ANALYSISRUN_MODE": "pipeline-worker"}
            for request, values in requests:
                proc = subprocess.Popen(
                    [sys.executable, str(self.entrypoint)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                processes.append(proc)
                futures.append(pool.submit(manage, proc, request, values))
            remaining = len(futures)
            lanes = {lane.lane_id: lane for lane in context.lanes}
            while remaining:
                try:
                    event = events.get(timeout=0.1)
                except Empty:
                    # Surface unexpected management-thread errors as well.
                    for future in futures:
                        if future.done():
                            future.result()
                    continue
                kind, *payload = event
                if kind == "error":
                    raise payload[0]
                if kind == "done":
                    remaining -= 1
                elif kind == "value":
                    on_value(*payload)
                elif kind == "image":
                    lane_id, (name, data, image_type) = payload
                    self.on_image(lanes[lane_id].data_name, name, data, image_type)
            for future in futures:
                future.result()
        finally:
            stopped.set()
            # Processes are created only in this thread, so cancellation cannot
            # miss a child that is still being registered by a management thread.
            try:
                for proc in processes:
                    _stop_process(proc)
            finally:
                pool.shutdown(wait=True)
                for proc in processes:
                    for stream in (proc.stdin, proc.stdout, proc.stderr):
                        stream.close()


def execute_entrypoint(
    stages: Sequence[Stage],
    *,
    entrypoint: Path,
    params=None,
    image_analysis_results=None,
    targets=None,
    output=None,
    output_dir: Path | None = None,
    mode: str | None = None,
    stdout=None,
    stderr=None,
):
    """Internal execution entrance, pending the public builder in #86.

    Reconstruct stages before calling this in both parent and worker. Worker
    dispatch precedes input loading by the caller; no parent inputs are required.
    Notebook callers select sequential mode and provide a figure Output.
    """
    mode = (
        mode
        or os.environ.get("ANALYSISRUN_MODE")
        or (
            "sequential" if get_interactivity() == "notebook" else "parallel-entrypoint"
        )
    )
    if mode == "sequential":
        return execute_sequential(
            stages,
            params=params,
            image_analysis_results=image_analysis_results,
            targets=targets,
            output=output,
        )
    stdout = stdout if stdout is not None else sys.stdout.buffer
    stderr = stderr if stderr is not None else sys.stderr.buffer
    if mode == "pipeline-worker":
        with redirect_stdout_to_stderr(stderr):
            status = run_worker(stages, sys.stdin.buffer, stdout)
        sys.exit(status)
    if mode not in {"parallel-entrypoint", "parallel-entrypoint-streaming"}:
        raise ValueError(f"unsupported pipeline mode: {mode!r}")

    def run(on_image):
        executor = MultiprocessMapExecutor(stages, entrypoint, on_image)
        return execute_pipeline(
            stages,
            params=params,
            image_analysis_results=image_analysis_results,
            targets=targets,
            output=output,
            map_executor=executor,
            snapshot_factory=file_snapshot,
        )

    if mode == "parallel-entrypoint":
        if output_dir is None:
            raise ValueError("parallel-entrypoint requires output_dir")
        return run(
            lambda data_name, name, data, image_type: _save_image_bytes_to_dir(
                name, data, output_dir
            )
        )
    with (
        gzip.GzipFile(fileobj=stdout, mode="wb") as compressed,
        writing_tar(compressed) as archive,
    ):

        def on_image(data_name, name, data, image_type):
            _add(archive, f"{data_name}/images/{name}", data, image_type)
            compressed.flush()

        try:
            with redirect_stdout_to_stderr(stderr):
                result = run(on_image)
            _write_parallel_streaming_result_entries(archive, result)
        except Exception as exc:
            logger.exception("pipeline execution failed")
            exit_with_error_streaming(
                ExitCodes.PROCESSING_ERROR, str(exc), archive, stderr, exc
            )
    sys.exit(0)
