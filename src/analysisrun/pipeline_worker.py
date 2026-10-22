"""Trusted, same-environment Pipeline worker transport (internal API).

Workers reconstruct definitions through the original entrypoint. Never load these
pickle archives from an untrusted source. Only the main thread publishes events.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import tarfile
import traceback
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from queue import Empty, Full, Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import MappingProxyType
from typing import IO, Any, Literal, Protocol, TypedDict, overload

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import figure

from .pipeline import (
    DataFrameInputSnapshot,
    Inputs,
    Lane,
    MapExecutionContext,
    MapSegment,
    Stage,
    _validate_stages,
    execute_map_segment,
)
from .tar import TarEntryWriter

logger = logging.getLogger(__name__)

# data_name, image_name, encoded image, image_type
ImageSink = Callable[[str, str, bytes, str | None], None]
type ImagePayload = tuple[str, bytes, str | None]
type ResponseEvent = (
    tuple[Literal["value"], int, Any] | tuple[Literal["image"], int, ImagePayload]
)
type WorkerEvent = (
    ResponseEvent | tuple[Literal["error"], TransferError] | tuple[Literal["done"]]
)


class ResponseSink(Protocol):
    @overload
    def __call__(
        self, kind: Literal["value"], lane_id: int, payload: Any, /
    ) -> None: ...

    @overload
    def __call__(
        self, kind: Literal["image"], lane_id: int, payload: ImagePayload, /
    ) -> None: ...


class WorkerLane(TypedDict):
    lane_id: int
    data_name: str
    sample_name: str
    input_path: str


class WorkerRequest(TypedDict):
    protocol_version: int
    manifest: str
    stage_start: int
    stage_end: int
    lanes: list[WorkerLane]


class _ReadableBinary(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class _WritableBinary(Protocol):
    def write(self, data: Buffer, /) -> int: ...
    def flush(self) -> None: ...


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
    def identity(func: Callable[..., Any]) -> list[str]:
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
    def __init__(self, lanes: tuple[Lane, ...], paths: Mapping[int, Path]) -> None:
        self.lanes = lanes
        self.paths = MappingProxyType(dict(paths))

    def restore(self, lane_id: int) -> Inputs:
        return pickle.loads(self.paths[lane_id].read_bytes())

    def restore_all(self) -> Mapping[str, Inputs]:
        return MappingProxyType(
            {lane.data_name: self.restore(lane.lane_id) for lane in self.lanes}
        )


@contextmanager
def file_snapshot(
    inputs: Inputs, lanes: tuple[Lane, ...]
) -> Iterator[FileInputSnapshot]:
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

    def __init__(self, stream: _WritableBinary) -> None:
        self.stream = stream
        self.position = 0

    def write(self, data: Buffer, /) -> int:
        written = self.stream.write(data)
        self.position += written
        return written

    def tell(self) -> int:
        return self.position

    def flush(self) -> None:
        self.stream.flush()

    def read(self, size: int = -1, /) -> bytes:
        raise UnsupportedOperation("write-only TAR stream")

    def seek(self, offset: int, whence: int = 0, /) -> int:
        raise UnsupportedOperation("non-seekable TAR stream")

    def close(self) -> None:
        # This adapter borrows the stream; the caller owns its lifetime.
        self.flush()


@contextmanager
def writing_tar(stream: _WritableBinary) -> Iterator[tarfile.TarFile]:
    writer = _CountingWriter(stream)
    with tarfile.open(fileobj=writer, mode="w") as archive:
        try:
            yield archive
        finally:
            # TarFile.__exit__ omits end markers on exceptions. Explicitly close
            # first: our protocol requires end markers even after SystemExit.
            archive.close()
            writer.flush()


def _add(
    archive: TarEntryWriter, name: str, data: bytes, image_type: str | None = None
) -> None:
    _add_tar_binary_entry(archive, name, data, image_type=image_type)


class _UniqueEntries:
    """Enforce uniqueness across images, final results, and the error entry."""

    def __init__(self, archive: TarEntryWriter) -> None:
        self.archive = archive
        self.names: set[str] = set()

    def addfile(
        self, tarinfo: tarfile.TarInfo, fileobj: IO[bytes] | None = None
    ) -> None:
        if tarinfo.name in self.names:
            raise TransferError(f"duplicate output TAR entry: {tarinfo.name}")
        self.names.add(tarinfo.name)
        self.archive.addfile(tarinfo, fileobj)


class _TrackedReader:
    def __init__(self, stream: _ReadableBinary) -> None:
        self.stream = stream
        self.size = 0
        self.tail = b""

    def read(self, size: int = -1, /) -> bytes:
        data = self.stream.read(size)
        self.size += len(data)
        self.tail = (self.tail + data)[-1024:]
        return data

    def tell(self) -> int:
        return self.size

    def write(self, data: Buffer, /) -> int:
        raise UnsupportedOperation("read-only TAR stream")

    def seek(self, offset: int, whence: int = 0, /) -> int:
        raise UnsupportedOperation("non-seekable TAR stream")

    def close(self) -> None:
        # The caller, rather than this adapter, owns the underlying stream.
        pass


def _read_entries(
    stream: _ReadableBinary, on_entry: Callable[[tarfile.TarInfo, bytes], None]
) -> None:
    reader = _TrackedReader(stream)
    with tarfile.open(fileobj=reader, mode="r|", bufsize=512) as archive:
        for member in archive:
            if not member.isfile():
                raise TransferError(f"unexpected non-file entry: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise TransferError(f"cannot extract entry: {member.name}")
            on_entry(member, extracted.read())
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


def _validate_request(request: WorkerRequest, stages: Sequence[Stage]) -> MapSegment:
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
    ids: set[int] = set()
    names: set[str] = set()
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
    def __init__(
        self, archive: TarEntryWriter, stream: _WritableBinary, lane_id: int
    ) -> None:
        self.archive, self.stream, self.lane_id = archive, stream, lane_id

    def __call__(
        self, fig: figure.Figure, name: str, image_type: str, **kwargs: Any
    ) -> None:
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


def run_worker(stages: Sequence[Stage], stdin: IO[bytes], stdout: IO[bytes]) -> int:
    """Read the complete request before writing any response bytes."""
    entries: dict[str, bytes] = {}
    lane: WorkerLane | None = None
    stage_index: int | None = None

    def collect(member: tarfile.TarInfo, data: bytes) -> None:
        if member.name in entries:
            raise TransferError(f"duplicate request entry: {member.name}")
        entries[member.name] = data

    # Even request failures produce a structured response, after stdin is drained.
    try:
        _read_entries(stdin, collect)
        request: WorkerRequest = json.loads(entries["request.json"])
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

                def before_stage(index: int) -> None:
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


def _write_error(
    archive: TarEntryWriter,
    exc: Exception,
    lane: WorkerLane | None,
    stage_index: int | None,
) -> None:
    error = {
        "protocol_version": 1,
        "lane_id": lane["lane_id"] if lane else None,
        "data_name": lane["data_name"] if lane else None,
        "sample_name": lane["sample_name"] if lane else None,
        "stage_kind": "map" if stage_index is not None else None,
        "stage_index": stage_index,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    _add(archive, "error.json", json.dumps(error, ensure_ascii=False).encode())


def receive_response(
    stream: _ReadableBinary,
    request: WorkerRequest,
    emit: ResponseSink,
) -> None:
    expected = {lane["lane_id"]: lane for lane in request["lanes"]}
    seen: set[int] = set()

    def receive(member: tarfile.TarInfo, data: bytes) -> None:
        if member.name == "error.json":
            error = json.loads(data)
            if error["protocol_version"] != 1:
                raise TransferError("response protocol_version mismatch")
            exc = TransferError(json.dumps(error, ensure_ascii=False))
            exc.add_note(
                f"pipeline stage=map stage_index={error.get('stage_index')} "
                f"lane_id={error.get('lane_id')} "
                f"data_name={error.get('data_name')!r} "
                f"sample_name={error.get('sample_name')!r}"
            )
            raise exc
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


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
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
    ) -> None:
        self.manifest = canonical_manifest(stages)
        self.entrypoint = Path(entrypoint).resolve()
        self.on_image = on_image
        self.max_workers = worker_limit() if max_workers is None else max_workers
        if type(self.max_workers) is not int or self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if type(queue_size) is not int or queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")
        self.queue_size = queue_size

    def execute(
        self,
        segment: MapSegment,
        context: MapExecutionContext,
        on_value: Callable[[int, Any], None],
    ) -> None:
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
        requests: list[tuple[WorkerRequest, dict[int, bytes]]] = []
        for chunk in chunks:
            request_lanes: list[WorkerLane] = [
                {
                    "lane_id": lane.lane_id,
                    "data_name": lane.data_name,
                    "sample_name": lane.sample_name,
                    "input_path": str(context.snapshot.paths[lane.lane_id]),
                }
                for lane in chunk
            ]
            request: WorkerRequest = {
                "protocol_version": 1,
                "manifest": self.manifest,
                "stage_start": segment.start,
                "stage_end": segment.end,
                "lanes": request_lanes,
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

        events: Queue[WorkerEvent] = Queue(maxsize=self.queue_size)
        stopped = Event()
        processes: list[subprocess.Popen[bytes]] = []

        def emit(event: WorkerEvent) -> None:
            while not stopped.is_set():
                try:
                    events.put(event, timeout=0.05)
                    return
                except Full:
                    pass

        def emit_response(
            kind: Literal["value", "image"], lane_id: int, payload: Any
        ) -> None:
            if kind == "value":
                emit(("value", lane_id, payload))
            else:
                emit(("image", lane_id, payload))

        def manage(
            proc: subprocess.Popen[bytes],
            request: WorkerRequest,
            values: Mapping[int, bytes],
        ) -> None:
            stdin, stdout, stderr = proc.stdin, proc.stdout, proc.stderr
            if stdin is None or stdout is None or stderr is None:
                raise TransferError("worker requires stdin, stdout, and stderr pipes")
            diagnostics = bytearray()

            def drain() -> None:
                while data := stderr.read(65536):
                    diagnostics.extend(data)

            drainer = Thread(target=drain)
            drainer.start()
            error: Exception | None = None
            try:
                with stdin, writing_tar(stdin) as archive:
                    _add(archive, "request.json", json.dumps(request).encode())
                    _add(archive, "params.pkl", params)
                    _add(archive, "synthesized.pkl", synthesized)
                    for lane_id, data in values.items():
                        _add(archive, f"lanes/{lane_id}/value.pkl", data)
                receive_response(stdout, request, emit_response)
                if proc.wait() != 0:
                    raise TransferError(f"worker exited with code {proc.returncode}")
            except Exception as exc:
                logger.exception("worker communication failed")
                error = exc
                _stop_process(proc)
            finally:
                proc.wait()
                drainer.join()
                stdout.close()
                stderr.close()
            if error:
                wrapped = TransferError(
                    f"worker failure: stage=map stage_start={segment.start} "
                    f"stage_end={segment.end} lanes={request['lanes']!r}; "
                    f"{type(error).__name__}: {error}; exit={proc.returncode}; "
                    f"stderr={diagnostics.decode(errors='replace')}"
                )
                wrapped.__cause__ = error
                for note in getattr(error, "__notes__", ()):
                    wrapped.add_note(note)
                if not getattr(wrapped, "__notes__", ()):
                    wrapped.add_note(
                        f"pipeline stage=map stage_start={segment.start} "
                        f"stage_end={segment.end} "
                        + ", ".join(
                            f"lane_id={lane['lane_id']} "
                            f"data_name={lane['data_name']!r} "
                            f"sample_name={lane['sample_name']!r}"
                            for lane in request["lanes"]
                        )
                    )
                emit(("error", wrapped))
            else:
                emit(("done",))

        pool = ThreadPoolExecutor(max_workers=len(chunks))
        futures: list[Future[None]] = []
        try:
            env = os.environ | {"ANALYSISRUN_PROCESS_ROLE": "parallel-worker"}
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
                if event[0] == "error":
                    raise event[1]
                if event[0] == "done":
                    remaining -= 1
                elif event[0] == "value":
                    on_value(event[1], event[2])
                elif event[0] == "image":
                    lane_id, (name, data, image_type) = event[1], event[2]
                    try:
                        self.on_image(lanes[lane_id].data_name, name, data, image_type)
                    except Exception as exc:
                        logger.exception("pipeline image output failed")
                        lane = lanes[lane_id]
                        exc.add_note(
                            f"pipeline stage=map stage_start={segment.start} "
                            f"stage_end={segment.end} lane_id={lane_id} "
                            f"data_name={lane.data_name!r} "
                            f"sample_name={lane.sample_name!r}"
                        )
                        raise
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
                        if stream is not None:
                            stream.close()


def _split_evenly_in_order[T](items: list[T], chunks: int) -> list[list[T]]:
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    if not items:
        return []

    chunk_count = min(chunks, len(items))
    base, extra = divmod(len(items), chunk_count)

    result: list[list[T]] = []
    start = 0
    for idx in range(chunk_count):
        size = base + (1 if idx < extra else 0)
        end = start + size
        result.append(items[start:end])
        start = end
    return result


def _add_tar_binary_entry(
    tar: TarEntryWriter,
    name: str,
    content: bytes,
    *,
    image_type: str | None = None,
) -> None:
    tar_info = tarfile.TarInfo(name=name)
    tar_info.size = len(content)
    pax_headers = {"is_file": "true"}
    if image_type is not None:
        pax_headers["image_type"] = image_type
    tar_info.pax_headers = pax_headers
    tar.addfile(tar_info, BytesIO(content))


def _write_streaming_result_entries(
    tar: TarEntryWriter,
    result_df: pd.DataFrame,
) -> None:
    csv_buf = BytesIO()
    result_df = result_df.astype(str)
    result_df.to_csv(csv_buf, index=False)
    _add_tar_binary_entry(tar, "result_csv", csv_buf.getvalue())

    for idx, row in result_df.iterrows():
        data_name = row.get("data", idx)
        json_bytes = row.to_json().encode("utf-8")
        _add_tar_binary_entry(tar, f"{data_name!s}/result_json", json_bytes)
