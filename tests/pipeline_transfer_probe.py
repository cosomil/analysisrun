"""#82 の独立した転送実験。本番 Executor ではなく、信頼済み pickle 専用。"""

import copy
import json
import logging
import pickle
import subprocess
import sys
import tarfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread, Timer
from time import monotonic, sleep
from types import MappingProxyType
from typing import NamedTuple

import pandas as pd

from tests.pipeline_probe import Stage, canonical_manifest

logger = logging.getLogger(__name__)


class Inputs(NamedTuple):
    cells: pd.DataFrame


class Prepared(NamedTuple):
    cells: pd.DataFrame
    spots: pd.DataFrame


@dataclass
class Feature:
    score: int


class TransferError(RuntimeError):
    pass


def dumps(value, label):
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise TransferError(
            f"cannot pickle {label}: {type(exc).__name__}: {exc}"
        ) from exc


def copy_inputs(inputs):
    return type(inputs)(*(frame.copy(deep=True) for frame in inputs))


def prepare(inputs):
    inputs.cells["score"] += 10
    return Prepared(inputs.cells, inputs.cells.copy(deep=True))


def split(inputs, targets):
    lanes = {}
    for name in targets:
        frames = [frame.loc[frame["lane"] == name].copy(deep=True) for frame in inputs]
        if any(frame.empty for frame in frames):
            raise ValueError(f"missing lane: {name}")
        lanes[name] = type(inputs)(*frames)
    return lanes


@contextmanager
def snapshots(inputs, targets, preprocess=None):
    """実行開始時に確定し、呼出元が子を回収してから context を抜ける。"""
    transformed = preprocess(copy_inputs(inputs)) if preprocess else inputs
    with TemporaryDirectory(prefix="pipeline-transfer-") as directory:
        lanes = []
        for lane_id, (name, lane) in enumerate(split(transformed, targets).items()):
            path = Path(directory) / f"{lane_id}.pkl"
            path.write_bytes(dumps(lane, f"input lane={lane_id} data_name={name}"))
            lanes.append(
                {
                    "lane_id": lane_id,
                    "data_name": name,
                    "sample_name": targets[name],
                    "input_path": str(path),
                }
            )
        yield lanes


def restore_all(lanes):
    return MappingProxyType(
        {
            lane["data_name"]: pickle.loads(Path(lane["input_path"]).read_bytes())
            for lane in lanes
        }
    )


def parent_args(lanes, params, values, synthesized):
    return {
        "inputs": restore_all(lanes),
        "params": copy.deepcopy(params),
        "values": MappingProxyType(values),
        "synthesized": synthesized,
    }


def notebook_snapshot(inputs, targets, preprocess=None):
    transformed = preprocess(copy_inputs(inputs)) if preprocess else inputs
    return {
        name: copy_inputs(lane) for name, lane in split(transformed, targets).items()
    }


def notebook_restore(snapshot):
    return MappingProxyType(
        {name: copy_inputs(lane) for name, lane in snapshot.items()}
    )


def first_map(args):
    # ローカル lambda を含む中間値は pickle 不可能。次 map に直接渡す。
    score = int(args["inputs"].cells["score"].iloc[0])
    args["inputs"].cells["score"] = -1
    args["inputs"].spots["score"] = -1
    args["params"]["history"].append("changed")
    args["synthesized"]["history"].append("changed")
    args["output"](b"first image", "first.png")
    return {"score": score, "identity": lambda: score, "history": ["first"]}


def second_map(args):
    value = args["value"]
    if (
        int(args["inputs"].cells["score"].iloc[0]) != value["identity"]()
        or int(args["inputs"].spots["score"].iloc[0]) != value["score"]
        or args["params"]["history"]
        or args["synthesized"]["history"]
    ):
        raise ValueError("stage isolation failed")
    value["history"].append("second")
    args["output"](b"second image", "second.png")
    return Feature(value["score"] + len(value["history"]))


def echo_map(args):
    return args["value"]


def none_map(args):
    return None


def fail_map(args):
    raise LookupError(f"failed {args['data_name']}")


def unpicklable_map(args):
    return lambda: None


def unused_parent(args):
    raise RuntimeError("parent stage must not execute in child")


def definition():
    # preprocess / synthesize も含む全 Manifest を比較する。
    return (
        Stage("preprocess", prepare),
        Stage("synthesize", unused_parent),
        Stage("map", first_map),
        Stage("map", second_map),
        Stage("synthesize", unused_parent),
        Stage("map", echo_map),
        Stage("map", fail_map),
        Stage("map", unpicklable_map),
        Stage("map", none_map),
        Stage("finalize", unused_parent),
    )


def request_for(lanes, start=2, end=4):
    return {
        "protocol_version": 1,
        "manifest": canonical_manifest(definition()),
        "stage_start": start,
        "stage_end": end,
        "lanes": lanes,
    }


def validate(request):
    if request["protocol_version"] != 1:
        raise ValueError("protocol_version mismatch")
    stages = definition()
    if request["manifest"] != canonical_manifest(stages):
        raise ValueError("pipeline manifest mismatch")
    start, end = request["stage_start"], request["stage_end"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(stages)
    ):
        raise ValueError("invalid map segment range")
    if any(stage.kind != "map" for stage in stages[start:end]):
        raise ValueError("segment contains non-map stage")
    ids = [lane["lane_id"] for lane in request["lanes"]]
    names = [lane["data_name"] for lane in request["lanes"]]
    if (
        any(type(i) is not int or i < 0 for i in ids)
        or len(set(ids)) != len(ids)
        or len(set(names)) != len(names)
    ):
        raise ValueError("invalid lane set")
    return stages


def add(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, BytesIO(data))


def flush(tar, stream):
    # tarfile w| は内部で record をバッファする。レーン/画像の到着を
    # 子終了より前に観測するため、残った block をここで送り出す。
    tar.fileobj.fileobj.write(tar.fileobj.buf)
    tar.fileobj.buf = b""
    stream.flush()


def write_request(stream, request, params, synthesized, values):
    with tarfile.open(fileobj=stream, mode="w|") as tar:
        add(tar, "request.json", json.dumps(request).encode())
        add(tar, "params.pkl", dumps(params, f"params stage={request['stage_start']}"))
        add(
            tar,
            "synthesized.pkl",
            dumps(synthesized, f"synthesized stage={request['stage_start']}"),
        )
        for lane in request["lanes"]:
            lane_id = lane["lane_id"]
            add(
                tar,
                f"lanes/{lane_id}/value.pkl",
                dumps(
                    values[lane_id],
                    f"value lane={lane_id} data_name={lane['data_name']} stage={request['stage_start']}",
                ),
            )


def child(stream_in, stream_out, fault="", ack=None):
    request = None
    lane = None
    stage_index = None
    with tarfile.open(fileobj=stream_out, mode="w|") as tar:
        try:
            entries = {}
            with tarfile.open(fileobj=stream_in, mode="r|*") as incoming:
                for member in incoming:
                    if member.name in entries:
                        raise ValueError(f"duplicate request entry: {member.name}")
                    entries[member.name] = incoming.extractfile(member).read()
            request = json.loads(entries["request.json"])
            stages = validate(request)
            # 故意に stderr pipe 容量を超えさせ、drain の成立を検証。
            if fault == "stderr":
                sys.stderr.write("diagnostic\n" * 10000)
                sys.stderr.flush()
            for lane in request["lanes"]:
                stage_index = None
                lane_id = lane["lane_id"]
                value = pickle.loads(entries[f"lanes/{lane_id}/value.pkl"])

                def output(data, name, lane_id=lane_id):
                    add(tar, f"lanes/{lane_id}/images/{name}", data)
                    flush(tar, stream_out)

                for stage_index in range(request["stage_start"], request["stage_end"]):
                    args = dict(
                        lane,
                        inputs=pickle.loads(Path(lane["input_path"]).read_bytes()),
                        params=pickle.loads(entries["params.pkl"]),
                        synthesized=pickle.loads(entries["synthesized.pkl"]),
                        value=value,
                        output=output,
                    )
                    value = stages[stage_index].func(args)
                if fault == "missing":
                    continue
                data = dumps(
                    value,
                    f"value lane={lane_id} data_name={lane['data_name']} stage={stage_index}",
                )
                add(tar, f"lanes/{lane_id}/value.pkl", data)
                flush(tar, stream_out)
                if fault == "duplicate":
                    add(tar, f"lanes/{lane_id}/value.pkl", data)
                if ack:
                    deadline = monotonic() + 10
                    while not Path(f"{ack}.{lane_id}").exists():
                        if monotonic() > deadline:
                            raise TimeoutError("parent did not observe streaming value")
                        sleep(0.01)
            if fault == "corrupt":
                stream_out.write(b"broken header" * 40)
            if fault == "nonzero":
                return 7
            if fault == "error":
                raise RuntimeError("injected error")
        except Exception as exc:
            logger.exception("child execution failed")
            error = {
                "protocol_version": 1,
                "lane_id": lane["lane_id"] if lane else None,
                "data_name": lane["data_name"] if lane else None,
                "stage_index": stage_index,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            add(tar, "error.json", json.dumps(error).encode())
            return 1
    return 0


class TrackedReader:
    """tarfile が許容する EOF marker 欠損も検出する小さな実験用 reader。"""

    def __init__(self, stream):
        self.stream = stream
        self.size = 0
        self.tail = b""

    def read(self, size=-1):
        data = self.stream.read(size)
        self.size += len(data)
        self.tail = (self.tail + data)[-1024:]
        return data


def receive(stream, request, values, on_event):
    expected = {lane["lane_id"] for lane in request["lanes"]}
    seen = set()
    reader = TrackedReader(stream)
    with tarfile.open(fileobj=reader, mode="r|*", bufsize=512) as tar:
        for member in tar:
            data = tar.extractfile(member).read()
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
                raise TransferError(f"unknown lane={lane_id}")
            if parts[2:] == ["value.pkl"]:
                if lane_id in seen:
                    raise TransferError(f"duplicate value.pkl lane={lane_id}")
                values[lane_id] = pickle.loads(data)
                seen.add(lane_id)
                on_event("value", lane_id, values[lane_id])
            elif parts[2] == "images" and len(parts) > 3:
                on_event("image", lane_id, data)
            else:
                raise TransferError(f"unexpected response entry: {member.name}")
    while reader.read(65536):
        pass
    if reader.size % 512 or reader.tail != bytes(1024):
        raise TransferError("corrupt TAR: missing end markers")
    if seen != expected:
        raise TransferError(f"missing value.pkl lanes={sorted(expected - seen)}")


def run_child(
    request,
    params,
    synthesized,
    values,
    on_event=lambda *args: None,
    fault="",
    ack=None,
    processes=None,
):
    """一子のみのハーネス。完了待ちの前に values を更新する。"""
    command = [sys.executable, "-m", "tests.pipeline_transfer_probe", fault]
    if ack:
        command.append(str(ack))
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if processes is not None:
        processes.append(proc)
    stderr = bytearray()
    write_errors = []

    def write():
        try:
            write_request(proc.stdin, request, params, synthesized, values)
        except Exception as exc:
            logger.exception("request transfer failed")
            write_errors.append(exc)
        finally:
            proc.stdin.close()

    def drain():
        stderr.extend(proc.stderr.read())

    writer = Thread(target=write)
    drainer = Thread(target=drain)
    writer.start()
    drainer.start()
    # 実験のハングは30秒で停止。製品の停止/回収方針は #84。
    watchdog = Timer(30, proc.kill)
    watchdog.start()
    error = None
    try:
        receive(proc.stdout, request, values, on_event)
    except Exception as exc:
        logger.exception("response transfer failed")
        error = exc
        proc.kill()
    finally:
        proc.wait()
        writer.join()
        drainer.join()
        watchdog.cancel()
        proc.stdout.close()
        proc.stderr.close()
    if write_errors:
        raise write_errors[0]
    if error or proc.returncode:
        raise TransferError(
            f"TAR/child failure: {error or 'child failed'}; exit={proc.returncode}; "
            f"stderr={stderr.decode(errors='replace')}"
        ) from error
    return stderr.decode()


if __name__ == "__main__":
    # -m 実行でも pickle 型の module は import 時と同じに固定する。
    from tests.pipeline_transfer_probe import child as entry

    sys.exit(
        entry(
            sys.stdin.buffer,
            sys.stdout.buffer,
            sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else None,
        )
    )
