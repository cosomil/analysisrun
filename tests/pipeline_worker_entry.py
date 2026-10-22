"""Entrypoint shared by subprocess integration tests; pickle types are importable."""

from __future__ import annotations

import json
import os
import pickle
import signal
import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from time import monotonic, sleep
from typing import Any, NamedTuple

import matplotlib

from analysisrun import MapArgs, PreprocessArgs, SynthesizeArgs
from analysisrun.env import ExecutionMode
from analysisrun.pipeline import Stage

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysisrun import pipeline_worker as transport
from analysisrun.pipeline_entrypoint import execute_entrypoint


class Inputs(NamedTuple):
    cells: pd.DataFrame


class Prepared(NamedTuple):
    cells: pd.DataFrame
    spots: pd.DataFrame


def wait_for(path: Path):
    deadline = monotonic() + 15
    while not path.exists():
        if monotonic() >= deadline:
            raise TimeoutError(f"waiting for {path}")
        sleep(0.01)


def prepare(args: PreprocessArgs[Any, Inputs]) -> Prepared:
    if args.params.get("fault") == "preprocess":
        raise LookupError("injected preprocess failure")
    directory = Path(args.params["directory"])
    with (directory / "preprocess").open("a") as file:
        file.write(f"{os.getpid()}\n")
    args.image_analysis_results.cells["score"] += 10
    return Prepared(
        args.image_analysis_results.cells,
        args.image_analysis_results.cells.copy(deep=True),
    )


def first(
    args: MapArgs[Any, Prepared, None, dict[str, Any]],
) -> dict[str, Callable[[], int]]:
    print("worker diagnostic")
    directory = Path(args.params["directory"])
    lane = args.data_name
    mode = args.params.get("fault", "")
    if mode == "kill-peer" and lane == "2":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (directory / f"started-{lane}").touch()
    if mode in {"parallel", "fail-peer", "kill-peer", "output-fail"} and lane in {
        "0",
        "2",
    }:
        wait_for(directory / f"started-{'2' if lane == '0' else '0'}")
    if mode in {"fail-peer", "kill-peer"}:
        if lane == "0":
            raise LookupError("injected map failure")
        if lane == "2":
            # Must be killed; never proceed to synthesize/finalize.
            sleep(60)
    if mode == "stderr":
        sys.stderr.write("diagnostic\n" * 10000)
        sys.stderr.flush()
    emit_image(args)
    if mode == "duplicate-image":
        emit_image(args)
    if mode == "fail-after-image" and lane == "0":
        raise LookupError("injected map failure")
    if mode == "image-ack":
        wait_for(directory / f"imageack-{lane}")
    if mode == "output-fail" and lane == "2":
        sleep(60)
    score = int(args.image_analysis_results.cells.score.iloc[0])
    args.image_analysis_results.cells["score"] = -1
    args.image_analysis_results.spots["score"] = -1
    args.params["history"].append("mutated")
    if os.environ.get("ANALYSISRUN_PROCESS_ROLE") == "parallel-worker":
        args.synthesized["history"].append("mutated")
    # Consecutive maps must not pickle this intermediate lambda.
    return {"get_score": lambda: score}


def emit_image(args: MapArgs[Any, Any, Any, Any]) -> None:
    fig, ax = plt.subplots()
    lane = args.data_name
    ax.plot([0, 1], [0, int(lane)])
    name = (
        "common.png"
        if args.params.get("fault") == "common-image"
        else f"lane-{lane}.png"
    )
    args.output(fig, name, "qc", format="png")


def second(
    args: MapArgs[Any, Prepared, dict[str, Callable[[], int]], dict[str, Any]],
) -> int | Callable[[], None]:
    if args.params.get("fault") == "consecutive-image":
        emit_image(args)
    if args.params.get("fault") == "unpicklable":
        return lambda: None
    score = args.value["get_score"]()
    if (
        args.params["history"]
        or args.image_analysis_results.cells.score.iloc[0] != score
        or args.image_analysis_results.spots.score.iloc[0] != score
        or args.synthesized["history"]
    ):
        raise ValueError("map isolation failed")
    directory = Path(args.params["directory"])
    (directory / f"finished-{args.data_name}").touch()
    return score * 2


def initial(args: SynthesizeArgs[Any, Prepared, None, None]) -> dict[str, Any]:
    return {"history": []}


def synthesize(
    args: SynthesizeArgs[Any, Prepared, int, dict[str, Any]],
) -> dict[str, Any]:
    if args.params.get("fault") == "synthesize":
        raise LookupError("injected synthesize failure")
    directory = Path(args.params["directory"])
    for lane in args.values:
        if not (directory / f"finished-{lane}").exists():
            raise ValueError("barrier passed before lane completion")
    (directory / "synthesize").touch()
    return {"total": sum(args.values.values()), "history": []}


def third(args: MapArgs[Any, Prepared, int, dict[str, Any]]) -> int:
    if args.params.get("fault") == "later-image":
        emit_image(args)
    directory = Path(args.params["directory"])
    if not (directory / "synthesize").exists() or args.synthesized["history"]:
        raise ValueError("invalid synthesized state")
    if os.environ.get("ANALYSISRUN_PROCESS_ROLE") == "parallel-worker":
        args.synthesized["history"].append("mutated")
    # Parent must receive the first lane before the child can finish the next.
    if args.params.get("fault") == "ack" and args.data_name == "1":
        wait_for(directory / "ack-0")
    return args.value + args.synthesized["total"]


def finalize(args: SynthesizeArgs[Any, Prepared, int, dict[str, Any]]) -> pd.DataFrame:
    if args.params.get("fault") == "finalize":
        raise LookupError("injected finalize failure")
    (Path(args.params["directory"]) / "finalize").touch()
    names = list(args.values)
    if args.params.get("fault") == "duplicate-result":
        names = ["same"] * len(names)
    return pd.DataFrame({"data": names, "score": list(args.values.values())})


def other(args: SynthesizeArgs[Any, Any, Any, Any]):
    raise RuntimeError("mismatched definition must not execute")


def definition():
    stages = [
        Stage("preprocess", prepare),
        Stage("synthesize", initial),
        Stage("map", first),
        Stage("map", second),
        Stage("synthesize", synthesize),
        Stage("map", third),
        Stage("finalize", finalize),
    ]
    if os.environ.get(
        "ANALYSISRUN_PROCESS_ROLE"
    ) == "parallel-worker" and os.environ.get("TEST_MISMATCH"):
        stages[-1] = Stage("finalize", other)
    return tuple(stages)


def inputs():
    return Inputs(pd.DataFrame({"Data": [str(i) for i in range(4)], "score": range(4)}))


def malformed_response(fault: str):
    entries = {}
    transport._read_entries(
        sys.stdin.buffer, lambda member, data: entries.update({member.name: data})
    )
    request = json.loads(entries["request.json"])
    lanes = request["lanes"]
    buffer = BytesIO()
    with transport.writing_tar(buffer) as archive:
        first = lanes[0]["lane_id"]
        transport._add(archive, f"lanes/{first}/value.pkl", pickle.dumps(first))
        if fault == "duplicate":
            transport._add(archive, f"lanes/{first}/value.pkl", pickle.dumps(first))
        elif fault == "unknown":
            transport._add(archive, "lanes/999/value.pkl", pickle.dumps(None))
        elif fault != "missing":
            for lane in lanes[1:]:
                data = b"broken" if fault == "corrupt" else pickle.dumps(None)
                transport._add(archive, f"lanes/{lane['lane_id']}/value.pkl", data)
    data = buffer.getvalue()
    if fault == "truncated":
        data = data[:2048]
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    sys.exit(7 if fault == "nonzero" else 0)


def execution_mode(value: str | None) -> ExecutionMode | None:
    match value:
        case None | "notebook" | "interactive" | "streaming" | "showschema":
            return value
        case _:
            raise ValueError(f"invalid test mode: {value!r}")


def main():
    stages = definition()
    if os.environ.get("ANALYSISRUN_PROCESS_ROLE") == "parallel-worker":
        if fault := os.environ.get("TEST_RESPONSE_FAULT"):
            malformed_response(fault)
        execute_entrypoint(stages, entrypoint=Path(__file__))
    mode = execution_mode(os.environ.get("TEST_EXECUTION_MODE"))
    execute_entrypoint(
        stages,
        entrypoint=Path(__file__),
        mode=mode,
        params={
            "directory": os.environ["TEST_DIRECTORY"],
            "history": [],
            "fault": os.environ.get("TEST_FAULT", ""),
        },
        image_analysis_results=inputs(),
        targets={str(i): f"sample {i}" for i in range(4)},
        output_dir=Path(os.environ["TEST_DIRECTORY"]) / "images",
    )


if __name__ == "__main__":
    # Ensure pickled NamedTuple classes retain their importable identity.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.pipeline_worker_entry import main as entry

    entry()
