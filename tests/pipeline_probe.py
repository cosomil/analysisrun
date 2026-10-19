"""#81 の独立実験。公開 API・本番の状態/転送形式ではない。

直接起動と import の双方で同じ関数識別子を得られるよう、実験用の
エントリポイントモジュールだけを正規化する。
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from analysisrun.tar import create_tar_from_dict, read_tar_as_dict

type State = dict
type StageFunc = Callable[[State], State]
type PrepareFunc = Callable[[State], int]


@dataclass(frozen=True)
class Stage:
    kind: str
    func: StageFunc | PrepareFunc


def callable_id(func: StageFunc | PrepareFunc) -> tuple[str, str]:
    module = func.__module__
    if module in {"__main__", __name__}:
        module = "<entrypoint>"
    return module, func.__qualname__


def canonical_manifest(stages: tuple[Stage, ...]) -> str:
    return json.dumps(
        {
            "version": 1,
            "stages": [
                {"kind": s.kind, "callable": callable_id(s.func)} for s in stages
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class Definition:
    stages: tuple[Stage, ...] = ()

    def append(self, kind: str, func: StageFunc | PrepareFunc) -> "Definition":
        if kind not in {"preprocess", "map", "synthesize"}:
            raise ValueError("unsupported stage kind")
        if kind == "preprocess" and self.stages:
            raise ValueError("preprocess must be first and unique")
        return Definition((*self.stages, Stage(kind, func)))

    def finalize(
        self,
        func: StageFunc,
        state: State,
        *,
        request: dict | None = None,
        calls: list[str] | None = None,
    ) -> dict:
        stages = (*self.stages, Stage("finalize", func))
        if request is None:
            # 逐次実行は Manifest を生成・検証する必要がない。
            selected = stages
        else:
            if request["manifest"] != canonical_manifest(stages):
                raise ValueError("pipeline manifest mismatch")
            index = request["stage_index"]
            if type(index) is not int or not 0 <= index < len(stages):
                raise ValueError("stage index out of range")
            if stages[index].kind != "map":
                raise ValueError("requested stage is not map")
            selected = (stages[index],)

        if calls is None:
            calls = []
        for stage in selected:
            calls.append(stage.func.__qualname__)
            if stage.kind == "preprocess":
                # ダミー元入力の変換結果を直接受け取り、他の状態は維持する。
                state = state | {"image_analysis_results": stage.func(state)}
            else:
                state = stage.func(state)
        return {"state": state, "calls": calls}


def prepare(state: State) -> int:
    return state["image_analysis_results"] + 1


def extract_features(state: State) -> State:
    return state | {"value": state["image_analysis_results"] * 2}


def evaluate_qc(state: State) -> State:
    return state | {"synthesized": state["value"] + state["image_analysis_results"]}


def apply_qc(state: State) -> State:
    return state | {
        "value": state["value"] + state["synthesized"] + state["image_analysis_results"]
    }


def other_map(state: State) -> State:
    raise RuntimeError("replacement callable must never execute")


def to_result(state: State) -> State:
    return state | {"result": state["value"]}


def other_finalize(state: State) -> State:
    raise RuntimeError("replacement finalize must never execute")


def build(with_preprocess: bool = True, mutation: str = "") -> Definition:
    definition = Definition()
    if with_preprocess:
        definition = definition.append("preprocess", prepare)
    definition = definition.append("map", extract_features)
    definition = definition.append("synthesize", evaluate_qc)
    definition = definition.append("map", apply_qc)
    stages = list(definition.stages)
    if mutation == "add":
        stages.append(Stage("map", extract_features))
    elif mutation == "delete":
        stages.pop()
    elif mutation == "reorder":
        stages[-1], stages[-2] = stages[-2], stages[-1]
    elif mutation == "kind":
        stages[-1] = Stage("synthesize", apply_qc)
    elif mutation == "callable":
        stages[-1] = Stage("map", other_map)
    return Definition(tuple(stages))


def run_child(request: dict, *, with_preprocess: bool = True, mutation: str = ""):
    """親から既存方式と同じ Python entrypoint + TAR/stdin で実行する。"""
    command = [sys.executable, str(Path(__file__).resolve())]
    if not with_preprocess:
        command.append("--without-preprocess")
    command.extend(["--mutation", mutation])
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        stdout, stderr = proc.communicate(
            create_tar_from_dict({"request": json.dumps(request)}).getvalue(),
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return (
        proc.returncode,
        json.loads(read_tar_as_dict(BytesIO(stdout))["response"]),
        stderr.decode(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-preprocess", action="store_true")
    parser.add_argument("--mutation", default="")
    args = parser.parse_args()
    # 再実行でも構築時にユーザー関数を呼ばず、finalize で初めて要求を扱う。
    definition = build(not args.without_preprocess, args.mutation)
    request = json.loads(read_tar_as_dict(sys.stdin.buffer)["request"])
    finalizer = other_finalize if args.mutation == "finalize" else to_result
    calls: list[str] = []
    try:
        result = definition.finalize(
            finalizer, request["state"], request=request, calls=calls
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        response = {"error": str(exc), "calls": calls}
        status = 1
    else:
        response = result | {
            "manifest": canonical_manifest(
                (*definition.stages, Stage("finalize", finalizer))
            ),
            "stage_index": request["stage_index"],
        }
        status = 0
    sys.stdout.buffer.write(
        create_tar_from_dict({"response": json.dumps(response)}).getvalue()
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
