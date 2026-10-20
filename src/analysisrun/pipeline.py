"""Internal Pipeline engine; public builders and ExecutionContext wiring come later.

Executors must finish (including worker collection) before execute returns or raises.
The runner keeps its snapshot alive until then. Values and synthesized objects are
passed by reference; only inputs and params are isolated for each user call.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

from .pipeable import Output, PreprocessArgs
from .typing import NamedTupleLike

type Inputs = NamedTupleLike[pd.DataFrame]


@dataclass(frozen=True)
class Stage:
    kind: Literal["preprocess", "map", "synthesize", "finalize"]
    func: Callable[[Any], Any]


@dataclass(frozen=True)
class Lane:
    lane_id: int
    data_name: str
    sample_name: str


@dataclass(frozen=True)
class MapSegment:
    """Half-open absolute range in the complete stage definition."""

    start: int
    end: int
    stages: tuple[Stage, ...]


@dataclass
class MapArgs:
    params: Any
    data_name: str
    sample_name: str
    image_analysis_results: Inputs
    value: Any
    synthesized: Any
    output: Output


@dataclass
class SynthesizeArgs:
    params: Any
    image_analysis_results: Mapping[str, Inputs]
    values: Mapping[str, Any]
    synthesized: Any


class InputSnapshot(Protocol):
    """Parent-side interface; file-backed implementations need no restore(lane)."""

    def restore_all(self) -> Mapping[str, Inputs]: ...


@runtime_checkable
class RestorableInputSnapshot(InputSnapshot, Protocol):
    def restore(self, lane_id: int) -> Inputs: ...


type SnapshotFactory = Callable[
    [Inputs, tuple[Lane, ...]], AbstractContextManager[InputSnapshot]
]


def _copy_inputs(inputs: Inputs) -> Inputs:
    return type(inputs)(*(frame.copy(deep=True) for frame in inputs))


class DataFrameInputSnapshot:
    """Copy DataFrames, preserving NamedTuple type, but not nested object cells."""

    def __init__(self, inputs: Inputs, lanes: tuple[Lane, ...]):
        self._lanes = lanes
        self._inputs: dict[int, Inputs] = {}
        names = [lane.data_name for lane in lanes]
        sources = {}
        for name, frame in inputs._asdict().items():
            if "Data" not in frame.columns:
                raise ValueError(f"input {name!r}: missing required 'Data' column")
            selected = frame.loc[frame["Data"].isin(names)]
            sources[name] = {
                data_name: data
                for data_name, data in selected.groupby(
                    "Data", sort=False, observed=True
                )
            }
        for lane in lanes:
            frames = []
            for name, source in sources.items():
                if lane.data_name not in source:
                    raise ValueError(
                        f"missing lane: {lane.data_name!r} in input {name!r}"
                    )
                frames.append(source[lane.data_name])
            self._inputs[lane.lane_id] = _copy_inputs(type(inputs)(*frames))

    def restore(self, lane_id: int) -> Inputs:
        return _copy_inputs(self._inputs[lane_id])

    def restore_all(self) -> Mapping[str, Inputs]:
        return MappingProxyType(
            {lane.data_name: self.restore(lane.lane_id) for lane in self._lanes}
        )


@contextmanager
def dataframe_snapshot(inputs: Inputs, lanes: tuple[Lane, ...]):
    yield DataFrameInputSnapshot(inputs, lanes)


@dataclass(frozen=True)
class MapExecutionContext:
    snapshot: InputSnapshot
    lanes: tuple[Lane, ...]
    params: Any
    values: Mapping[str, Any]
    synthesized: Any
    output: Output


class MapExecutor(Protocol):
    def execute(
        self,
        segment: MapSegment,
        context: MapExecutionContext,
        on_value: Callable[[int, Any], None],
    ) -> None: ...


class SequentialMapExecutor:
    def execute(
        self,
        segment: MapSegment,
        context: MapExecutionContext,
        on_value: Callable[[int, Any], None],
    ) -> None:
        snapshot = context.snapshot
        if not isinstance(snapshot, RestorableInputSnapshot):
            raise TypeError("sequential execution requires restore(lane_id)")
        for lane in context.lanes:
            value = context.values[lane.data_name]
            for stage in segment.stages:
                value = stage.func(
                    MapArgs(
                        deepcopy(context.params),
                        lane.data_name,
                        lane.sample_name,
                        snapshot.restore(lane.lane_id),
                        value,
                        context.synthesized,
                        context.output,
                    )
                )
            on_value(lane.lane_id, value)


def _validate_stages(stages: tuple[Stage, ...]) -> None:
    if not stages or stages[-1].kind != "finalize":
        raise ValueError("pipeline must end with finalize")
    for index, stage in enumerate(stages):
        if stage.kind not in {"preprocess", "map", "synthesize", "finalize"}:
            raise ValueError(f"unsupported stage kind: {stage.kind!r}")
        if stage.kind == "preprocess" and index != 0:
            raise ValueError("preprocess must be first and unique")
        if stage.kind == "finalize" and index != len(stages) - 1:
            raise ValueError("finalize must be last and unique")
        if not callable(stage.func):
            raise TypeError(f"stage {index} requires a callable")


def execute_pipeline(
    stages: Sequence[Stage],
    *,
    params: Any,
    image_analysis_results: Inputs,
    targets: Mapping[str, str],
    output: Output,
    map_executor: MapExecutor,
    snapshot_factory: SnapshotFactory,
) -> pd.DataFrame:
    """Run a resolved definition with independently injected execution/storage.

    on_value replaces only completed lane results, synchronously. A failure stops
    progression without rolling back completed results, images, or mutations to
    referenced value objects. Executors may not change the lane set or order.
    """
    stages = tuple(stages)
    _validate_stages(stages)
    targets = MappingProxyType(dict(targets))
    lanes = tuple(
        Lane(i, name, sample) for i, (name, sample) in enumerate(targets.items())
    )
    values = dict.fromkeys(targets)
    synthesized = None
    inputs = image_analysis_results
    index = 0
    if stages[0].kind == "preprocess":
        inputs = stages[0].func(
            PreprocessArgs(deepcopy(params), _copy_inputs(inputs), targets)
        )
        index = 1

    def on_value(lane_id: int, value: Any) -> None:
        if type(lane_id) is not int or not 0 <= lane_id < len(lanes):
            raise ValueError(f"unknown lane_id: {lane_id!r}")
        values[lanes[lane_id].data_name] = value

    with snapshot_factory(inputs, lanes) as snapshot:
        while index < len(stages):
            stage = stages[index]
            if stage.kind == "map":
                end = index + 1
                while end < len(stages) and stages[end].kind == "map":
                    end += 1
                map_executor.execute(
                    MapSegment(index, end, stages[index:end]),
                    MapExecutionContext(
                        snapshot,
                        lanes,
                        params,
                        MappingProxyType(values),
                        synthesized,
                        output,
                    ),
                    on_value,
                )
                index = end
                continue
            args = SynthesizeArgs(
                deepcopy(params),
                MappingProxyType(dict(snapshot.restore_all())),
                MappingProxyType(values),
                synthesized,
            )
            result = stage.func(args)
            if stage.kind == "finalize":
                if not isinstance(result, pd.DataFrame):
                    raise TypeError("finalize must return a pandas DataFrame")
                return result
            synthesized = result
            index += 1
    raise RuntimeError("pipeline ended without finalize")


def execute_sequential(
    stages: Sequence[Stage],
    *,
    params: Any,
    image_analysis_results: Inputs,
    targets: Mapping[str, str],
    output: Output,
) -> pd.DataFrame:
    return execute_pipeline(
        stages,
        params=params,
        image_analysis_results=image_analysis_results,
        targets=targets,
        output=output,
        map_executor=SequentialMapExecutor(),
        snapshot_factory=dataframe_snapshot,
    )
