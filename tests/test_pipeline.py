from contextlib import contextmanager
from types import SimpleNamespace
from typing import NamedTuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from analysisrun.pipeline import (
    DataFrameInputSnapshot,
    Lane,
    MapExecutionContext,
    MapSegment,
    SequentialMapExecutor,
    Stage,
    execute_pipeline,
    execute_sequential,
)


class Inputs(NamedTuple):
    cells: pd.DataFrame


class Prepared(NamedTuple):
    cells: pd.DataFrame
    spots: pd.DataFrame


@pytest.fixture
def inputs():
    return Inputs(
        pd.DataFrame(
            {
                "Data": ["a", "b", "unused"],
                "ImageAnalysisMethod": ["method"] * 3,
                "MultiPointIndex": [1] * 3,
                "score": [1, 2, 100],
                "object": [lambda: 1] * 3,
            }
        )
    )


def run(stages, inputs, **kwargs):
    return execute_sequential(
        stages,
        params={"history": []},
        image_analysis_results=inputs,
        targets={"b": "same sample", "a": "same sample"},
        output=lambda *args: None,
        **kwargs,
    )


@pytest.mark.parametrize("with_preprocess", [False, True])
def test_pipeline_accepts_data_only_lane_inputs(with_preprocess):
    inputs = Inputs(pd.DataFrame({"Data": ["a", "unused", "b", "a"]}))
    calls = []

    def prepare(args):
        frame = args.image_analysis_results.cells.assign(score=[10, 100, 20, 30])
        return Prepared(frame, frame[["Data"]])

    def mapping(args):
        calls.append(args.data_name)
        data = args.image_analysis_results
        assert isinstance(data, Prepared if with_preprocess else Inputs)
        assert data.cells.Data.tolist() == (
            ["b"] if args.data_name == "b" else ["a", "a"]
        )
        assert list(data.cells.columns) == (
            ["Data", "score"] if with_preprocess else ["Data"]
        )
        if with_preprocess:
            assert list(data.spots.columns) == ["Data"]
            return int(data.cells.score.sum())
        return len(data.cells)

    def finalize(args):
        assert list(args.image_analysis_results) == ["b", "a"]
        return pd.DataFrame(
            {"name": list(args.values), "value": list(args.values.values())}
        )

    stages = [Stage("preprocess", prepare)] if with_preprocess else []
    stages += [Stage("map", mapping), Stage("finalize", finalize)]
    result = run(stages, inputs)
    assert_frame_equal(
        result,
        pd.DataFrame(
            {"name": ["b", "a"], "value": [20, 40] if with_preprocess else [1, 2]}
        ),
    )
    assert calls == ["b", "a"]


def test_snapshot_ignores_scanner_column_contents():
    frame = pd.DataFrame(
        {
            "Data": ["b", "a"],
            "Filename": [None, "invalid"],
            "MultiPointIndex": [None, None],
            "ImageAnalysisMethod": [None, ""],
        }
    )
    snapshot = DataFrameInputSnapshot(
        Inputs(frame), (Lane(0, "a", "A"), Lane(1, "b", "B"))
    )
    restored = snapshot.restore_all()
    assert list(restored) == ["a", "b"]
    assert_frame_equal(restored["a"].cells, frame.iloc[[1]])
    assert_frame_equal(restored["b"].cells, frame.iloc[[0]])


@pytest.mark.parametrize("field", ["cells", "spots"])
@pytest.mark.parametrize("targets", [{}, {"a": "A"}])
@pytest.mark.parametrize("with_preprocess", [False, True])
def test_missing_data_column_identifies_input_before_stages(
    field, targets, with_preprocess
):
    frame = pd.DataFrame({"Data": ["a"], "score": [1]})
    prepared = Prepared(
        **{
            name: frame.drop(columns="Data") if name == field else frame
            for name in Prepared._fields
        }
    )

    def forbidden(args):
        pytest.fail("stage ran with missing Data column")

    stages = [Stage("preprocess", lambda a: prepared)] if with_preprocess else []
    stages += [
        Stage("synthesize", forbidden),
        Stage("map", forbidden),
        Stage("finalize", forbidden),
    ]
    with pytest.raises(
        ValueError, match=f"input '{field}': missing required 'Data' column"
    ):
        execute_sequential(
            stages,
            params={},
            image_analysis_results=Inputs(frame) if with_preprocess else prepared,
            targets=targets,
            output=lambda *a: None,
        )


@pytest.mark.parametrize("field", ["cells", "spots"])
def test_snapshot_rejects_missing_lane_in_each_minimal_input(field):
    frame = pd.DataFrame({"Data": ["a", "b"]})
    inputs = Prepared(
        **{
            name: frame.iloc[:1] if name == field else frame
            for name in Prepared._fields
        }
    )
    with pytest.raises(ValueError, match=f"missing lane: 'b' in input '{field}'"):
        DataFrameInputSnapshot(inputs, (Lane(0, "a", "A"), Lane(1, "b", "B")))


@pytest.mark.parametrize("preprocess", [False, True])
def test_qc_feedback_copies_and_call_order(inputs, monkeypatch, preprocess):
    # Neither input object cells nor intermediate/final values require pickle.
    import pickle

    monkeypatch.setattr(pickle, "dumps", lambda *a, **k: pytest.fail("pickle used"))
    original = inputs.cells.copy(deep=True)
    calls = []
    identities = {}
    shared = []
    expected_type = Prepared if preprocess else Inputs
    offset = 10 if preprocess else 0

    def check(args):
        assert args.params == {"history": []}
        args.params["history"].append("mutated")

    def prepare(args):
        check(args)
        calls.append("preprocess")
        assert list(args.targets.items()) == [
            ("b", "same sample"),
            ("a", "same sample"),
        ]
        with pytest.raises(TypeError):
            args.targets["new"] = "sample"
        args.image_analysis_results.cells["score"] += 10
        return Prepared(
            args.image_analysis_results.cells, args.image_analysis_results.cells.copy()
        )

    def first(args):
        check(args)
        calls.append((args.data_name, "first"))
        assert args.sample_name == "same sample"
        assert args.value is args.synthesized is None
        assert isinstance(args.image_analysis_results, expected_type)
        score = int(args.image_analysis_results.cells.score.iloc[0])
        for frame in args.image_analysis_results:
            frame["score"] = -99
        value = {"get": lambda: score, "history": []}
        identities[args.data_name] = value
        return value

    def second(args):
        check(args)
        calls.append((args.data_name, "second"))
        assert args.value is identities[args.data_name]
        assert (
            int(args.image_analysis_results.cells.score.iloc[0]) == args.value["get"]()
        )
        args.value["history"].append("second")
        return args.value

    def qc(args):
        check(args)
        calls.append("qc")
        assert list(args.values) == list(args.image_analysis_results) == ["b", "a"]
        assert args.synthesized is None
        for name, value in args.values.items():
            assert value is identities[name]
        with pytest.raises(TypeError):
            args.values["a"] = None
        with pytest.raises(TypeError):
            del args.image_analysis_results["b"]
        for lane in args.image_analysis_results.values():
            for frame in lane:
                frame["score"] = -99
        shared.append({"threshold": min(v["get"]() for v in args.values.values())})
        return shared[0]

    def correct(args):
        check(args)
        calls.append((args.data_name, "correct"))
        assert args.synthesized is shared[0]
        assert (
            int(args.image_analysis_results.cells.score.iloc[0]) == args.value["get"]()
        )
        return args.value["get"]() - args.synthesized["threshold"]

    def finalize(args):
        check(args)
        calls.append("finalize")
        assert args.synthesized is shared[0]
        assert args.image_analysis_results["a"].cells.score.iloc[0] == 1 + offset
        return pd.DataFrame(
            {"name": list(args.values), "score": list(args.values.values())}
        )

    stages = [Stage("preprocess", prepare)] if preprocess else []
    stages += [
        Stage("map", first),
        Stage("map", second),
        Stage("synthesize", qc),
        Stage("map", correct),
        Stage("finalize", finalize),
    ]
    assert_frame_equal(
        run(stages, inputs), pd.DataFrame({"name": ["b", "a"], "score": [1, 0]})
    )
    assert calls == (["preprocess"] if preprocess else []) + [
        ("b", "first"),
        ("b", "second"),
        ("a", "first"),
        ("a", "second"),
        "qc",
        ("b", "correct"),
        ("a", "correct"),
        "finalize",
    ]
    assert_frame_equal(inputs.cells, original)


@pytest.mark.parametrize("targets", [{}, {"a": "sample"}])
@pytest.mark.parametrize("with_map", [False, True])
def test_initial_state_consecutive_syntheses_and_none_updates(
    inputs, targets, with_map
):
    seen = []

    def first(args):
        assert dict(args.values) == dict.fromkeys(targets)
        assert args.synthesized is None
        seen.append("first")
        return {"state": []}

    def second(args):
        assert args.synthesized == {"state": []}
        seen.append("second")

    def mapping(args):
        assert args.value is args.synthesized is None
        seen.append("map")
        return "value"

    def reset(args):
        assert args.value == "value"

    def final(args):
        assert args.synthesized is None
        assert dict(args.values) == dict.fromkeys(targets)
        return pd.DataFrame()

    stages = [Stage("synthesize", first), Stage("synthesize", second)]
    if with_map:
        stages += [Stage("map", mapping), Stage("map", reset)]
    stages.append(Stage("finalize", final))
    execute_sequential(
        stages,
        params={},
        image_analysis_results=inputs,
        targets=targets,
        output=lambda *a: None,
    )
    assert seen == ["first", "second"] + (["map"] if targets and with_map else [])


def test_finalize_only(inputs):
    def final(args):
        assert dict(args.values) == {"b": None, "a": None}
        assert args.synthesized is None
        return pd.DataFrame()

    run([Stage("finalize", final)], inputs)


def test_snapshot_is_fixed_before_first_synthesize_and_reusable(inputs):
    lanes = (Lane(0, "a", "A"), Lane(1, "b", "B"))
    snapshot = DataFrameInputSnapshot(inputs, lanes)
    inputs.cells["score"] = 500
    restored = snapshot.restore(0)
    restored.cells["score"] = -1
    assert snapshot.restore(0).cells.score.iloc[0] == 1

    def synthesize(args):
        for lane in args.image_analysis_results.values():
            lane.cells["score"] = -100

    def mapping(args):
        assert args.image_analysis_results.cells.score.iloc[0] == 500
        return args.image_analysis_results.cells.score.iloc[0]

    result = run(
        [
            Stage("synthesize", synthesize),
            Stage("map", mapping),
            Stage(
                "finalize", lambda a: pd.DataFrame({"score": list(a.values.values())})
            ),
        ],
        inputs,
    )
    assert result.score.tolist() == [500, 500]


def test_preprocess_failure_has_no_snapshot_or_later_side_effects(inputs):
    events = []

    def prepare(args):
        args.image_analysis_results.cells["score"] = -1
        args.params["history"].append("changed")
        raise LookupError("preprocess failure")

    params = {"history": []}
    with pytest.raises(LookupError, match="preprocess failure"):
        execute_pipeline(
            [
                Stage("preprocess", prepare),
                Stage("finalize", lambda a: pytest.fail("continued")),
            ],
            params=params,
            image_analysis_results=inputs,
            targets={"a": "A"},
            output=lambda *a: None,
            map_executor=SequentialMapExecutor(),
            snapshot_factory=tracking_factory(events),
        )
    assert inputs.cells.score.tolist() == [1, 2, 100]
    assert params == {"history": []}
    assert events == []


@pytest.mark.parametrize("kind", ["synthesize", "finalize"])
def test_snapshot_closed_on_parent_failure(inputs, kind):
    events = []

    def fail(args):
        assert events == ["open"]
        raise LookupError("parent")

    stages = [Stage(kind, fail)]
    if kind == "synthesize":
        stages.append(Stage("finalize", lambda a: pytest.fail("continued")))
    with pytest.raises(LookupError, match="parent"):
        execute_pipeline(
            stages,
            params={},
            image_analysis_results=inputs,
            targets={"a": "A"},
            output=lambda *a: None,
            map_executor=SequentialMapExecutor(),
            snapshot_factory=tracking_factory(events),
        )
    assert events == ["open", "close"]


def test_segment_failure_preserves_completed_lanes_and_images(inputs):
    events = []
    captured = []

    class RecordingExecutor(SequentialMapExecutor):
        def execute(self, segment, context, on_value):
            captured.append(context.values)
            super().execute(segment, context, on_value)

    def first(args):
        args.output(args.data_name)
        return "intermediate"

    def second(args):
        if args.data_name == "a":
            raise LookupError("failed a")
        return "complete"

    def forbidden(args):
        pytest.fail("crossed a failed segment")

    with pytest.raises(LookupError, match="failed a"):
        execute_pipeline(
            [
                Stage("map", lambda a: "previous"),
                Stage("synthesize", lambda a: None),
                Stage("map", first),
                Stage("map", second),
                Stage("synthesize", forbidden),
                Stage("finalize", forbidden),
            ],
            params={},
            image_analysis_results=inputs,
            targets={"b": "B", "a": "A"},
            output=events.append,
            map_executor=RecordingExecutor(),
            snapshot_factory=tracking_factory([]),
        )
    assert events == ["b", "a"]
    assert dict(captured[-1]) == {"b": "complete", "a": "previous"}


@pytest.mark.parametrize("kind", ["synthesize", "finalize"])
def test_parent_failure_keeps_referenced_mutations_and_stops(inputs, kind):
    value = []
    shared = []
    views = []

    def fail(args):
        views.append(args.values)
        assert args.values["a"] is value
        assert args.synthesized is shared
        args.values["a"].append("changed")
        args.synthesized.append("changed")
        raise LookupError("parent failure")

    def forbidden(args):
        pytest.fail("continued after failure")

    stages = [
        Stage("map", lambda a: value),
        Stage("synthesize", lambda a: shared),
        Stage(kind, fail),
    ]
    if kind == "synthesize":
        stages += [Stage("map", forbidden), Stage("finalize", forbidden)]
    with pytest.raises(LookupError, match="parent failure"):
        run(stages, inputs)
    assert views[0]["a"] is value
    assert value == shared == ["changed"]


def test_callback_failure_aborts_before_next_lane(inputs):
    events = []
    lanes = (Lane(0, "a", "A"), Lane(1, "b", "B"))
    context = MapExecutionContext(
        DataFrameInputSnapshot(inputs, lanes),
        lanes,
        {},
        {"a": None, "b": None},
        None,
        lambda *a: None,
    )

    def callback(lane_id, value):
        events.append((lane_id, value))
        raise LookupError("callback")

    def mapping(args):
        events.append(args.data_name)

    with pytest.raises(LookupError, match="callback"):
        SequentialMapExecutor().execute(
            MapSegment(3, 4, (Stage("map", mapping),)), context, callback
        )
    assert events == ["a", (0, None)]


def tracking_factory(events):
    @contextmanager
    def factory(inputs, lanes):
        events.append("open")
        try:
            yield DataFrameInputSnapshot(inputs, lanes)
        finally:
            events.append("close")

    return factory


@pytest.mark.parametrize("fail", [False, True])
def test_injected_executor_absolute_ranges_and_snapshot_lifetime(inputs, fail):
    events = []
    instances = []

    @contextmanager
    def factory(data, lanes):
        events.append("open")
        # Deliberately no restore(lane_id): Runner only needs restore_all.
        snapshot = DataFrameInputSnapshot(data, lanes)
        try:
            yield SimpleNamespace(restore_all=snapshot.restore_all)
        finally:
            events.append("close")

    class Executor:
        def execute(self, segment, context, on_value):
            instances.append(self)
            assert events[-1] != "close"
            events.append((segment.start, segment.end))
            assert [(l.lane_id, l.data_name, l.sample_name) for l in context.lanes] == [
                (0, "b", "B"),
                (1, "a", "A"),
            ]
            on_value(0, "done")
            assert context.values["b"] == "done"  # synchronous update
            if fail:
                events.append("workers collected")
                raise LookupError("worker")
            on_value(1, None)

    stages = [
        Stage("preprocess", lambda a: a.image_analysis_results),
        Stage("map", lambda a: None),
        Stage("map", lambda a: None),
        Stage("synthesize", lambda a: None),
        Stage("map", lambda a: None),
        Stage("finalize", lambda a: pd.DataFrame()),
    ]
    kwargs = {
        "params": {},
        "image_analysis_results": inputs,
        "targets": {"b": "B", "a": "A"},
        "output": lambda *a: None,
        "map_executor": Executor(),
        "snapshot_factory": factory,
    }
    if fail:
        with pytest.raises(LookupError, match="worker"):
            execute_pipeline(stages, **kwargs)
        assert events == ["open", (1, 3), "workers collected", "close"]
    else:
        execute_pipeline(stages, **kwargs)
        assert events == ["open", (1, 3), (4, 5), "close"]
        assert instances[0] is instances[1]


def test_missing_preprocessed_lane_rejected_before_any_stage(inputs):
    def prepare(args):
        return Prepared(
            args.image_analysis_results.cells,
            args.image_analysis_results.cells.query("Data == 'a'"),
        )

    def forbidden(args):
        pytest.fail("missing lane passed validation")

    with pytest.raises(ValueError, match="missing lane: 'b'"):
        run(
            [
                Stage("preprocess", prepare),
                Stage("synthesize", forbidden),
                Stage("map", forbidden),
                Stage("finalize", forbidden),
            ],
            inputs,
        )


@pytest.mark.parametrize("result", [None, 1, pd.Series([1])])
def test_finalize_requires_dataframe(inputs, result):
    with pytest.raises(TypeError, match="finalize must return a pandas DataFrame"):
        run([Stage("finalize", lambda a: result)], inputs)


@pytest.mark.parametrize(
    "kinds",
    [
        [],
        ["map"],
        ["preprocess", "preprocess", "finalize"],
        ["map", "preprocess", "finalize"],
        ["finalize", "finalize"],
        ["unknown", "finalize"],
    ],
)
def test_invalid_definition_rejected_before_side_effects(inputs, kinds):
    def forbidden(args):
        pytest.fail("invalid definition ran")

    with pytest.raises(ValueError):
        run([Stage(kind, forbidden) for kind in kinds], inputs)
