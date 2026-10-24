"""#82 の状態所有権・実ストリーム・異常系の再現可能な検証。"""

from __future__ import annotations

import pickle
import tarfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pandas as pd
import pytest

from tests import pipeline_transfer_probe as p
from tests.support import read_tar_member


@pytest.fixture
def inputs():
    # data_name は TAR のパスとして安全である必要がない。
    return p.Inputs(pd.DataFrame({"lane": ["../a/name", "日本語/b"], "score": [1, 2]}))


@pytest.fixture
def targets():
    return {"../a/name": "sample a", "日本語/b": "sample b"}


def test_preprocess_snapshot_type_lifetime_and_parent_restore(
    inputs: p.Inputs, targets: dict[str, str]
):
    calls = []

    def preprocess(data: p.Inputs | p.Prepared):
        calls.append(1)
        return p.prepare(data)

    with p.snapshots(inputs, targets, preprocess) as lanes:
        paths = [Path(lane["input_path"]) for lane in lanes]
        assert all(path.exists() for path in paths)
        restored = p.restore_all(lanes)
        assert isinstance(restored, MappingProxyType)
        assert all(isinstance(value, p.Prepared) for value in restored.values())
        assert [int(value.cells.score.iloc[0]) for value in restored.values()] == [
            11,
            12,
        ]
        # synthesize 相当の破壊的変更も、後続 map/finalize 入力に伝播しない。
        restored["../a/name"].cells["score"] = -100
        assert isinstance(restored["../a/name"], p.Prepared)
        restored["../a/name"].spots["score"] = -100
        assert p.restore_all(lanes)["../a/name"].cells.score.iloc[0] == 11
        fresh = p.restore_all(lanes)["../a/name"]
        assert isinstance(fresh, p.Prepared)
        assert fresh.spots.score.iloc[0] == 11
        with pytest.raises(TypeError):
            cast(dict[str, Any], restored)["new"] = None
    assert calls == [1]
    assert inputs.cells.score.tolist() == [1, 2]
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.DataFrame({"x": [1, 2]}),
        p.Feature(3),
        {"tuple": (1, b"bytes"), "set": {2, 3}},
    ],
)
def test_pickle_roundtrip_request_response_and_file(
    inputs: p.Inputs, targets: dict[str, str], tmp_path: Path, value: Any
):
    path = tmp_path / "value.pkl"
    path.write_bytes(p.dumps(value, "test"))
    restored = pickle.loads(path.read_bytes())
    with p.snapshots(inputs, targets) as lanes:
        values = {lane["lane_id"]: restored for lane in lanes}
        # echo は params/synthesized が None や任意型でも扱える。
        request = p.request_for(lanes, 5, 6)
        p.run_child(request, value, value, values)
    for result in values.values():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(result, value)
        else:
            assert result == value


def test_segment_isolation_and_streaming_before_child_exit(
    inputs: p.Inputs, targets: dict[str, str], tmp_path: Path
):
    processes, events = [], []
    params = {"history": []}
    synthesized = {"history": []}
    ack = tmp_path / "parent-received"
    with p.snapshots(inputs, targets, p.prepare) as lanes:
        values = {0: None, 1: None}

        def on_event(kind: str, lane_id: int, data: Any):
            events.append((kind, lane_id, data))
            # 子はこの ack を待ってから次レーン/終了に進む。
            assert processes[0].poll() is None
            if kind == "value":
                assert values[lane_id] == data
                Path(f"{ack}.{lane_id}").touch()

        stderr = p.run_child(
            p.request_for(lanes),
            params,
            synthesized,
            values,
            on_event,
            fault="stderr",
            ack=ack,
            processes=processes,
        )
        assert values == {0: p.Feature(13), 1: p.Feature(14)}
        assert [(kind, lane) for kind, lane, _ in events] == [
            ("image", 0),
            ("image", 0),
            ("value", 0),
            ("image", 1),
            ("image", 1),
            ("value", 1),
        ]
        assert events[0][2] == b"first image"
        assert len(stderr) > 65536
        assert processes[0].returncode == 0
    assert params == synthesized == {"history": []}
    assert inputs.cells.score.tolist() == [1, 2]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"protocol_version": 2}, "protocol_version mismatch"),
        ({"manifest": "{}"}, "manifest mismatch"),
        ({"stage_start": 1, "stage_end": 3}, "non-map"),
        ({"stage_start": 3, "stage_end": 5}, "non-map"),
        ({"stage_start": 2, "stage_end": 2}, "invalid map segment"),
        ({"stage_start": -1}, "invalid map segment"),
        ({"stage_end": 100}, "invalid map segment"),
        ({"stage_start": True}, "invalid map segment"),
        ({"stage_end": 3.0}, "invalid map segment"),
    ],
)
def test_reject_before_any_stage(
    inputs: p.Inputs,
    targets: dict[str, str],
    change: dict[str, Any],
    message: str | None,
):
    with p.snapshots(inputs, targets, p.prepare) as lanes:
        request = p.request_for(lanes) | change
        values, events = {0: None, 1: None}, []
        with pytest.raises(p.TransferError, match=message) as exc:
            p.run_child(
                request,
                {"history": []},
                {"history": []},
                values,
                lambda *event: events.append(event),
            )
        assert '"lane_id": null' in str(exc.value)
        assert '"stage_index": null' in str(exc.value)
        assert values == {0: None, 1: None}
        assert events == []


@pytest.mark.parametrize(
    "fault, message",
    [
        ("missing", "missing value.pkl"),
        ("duplicate", "duplicate value.pkl"),
        ("nonzero", "exit=7"),
        ("error", "injected error"),
        ("corrupt", "TAR"),
    ],
)
def test_transport_failure_reaps_child_before_snapshot_cleanup(
    inputs: p.Inputs, targets: dict[str, str], fault: str, message: str | None
):
    processes, following = [], []
    paths = []
    with (
        pytest.raises(p.TransferError, match=message),
        p.snapshots(inputs, targets, p.prepare) as lanes,
    ):
        paths = [Path(lane["input_path"]) for lane in lanes]
        try:
            p.run_child(
                p.request_for(lanes, 5, 6),
                None,
                None,
                {0: None, 1: None},
                fault=fault,
                processes=processes,
            )
            following.append("synthesize/finalize")
        finally:
            assert all(proc.poll() is not None for proc in processes)
            assert all(path.exists() for path in paths)
    assert following == []
    assert all(not path.exists() for path in paths)


def test_failed_lane_preserves_received_values_images_and_absolute_stage(
    inputs: p.Inputs, targets: dict[str, str]
):
    with p.snapshots(inputs, targets, p.prepare) as lanes:
        values, events = {0: None, 1: None}, []
        with pytest.raises(p.TransferError, match="LookupError") as exc:
            p.run_child(
                p.request_for(lanes, 5, 8),
                None,
                None,
                values,
                lambda *event: events.append(event),
            )
        message = str(exc.value)
        assert '"stage_index": 6' in message
        assert '"lane_id": 0' in message
        assert "../a/name" in message
        assert "Traceback" in message
        assert "cannot pickle" not in message  # stage 7 は未実行。
    # 別の失敗: レーン0完了後、レーン1の入力ファイル復元が失敗する。
    with p.snapshots(inputs, targets, p.prepare) as lanes:
        Path(lanes[1]["input_path"]).write_bytes(b"broken pickle")
        values, events = {0: None, 1: None}, []
        with pytest.raises(p.TransferError, match="UnpicklingError"):
            p.run_child(
                p.request_for(lanes),
                {"history": []},
                {"history": []},
                values,
                lambda *event: events.append(event),
            )
        assert values == {0: p.Feature(13), 1: None}
        assert [event[:2] for event in events] == [
            ("image", 0),
            ("image", 0),
            ("value", 0),
        ]


def test_unpicklable_final_value_has_lane_and_stage(
    inputs: p.Inputs, targets: dict[str, str]
):
    with p.snapshots(inputs, targets) as lanes:
        with pytest.raises(p.TransferError, match="cannot pickle value lane=0") as exc:
            p.run_child(p.request_for(lanes, 7, 8), None, None, {0: None, 1: None})
        assert '"stage_index": 7' in str(exc.value)
        assert "../a/name" in str(exc.value)


def test_corrupt_received_value_has_lane_and_segment_context(
    inputs: p.Inputs, targets: dict[str, str]
):
    processes, events, following = [], [], []
    values = {0: None, 1: p.Feature(99)}
    with p.snapshots(inputs, targets, p.prepare) as lanes:
        paths = [Path(lane["input_path"]) for lane in lanes]
        with pytest.raises(p.TransferError, match="cannot unpickle value.pkl") as exc:
            try:
                p.run_child(
                    p.request_for(lanes),
                    {"history": []},
                    {"history": []},
                    values,
                    lambda *event: events.append(event),
                    fault="corrupt_value",
                    processes=processes,
                )
                following.append("synthesize/finalize")
            finally:
                assert all(proc.poll() is not None for proc in processes)
                assert all(path.exists() for path in paths)
        error = exc.value.__cause__
        assert isinstance(error, p.TransferError)
        assert isinstance(error.__cause__, pickle.UnpicklingError)
        for diagnostic in (
            "lane_id=1",
            "data_name='日本語/b'",
            "stage_start=2",
            "stage_end=4",
            "UnpicklingError",
        ):
            assert diagnostic in str(error)
        assert values == {0: p.Feature(13), 1: p.Feature(99)}
        assert [event[:2] for event in events] == [
            ("image", 0),
            ("image", 0),
            ("value", 0),
            ("image", 1),
            ("image", 1),
        ]
    assert following == []
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("field", ["params", "synthesized", "value"])
def test_parent_pickle_failure_has_data_context_and_reaps(
    inputs: p.Inputs, targets: dict[str, str], field: str
):
    processes = []
    with p.snapshots(inputs, targets) as lanes:
        bad = lambda: None
        with pytest.raises(p.TransferError, match=f"cannot pickle {field}"):
            p.run_child(
                p.request_for(lanes, 5, 6),
                bad if field == "params" else None,
                bad if field == "synthesized" else None,
                {0: bad if field == "value" else None, 1: None},
                processes=processes,
            )
        assert all(proc.poll() is not None for proc in processes)


def test_input_pickle_failure_cleans_directory_before_any_child(
    inputs: p.Inputs, targets: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    directories = []
    original = p.TemporaryDirectory

    def temporary(*args: Any, **kwargs: Any):
        directory = original(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(p, "TemporaryDirectory", temporary)
    inputs.cells["object"] = pd.Series([lambda: None, None], dtype=object)
    with (
        pytest.raises(p.TransferError, match="input lane=0 data_name=../a/name"),
        p.snapshots(inputs, targets),
    ):
        pytest.fail("snapshot failure must precede child startup")
    assert all(not directory.exists() for directory in directories)


def test_notebook_copies_without_pickle_and_preserves_type(
    inputs: p.Inputs, targets: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("Notebook must not require pickle")

    monkeypatch.setattr(p.pickle, "dumps", forbidden)
    monkeypatch.setattr(p.pickle, "loads", forbidden)
    inputs.cells["object"] = pd.Series([lambda: None, lambda: None], dtype=object)
    snapshot = p.notebook_snapshot(inputs, targets, p.prepare)
    inputs.cells["score"] = -2
    first = p.notebook_restore(snapshot)
    assert isinstance(first["../a/name"], p.Prepared)
    first["../a/name"].cells["score"] = -3
    second = p.notebook_restore(snapshot)
    assert second["../a/name"].cells.score.iloc[0] == 11
    assert isinstance(second["日本語/b"], p.Prepared)
    assert second["日本語/b"].spots.score.iloc[0] == 12
    with pytest.raises(TypeError):
        cast(dict[str, Any], second)["new"] = None


@pytest.mark.parametrize("cut", [1, 1024, 10240])
def test_truncated_tar_is_rejected(cut: int):
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w|") as tar:
        p.add(tar, "lanes/0/value.pkl", p.dumps(None, "value"))
    data = stream.getvalue()
    # 最後のケースは payload の途中まで切断。
    truncated = data[: 2048 - cut] if cut < 2048 else data[:520]
    with pytest.raises((p.TransferError, tarfile.TarError, EOFError)):
        p.receive(
            BytesIO(truncated), {"lanes": [{"lane_id": 0}]}, {}, lambda *args: None
        )


def test_response_version_mismatch():
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w|") as tar:
        p.add(tar, "error.json", b'{"protocol_version": 2}')
    with pytest.raises(p.TransferError, match="response protocol_version mismatch"):
        p.receive(BytesIO(stream.getvalue()), {"lanes": []}, {}, lambda *args: None)


def test_parent_argument_ownership(inputs: p.Inputs, targets: dict[str, str]):
    params = {"history": []}
    values = {name: p.Feature(1) for name in targets}
    synthesized = values["../a/name"]
    with p.snapshots(inputs, targets) as lanes:
        first = p.parent_args(lanes, params, values, synthesized)
        first["params"]["history"].append(1)
        second = p.parent_args(lanes, params, values, synthesized)
        assert second["params"] == params == {"history": []}
        assert first["synthesized"] is second["synthesized"] is synthesized
        assert first["values"]["../a/name"] is values["../a/name"]
        with pytest.raises(TypeError):
            first["values"]["new"] = None


def test_none_return_replaces_previous_value(inputs: p.Inputs, targets: dict[str, str]):
    with p.snapshots(inputs, targets) as lanes:
        values = {0: p.Feature(1), 1: p.Feature(2)}
        p.run_child(p.request_for(lanes, 8, 9), None, None, values)
        assert values == {0: None, 1: None}


@pytest.mark.parametrize("preprocess", [None, p.prepare])
def test_snapshot_is_fixed_before_first_map(
    inputs: p.Inputs,
    targets: dict[str, str],
    preprocess: Callable[[p.Inputs | p.Prepared], p.Inputs | p.Prepared] | None,
):
    with p.snapshots(inputs, targets, preprocess) as lanes:
        inputs.cells["score"] = -100
        expected = 11 if preprocess else 1
        assert p.restore_all(lanes)["../a/name"].cells.score.iloc[0] == expected


def test_missing_lane_is_rejected_before_children(
    inputs: p.Inputs, targets: dict[str, str]
):
    with (
        pytest.raises(ValueError, match="missing lane"),
        p.snapshots(inputs, targets | {"missing": "sample"}),
    ):
        pytest.fail("invalid lane set must not reach execution")


def test_request_schema_and_assigned_lane_subset(
    inputs: p.Inputs, targets: dict[str, str]
):
    with p.snapshots(inputs, targets) as lanes:
        request = p.request_for(lanes[1:], 5, 6)
        stream = BytesIO()
        p.write_request(stream, request, None, None, {1: None})
        with tarfile.open(fileobj=BytesIO(stream.getvalue())) as archive:
            assert archive.getnames() == [
                "request.json",
                "params.pkl",
                "synthesized.pkl",
                "lanes/1/value.pkl",
            ]
            for name in ["params.pkl", "synthesized.pkl", "lanes/1/value.pkl"]:
                assert pickle.loads(read_tar_member(archive, name)) is None
        values = {1: p.Feature(3)}
        p.run_child(request, None, None, values)
        assert values == {1: p.Feature(3)}


def test_no_lanes_needs_no_map_call():
    values = {}
    p.run_child(p.request_for([], 2, 4), None, None, values)
    assert values == {}
