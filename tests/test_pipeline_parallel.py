import gzip
import json
import os
import pickle
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from threading import get_ident

import pandas as pd
import pytest

from analysisrun import pipeline_parallel as p
from analysisrun.pipeline import (
    Lane,
    MapExecutionContext,
    MapSegment,
    Stage,
    execute_pipeline,
    execute_sequential,
)
from tests import pipeline_worker_entry as entry

ENTRYPOINT = Path(entry.__file__)


def params(directory, fault=""):
    return {"directory": str(directory), "history": [], "fault": fault}


def targets():
    return {str(i): f"sample {i}" for i in range(4)}


def run(directory, fault="", on_image=None, on_value=None, max_workers=2):
    stages = entry.definition()
    executor = p.MultiprocessMapExecutor(
        stages,
        ENTRYPOINT,
        on_image or (lambda *args: None),
        max_workers=max_workers,
        queue_size=1,
    )
    if on_value:
        execute = executor.execute

        def observed(segment, context, callback):
            def notify(lane_id, value):
                callback(lane_id, value)
                on_value(segment, context, lane_id, value)

            execute(segment, context, notify)

        executor.execute = observed
    return execute_pipeline(
        stages,
        params=params(directory, fault),
        image_analysis_results=entry.inputs(),
        targets=targets(),
        output=None,
        map_executor=executor,
        snapshot_factory=p.file_snapshot,
    )


def track_children(monkeypatch):
    processes = []
    original = p.subprocess.Popen

    def popen(*args, **kwargs):
        proc = original(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(p.subprocess, "Popen", popen)
    return processes


def test_parallel_segments_equivalent_and_reaped(tmp_path, monkeypatch):
    processes = track_children(monkeypatch)
    thread_id = get_ident()
    images = []
    snapshots = []

    def on_image(*image):
        assert get_ident() == thread_id
        images.append(image)

    def on_value(segment, context, lane_id, value):
        assert get_ident() == thread_id
        snapshots.append(tuple(context.snapshot.paths.values()))
        assert all(path.exists() for path in context.snapshot.paths.values())

    result = run(tmp_path, "parallel", on_image, on_value)
    assert result.score.tolist() == [112, 114, 116, 118]
    assert len(images) == 4
    assert {image[0] for image in images} == set(targets())
    assert all(image[2].startswith(b"\x89PNG") and image[3] == "qc" for image in images)
    assert len(processes) == 4  # two fresh workers for each segment
    assert all(proc.poll() == 0 for proc in processes)
    assert len((tmp_path / "preprocess").read_text().splitlines()) == 1
    assert all(paths == snapshots[0] for paths in snapshots)
    assert all(not path.exists() for path in snapshots[0])
    sequential_dir = tmp_path / "sequential"
    sequential_dir.mkdir()

    def discard(fig, *args, **kwargs):
        entry.plt.close(fig)

    expected = execute_sequential(
        entry.definition(),
        params=params(sequential_dir),
        image_analysis_results=entry.inputs(),
        targets=targets(),
        output=discard,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_value_notified_before_worker_finishes(tmp_path):
    def on_value(segment, context, lane_id, value):
        if segment.start == 5 and lane_id == 0:
            assert context.values["0"] == value
            (tmp_path / "ack-0").touch()

    assert len(run(tmp_path, "ack", on_value=on_value, max_workers=1)) == 4


@pytest.mark.parametrize("fault", ["fail-peer", "kill-peer", "output-fail"])
def test_failure_reaps_all_and_stops_progress(tmp_path, monkeypatch, fault):
    processes = track_children(monkeypatch)
    paths = []
    original = p.file_snapshot

    from contextlib import contextmanager

    @contextmanager
    def snapshot(*args):
        with original(*args) as snap:
            paths.extend(snap.paths.values())
            try:
                yield snap
            finally:
                assert all(proc.poll() is not None for proc in processes)
                assert all(path.exists() for path in paths)

    monkeypatch.setattr(p, "file_snapshot", snapshot)

    def on_image(*args):
        if fault == "output-fail":
            raise RuntimeError("output failed")

    with pytest.raises((p.TransferError, RuntimeError), match="failure|output failed"):
        run(tmp_path, fault, on_image=on_image)
    assert len(processes) == 2
    assert all(proc.poll() is not None for proc in processes)
    assert all(not path.exists() for path in paths)
    assert not (tmp_path / "synthesize").exists()
    assert not (tmp_path / "finalize").exists()


def test_manifest_checks_parent_stages_before_maps(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MISMATCH", "1")
    with pytest.raises(p.TransferError, match="pipeline manifest mismatch"):
        run(tmp_path)
    assert not list(tmp_path.glob("started-*"))


def test_large_stderr_is_drained(tmp_path):
    assert len(run(tmp_path, "stderr")) == 4


@pytest.mark.parametrize("setting", ["", "0", "-1", "x", "1.5", " 2", "２"])
def test_worker_limit_invalid(monkeypatch, setting):
    monkeypatch.setenv("ANALYSISRUN_MAX_WORKERS", setting)
    with pytest.raises(ValueError, match="positive integer"):
        p.worker_limit()


def test_worker_limit_default_and_split(monkeypatch):
    monkeypatch.delenv("ANALYSISRUN_MAX_WORKERS", raising=False)
    monkeypatch.setattr(p.os, "process_cpu_count", lambda: None)
    assert p.worker_limit() == 1
    monkeypatch.setattr(p.os, "process_cpu_count", lambda: 4)
    assert p.worker_limit() == 4
    monkeypatch.setenv("ANALYSISRUN_MAX_WORKERS", "3")
    assert p.worker_limit() == 3
    chunks = p._split_evenly_in_order(list(range(11)), 4)
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 2]
    assert [item for chunk in chunks for item in chunk] == list(range(11))
    assert p._split_evenly_in_order([1], 4) == [[1]]


def archive(entries):
    stream = BytesIO()
    with p.writing_tar(stream) as tar:
        for name, data in entries:
            p._add(tar, name, data)
    return stream.getvalue()


def request():
    return {
        "lanes": [{"lane_id": 0, "data_name": "../日本語"}],
        "stage_start": 2,
        "stage_end": 4,
    }


@pytest.mark.parametrize(
    "entries,match",
    [
        ([], "missing value"),
        ([("lanes/1/value.pkl", pickle.dumps(None))], "unknown lane"),
        ([("lanes/0/value.pkl", pickle.dumps(None))] * 2, "duplicate value"),
        ([("lanes/0/value.pkl", b"broken")], "cannot unpickle.*stage_start=2"),
        ([("error.json", b'{"protocol_version":2}')], "version mismatch"),
        ([("error.json", b'{"protocol_version":1,"message":"failed"}')], "failed"),
    ],
)
def test_invalid_response(entries, match):
    with pytest.raises(p.TransferError, match=match):
        p.receive_response(BytesIO(archive(entries)), request(), lambda *args: None)


@pytest.mark.parametrize("cut", [512, 1024, 10240])
def test_truncated_response(cut):
    data = archive([("lanes/0/value.pkl", pickle.dumps(None))])
    with pytest.raises((p.TransferError, tarfile.TarError)):
        p.receive_response(
            BytesIO(data[: max(520, 2048 - cut)]), request(), lambda *args: None
        )


def test_file_snapshot_types_isolation_and_cleanup(tmp_path):
    lanes = tuple(Lane(i, str(i), str(i)) for i in range(4))
    with p.file_snapshot(entry.inputs(), lanes) as snapshot:
        paths = list(snapshot.paths.values())
        assert all(path.is_absolute() for path in paths)
        first = snapshot.restore(0)
        first.cells["score"] = -1
        assert snapshot.restore(0).cells.score.iloc[0] == 0
        assert isinstance(snapshot.restore(0), entry.Inputs)
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("field", ["params", "synthesized", "value"])
def test_pickle_failure_precedes_children(tmp_path, monkeypatch, field):
    def forbidden(*args, **kwargs):
        pytest.fail("must not spawn")

    monkeypatch.setattr(p.subprocess, "Popen", forbidden)
    stages = (Stage("map", entry.third), Stage("finalize", entry.finalize))
    lanes = (Lane(0, "0", "sample"),)
    executor = p.MultiprocessMapExecutor(stages, ENTRYPOINT, lambda *args: None)
    with p.file_snapshot(entry.inputs(), lanes) as snapshot:
        bad = lambda: None
        context = MapExecutionContext(
            snapshot,
            lanes,
            bad if field == "params" else {},
            {"0": bad if field == "value" else None},
            bad if field == "synthesized" else None,
            None,
        )
        with pytest.raises(p.TransferError, match=f"cannot pickle {field}"):
            executor.execute(MapSegment(0, 1, stages[:1]), context, lambda *args: None)


def test_zero_lanes_no_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **k: pytest.fail("no spawn"))
    stages = (Stage("map", entry.third), Stage("finalize", entry.finalize))
    result = execute_pipeline(
        stages,
        params=params(tmp_path),
        image_analysis_results=entry.inputs(),
        targets={},
        output=None,
        snapshot_factory=p.file_snapshot,
        map_executor=p.MultiprocessMapExecutor(stages, ENTRYPOINT, lambda *a: None),
    )
    assert result.empty


@pytest.mark.parametrize("fault", ["", "fail-peer", "fail-after-image"])
def test_external_streaming_contract(tmp_path, fault):
    proc = subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        capture_output=True,
        timeout=30,
        check=False,
        env=os.environ
        | {
            "ANALYSISRUN_MODE": "parallel-entrypoint-streaming",
            "ANALYSISRUN_MAX_WORKERS": "2",
            "TEST_DIRECTORY": str(tmp_path),
            "TEST_FAULT": fault,
        },
    )
    with tarfile.open(fileobj=BytesIO(proc.stdout)) as tar:
        names = tar.getnames()
        if fault:
            assert proc.returncode != 0
            assert names[-1] == "error"
            assert "injected map failure" in tar.extractfile("error").read().decode()
            assert "result_csv" not in names
            if fault == "fail-after-image":
                assert any("/images/" in name for name in names)
        else:
            assert proc.returncode == 0, proc.stderr.decode()
            assert "result_csv" in names
            assert len([name for name in names if "/images/" in name]) == 4
            assert json.loads(tar.extractfile("0/result_json").read())["score"] == "112"
    assert gzip.decompress(proc.stdout)[-1024:] == bytes(1024)


def test_parallel_entrypoint_returns_frame_and_saves_images(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MAX_WORKERS", "2")
    result = p.execute_entrypoint(
        entry.definition(),
        entrypoint=ENTRYPOINT,
        params=params(tmp_path),
        image_analysis_results=entry.inputs(),
        targets=targets(),
        mode="parallel-entrypoint",
        output_dir=tmp_path / "images",
    )
    assert len(result) == 4
    assert len(list((tmp_path / "images").glob("*.png"))) == 4


@pytest.mark.parametrize(
    "field", ["protocol_version", "stage_start", "stage_end", "lanes"]
)
def test_worker_rejects_invalid_requests_before_map(field):
    stages = entry.definition()
    req = {
        "protocol_version": 1,
        "manifest": p.canonical_manifest(stages),
        "stage_start": 2,
        "stage_end": 4,
        "lanes": [],
    }
    req[field] = {
        "protocol_version": 2,
        "stage_start": True,
        "stage_end": 5,
        "lanes": [{"lane_id": -1}],
    }[field]
    # Invalid schema, including missing lane fields, must yield error.json.
    response = BytesIO()
    status = p.run_worker(
        stages,
        BytesIO(
            archive(
                [
                    ("request.json", json.dumps(req).encode()),
                    ("params.pkl", pickle.dumps(None)),
                    ("synthesized.pkl", pickle.dumps(None)),
                ]
            )
        ),
        response,
    )
    assert status == 1
    with tarfile.open(fileobj=BytesIO(response.getvalue())) as tar:
        assert tar.getnames() == ["error.json"]
        error = json.loads(tar.extractfile("error.json").read())
        assert error["stage_index"] is None
    assert response.getvalue()[-1024:] == bytes(1024)


def test_notebook_entrance_avoids_binary_stdout_and_pickle(tmp_path, monkeypatch):
    from io import StringIO

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.setattr(p, "get_interactivity", lambda: "notebook")
    monkeypatch.setattr(p.sys, "stdout", StringIO())
    monkeypatch.setattr(p.pickle, "dumps", lambda *a, **k: pytest.fail("no pickle"))
    result = p.execute_entrypoint(
        [Stage("finalize", entry.finalize)],
        entrypoint=ENTRYPOINT,
        params=params(tmp_path),
        image_analysis_results=entry.inputs(),
        targets={},
    )
    assert result.empty


def test_error_archive_closed_on_system_exit():
    stream = BytesIO()
    with pytest.raises(SystemExit), p.writing_tar(stream) as tar:
        p._add(tar, "error", b"failed")
        raise SystemExit(1)
    assert stream.getvalue()[-1024:] == bytes(1024)


def test_input_pickle_failure_cleans_snapshot_directory(monkeypatch):
    directories = []
    original = p.TemporaryDirectory

    def temporary(*args, **kwargs):
        directory = original(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(p, "TemporaryDirectory", temporary)
    inputs = entry.inputs()
    inputs.cells["object"] = [lambda: None] * 4
    with (
        pytest.raises(p.TransferError, match="cannot pickle input lane_id=0"),
        p.file_snapshot(inputs, (Lane(0, "0", "sample"),)),
    ):
        pytest.fail("must fail before yielding snapshot")
    assert directories and all(not path.exists() for path in directories)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing", "missing value"),
        ("duplicate", "duplicate value"),
        ("unknown", "unknown lane"),
        ("corrupt", "cannot unpickle"),
        ("truncated", "end markers"),
        ("nonzero", "code 7"),
    ],
)
def test_bad_worker_response_retains_received_values_and_blocks_barrier(
    tmp_path,
    monkeypatch,
    fault,
    match,
):
    processes = track_children(monkeypatch)
    monkeypatch.setenv("TEST_RESPONSE_FAULT", fault)
    values = {}

    def on_value(segment, context, lane_id, value):
        values[lane_id] = value

    with pytest.raises(p.TransferError, match=match):
        run(tmp_path, on_value=on_value)
    assert values  # accepted completions survive the subsequent failure
    assert all(proc.poll() is not None for proc in processes)
    assert not (tmp_path / "synthesize").exists()
    assert not (tmp_path / "finalize").exists()


def test_truncation_inside_zero_payload_cannot_fake_end_markers():
    # Ending with zeros is insufficient: both end blocks must follow the payload.
    data = archive([("lanes/0/value.pkl", bytes(4096))])[:5632]
    with pytest.raises(p.TransferError, match="end markers"):
        p._read_entries(BytesIO(data), lambda *args: None)


def test_images_arrive_before_worker_can_complete(tmp_path):
    def on_image(data_name, *args):
        (tmp_path / f"imageack-{data_name}").touch()

    assert len(run(tmp_path, "image-ack", on_image=on_image)) == 4


def test_on_value_failure_stops_and_reaps(tmp_path, monkeypatch):
    processes = track_children(monkeypatch)

    def on_value(*args):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        run(tmp_path, on_value=on_value)
    assert all(proc.poll() is not None for proc in processes)
    assert not (tmp_path / "synthesize").exists()


@pytest.mark.parametrize(
    "fault,message",
    [
        ("duplicate-image", "duplicate output TAR entry"),
        ("consecutive-image", "duplicate output TAR entry"),
        ("later-image", "duplicate output TAR entry"),
        ("duplicate-result", "duplicate output TAR entry: same/result_json"),
        ("preprocess", "injected preprocess failure"),
        ("synthesize", "injected synthesize failure"),
        ("finalize", "injected finalize failure"),
        ("unpicklable", "cannot pickle value"),
        ("common-image", None),
    ],
)
def test_streaming_failures_and_unique_names(tmp_path, fault, message):
    proc = subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        capture_output=True,
        timeout=30,
        check=False,
        env=os.environ
        | {
            "ANALYSISRUN_MODE": "parallel-entrypoint-streaming",
            "ANALYSISRUN_MAX_WORKERS": "1",
            "TEST_DIRECTORY": str(tmp_path),
            "TEST_FAULT": fault,
        },
    )
    with tarfile.open(fileobj=BytesIO(proc.stdout)) as tar:
        names = tar.getnames()
        assert len(names) == len(set(names))
        if message:
            assert proc.returncode != 0
            assert names[-1] == "error"
            assert message in tar.extractfile("error").read().decode()
            assert b"^__^" not in proc.stdout
            if fault != "preprocess":
                assert any("/images/" in name for name in names)
            if fault in {
                "preprocess",
                "unpicklable",
                "duplicate-image",
                "consecutive-image",
            }:
                assert not (tmp_path / "synthesize").exists()
            if fault != "duplicate-result":
                assert not (tmp_path / "finalize").exists()
        else:
            assert proc.returncode == 0, proc.stderr.decode()
            assert {f"{lane}/images/common.png" for lane in targets()} <= set(names)
    assert gzip.decompress(proc.stdout)[-1024:] == bytes(1024)


@pytest.mark.parametrize(
    "fault,kind",
    [
        ("preprocess", "preprocess"),
        ("fail-after-image", "map"),
        ("synthesize", "synthesize"),
        ("finalize", "finalize"),
    ],
)
def test_terminal_failure_traceback_then_cowsay(tmp_path, fault, kind):
    proc = subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        capture_output=True,
        timeout=30,
        check=False,
        env=os.environ
        | {
            "ANALYSISRUN_MODE": "parallel-entrypoint",
            "ANALYSISRUN_MAX_WORKERS": "1",
            "TEST_DIRECTORY": str(tmp_path),
            "TEST_FAULT": fault,
        },
    )
    assert proc.returncode != 0
    diagnostic = proc.stderr.decode()
    assert (
        diagnostic.index("Traceback")
        < diagnostic.index("お問い合わせ")
        < diagnostic.index("^__^")
    )
    assert f"stage={kind}" in diagnostic
    assert b"^__^" not in proc.stdout
    if kind == "map":
        assert "sample 0" in diagnostic
        assert (tmp_path / "images" / "lane-0.png").exists()


def test_local_same_image_name_overwrites_across_lanes(tmp_path, monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MAX_WORKERS", "1")
    result = p.execute_entrypoint(
        entry.definition(),
        entrypoint=ENTRYPOINT,
        params=params(tmp_path, "common-image"),
        image_analysis_results=entry.inputs(),
        targets=targets(),
        mode="parallel-entrypoint",
        output_dir=tmp_path / "images",
    )
    assert len(result) == 4
    assert [path.name for path in (tmp_path / "images").iterdir()] == ["common.png"]
    assert (tmp_path / "images" / "common.png").read_bytes().startswith(b"\x89PNG")


def test_gzip_images_are_readable_before_pipeline_finishes(tmp_path):
    from threading import Thread

    proc = subprocess.Popen(
        [sys.executable, str(ENTRYPOINT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ
        | {
            "ANALYSISRUN_MODE": "parallel-entrypoint-streaming",
            "ANALYSISRUN_MAX_WORKERS": "1",
            "TEST_DIRECTORY": str(tmp_path),
            "TEST_FAULT": "image-ack",
        },
    )
    images, errors = [], []

    def consume():
        try:
            with (
                gzip.GzipFile(fileobj=proc.stdout.raw) as compressed,
                tarfile.open(fileobj=compressed, mode="r|", bufsize=512) as tar,
            ):
                for member in tar:
                    data = tar.extractfile(member).read()
                    if "/images/" in member.name:
                        assert proc.poll() is None
                        assert data.startswith(b"\x89PNG")
                        lane = member.name.split("/")[0]
                        images.append(lane)
                        (tmp_path / f"imageack-{lane}").touch()
        except (AssertionError, OSError, tarfile.TarError, EOFError) as exc:
            errors.append(exc)

    consumer = Thread(target=consume)
    consumer.start()
    try:
        proc.wait(timeout=25)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        consumer.join(timeout=5)
    assert not consumer.is_alive()
    assert proc.returncode == 0, proc.stderr.read().decode()
    assert not errors
    assert images == list(targets())


def test_worker_failure_reports_exact_stage_lane_sample_and_stderr(tmp_path):
    with pytest.raises(p.TransferError) as caught:
        run(tmp_path, "fail-after-image", max_workers=1)
    message = str(caught.value)
    for detail in (
        '"stage_index": 2',
        '"lane_id": 0',
        '"sample_name": "sample 0"',
        '"stage_kind": "map"',
        "LookupError",
        "worker diagnostic",
        "stderr=",
    ):
        assert detail in message
    assert "stage=map stage_index=2" in caught.value.__notes__[0]


@pytest.mark.parametrize("field", ["input", "params", "synthesized", "value"])
def test_worker_deserialization_failure_has_lane_context(tmp_path, field):
    stages = entry.definition()
    path = tmp_path / "input.pkl"
    path.write_bytes(b"broken" if field == "input" else pickle.dumps(entry.inputs()))
    req = {
        "protocol_version": 1,
        "manifest": p.canonical_manifest(stages),
        "stage_start": 2,
        "stage_end": 4,
        "lanes": [
            {
                "lane_id": 0,
                "data_name": "0",
                "sample_name": "sample 0",
                "input_path": str(path),
            }
        ],
    }
    data = archive(
        [
            ("request.json", json.dumps(req).encode()),
            ("params.pkl", b"broken" if field == "params" else pickle.dumps({})),
            (
                "synthesized.pkl",
                b"broken" if field == "synthesized" else pickle.dumps(None),
            ),
            (
                "lanes/0/value.pkl",
                b"broken" if field == "value" else pickle.dumps(None),
            ),
        ]
    )
    response = BytesIO()
    assert p.run_worker(stages, BytesIO(data), response) == 1
    with tarfile.open(fileobj=BytesIO(response.getvalue())) as tar:
        assert tar.getnames() == ["error.json"]
        error = json.loads(tar.extractfile("error.json").read())
        assert error["error_type"] == "UnpicklingError"
        assert error["lane_id"] == 0
        assert error["data_name"] == "0"
        assert error["sample_name"] == "sample 0"
        assert error["stage_index"] == (None if field == "value" else 2)
        assert "Traceback" in error["traceback"]
    assert response.getvalue()[-1024:] == bytes(1024)


def test_notebook_output_and_original_failure_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.setattr(p, "get_interactivity", lambda: "notebook")
    monkeypatch.setattr(p.pickle, "dumps", lambda *a: pytest.fail("no pickle"))
    shown = []
    error = LookupError("notebook failure")

    def display(fig, name, image_type, **kwargs):
        assert kwargs == {"format": "png"}
        shown.append((fig, name, image_type))
        entry.plt.close(fig)

    def mapping(args):
        entry.emit_image(args)

    def fail(args):
        raise error

    with pytest.raises(LookupError) as caught:
        p.execute_entrypoint(
            [
                Stage("map", mapping),
                Stage("map", mapping),
                Stage("synthesize", fail),
                Stage("finalize", lambda a: pytest.fail("continued")),
            ],
            entrypoint=ENTRYPOINT,
            params=params(tmp_path, "common-image"),
            image_analysis_results=entry.inputs(),
            targets={"0": "sample 0"},
            output=display,
        )
    assert caught.value is error
    assert "stage=synthesize stage_index=2" in error.__notes__[0]
    assert len(shown) == 2
    assert all(
        name == "common.png" and image_type == "qc" for _, name, image_type in shown
    )
