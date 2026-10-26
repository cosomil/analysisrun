"""公開 QC サンプルの最終成果物を各実行モードで検証する。"""

import gzip
import importlib
import json
import os
import subprocess
import sys
import tarfile
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from analysisrun.tar import create_tar_from_dict
from tests.support import read_tar_member

example = importlib.import_module("example.qc-feedback.__main__")
ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path: Path, values: list[int], names: list[str]) -> pd.DataFrame:
    lanes = [f"{i:04d}" for i in range(len(values))]
    pd.DataFrame(
        {
            "MultiPointIndex": 1,
            "Entity": "Activity Spots",
            "Value": values,
            "Filename": [f"Measurement_000_{lane}.nd2" for lane in lanes],
        }
    ).to_csv(tmp_path / "measurements.csv", index=False)
    pd.DataFrame({"data": lanes, "sample": names}).to_csv(
        tmp_path / "samples.csv", index=False
    )
    return pd.DataFrame(
        {
            "data": lanes,
            "sample_name": names,
            "measurement": [float(value) for value in values],
        }
    )


def run_script(tmp_path: Path, mode: str, *, failure: bool = False):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ANALYSISRUN_") and key != "PSEUDO_NBENV"
    }
    env.update(
        QC_INPUT_DIR=str(tmp_path),
        QC_OUTPUT_DIR=str(tmp_path / "output"),
        ANALYSISRUN_MAX_WORKERS="2",
        MPLBACKEND="Agg",
    )
    payload = b""
    if mode == "interactive":
        env["ANALYSISRUN_INTERACTIVITY"] = "terminal"
    else:
        payload = create_tar_from_dict(
            {
                "params": example.Params().model_dump_json(),
                "sample_names": BytesIO((tmp_path / "samples.csv").read_bytes()),
                "image_analysis_results/measurements": BytesIO(
                    (tmp_path / "measurements.csv").read_bytes()
                ),
            }
        ).getvalue()
    script = (
        ROOT / "tests/qc_failure_entry.py"
        if failure
        else ROOT / "example/qc-feedback/__main__.py"
    )
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        input=payload,
        capture_output=True,
        timeout=60,
        check=False,
    )


def archive_entries(payload: bytes) -> dict[str, bytes]:
    # gzip の CRC / footer と TAR の終端ブロックも確認する。
    raw = gzip.decompress(payload)
    assert len(raw) % 512 == 0
    with tarfile.open(fileobj=BytesIO(raw), mode="r:") as archive:
        entries = {member.name: read_tar_member(archive, member) for member in archive}
        assert raw[archive.offset : archive.offset + 1024] == b"\0" * 1024
    return entries


@pytest.mark.parametrize("mode", ["notebook", "interactive", "streaming"])
@pytest.mark.parametrize(
    "values,names,scores,analysis_ok,qc_ok",
    [
        (
            [10, 10, 12, 4],
            ["QC", "QC", "A", "B"],
            [1.0, 1.0, 1.2, 0.4],
            [True, True, True, False],
            True,
        ),
        (
            [10, 4, 10, 12],
            ["QC", "QC", "A", "B"],
            [1.0, 0.4, 1.0, 1.2],
            [True, False, True, True],
            False,
        ),
        ([4, 12], ["A", "B"], [0.5, 1.5], [False, True], False),
        ([8], ["QC"], [1.0], [True], True),
        ([12, 8, 4], ["A", "QC", "QC"], [1.5, 1.0, 0.5], [True, True, False], False),
    ],
    ids=["qc-pass", "qc-partial-failure", "no-qc", "single-lane", "three-lanes"],
)
def test_qc_feedback_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    values: list[int],
    names: list[str],
    scores: list[float],
    analysis_ok: list[bool],
    qc_ok: bool,
):
    expected = inputs(tmp_path, values, names).assign(
        score=scores,
        analysis_ok=analysis_ok,
        qc_ok=qc_ok,
        ok=[ok and qc_ok for ok in analysis_ok],
    )
    if mode == "notebook":
        monkeypatch.setenv("ANALYSISRUN_INTERACTIVITY", "notebook")
        monkeypatch.setenv("QC_INPUT_DIR", str(tmp_path))
        monkeypatch.setenv("QC_OUTPUT_DIR", str(tmp_path / "output"))
        shown = []

        def show(*, block: bool | None = None) -> None:
            shown.append(plt.gcf())

        monkeypatch.setattr(plt, "show", show)
        result = example.main()
        assert len(shown) == len(values)
        assert not list((tmp_path / "output").glob("*.png"))
    else:
        completed = run_script(tmp_path, mode)
        assert completed.returncode == 0, completed.stderr.decode()
        if mode == "streaming":
            entries = archive_entries(completed.stdout)
            expected_names = {"result_csv"}
            for row in expected.astype(str).to_dict("records"):
                lane = row["data"]
                expected_names.update(
                    {f"{lane}/images/{lane}_qc.png", f"{lane}/result_json"}
                )
                assert json.loads(entries[f"{lane}/result_json"]) == row
            assert set(entries) == expected_names
            images = [
                content for name, content in entries.items() if name.endswith(".png")
            ]
            result = pd.read_csv(BytesIO(entries["result_csv"]), dtype={"data": str})
            assert not (tmp_path / "output").exists()
        else:
            assert completed.stdout == b""
            result = pd.read_csv(tmp_path / "output/result.csv", dtype={"data": str})
            images = [
                (tmp_path / f"output/{lane}_qc.png").read_bytes()
                for lane in expected.data
            ]
        assert len(images) == len(values)
        assert all(content.startswith(b"\x89PNG\r\n\x1a\n") for content in images)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_qc_notebook_preserves_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    inputs(tmp_path, [10], ["FAIL_AFTER_OUTPUT"])
    monkeypatch.setenv("ANALYSISRUN_INTERACTIVITY", "notebook")
    monkeypatch.setenv("QC_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv("QC_OUTPUT_DIR", str(tmp_path / "output"))
    shown = []
    monkeypatch.setattr(plt, "show", lambda **kwargs: shown.append(plt.gcf()))
    original = example.apply_qc_evaluation
    error = LookupError("QC report failure after output")

    def fail(args: Any) -> NoReturn:
        original(args)
        raise error

    monkeypatch.setattr(example, "apply_qc_evaluation", fail)
    with pytest.raises(LookupError) as caught:
        example.main()
    assert caught.value is error
    assert traceback.extract_tb(error.__traceback__)[-1].name == "fail"
    assert len(shown) == 1
    assert not (tmp_path / "output").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("mode", ["interactive", "streaming"])
def test_qc_script_failure_retains_partial_images(tmp_path: Path, mode: str):
    inputs(tmp_path, [10, 12], ["QC", "FAIL_AFTER_OUTPUT"])
    completed = run_script(tmp_path, mode, failure=True)
    assert completed.returncode != 0
    stderr = completed.stderr.decode()
    assert "QC report failure after output" in stderr
    assert "Traceback (most recent call last)" in stderr
    assert not (tmp_path / "output/result.csv").exists()
    if mode == "interactive":
        assert stderr.index("Traceback (most recent call last)") < stderr.index("^__^")
        assert completed.stdout == b""
        assert (tmp_path / "output/0001_qc.png").read_bytes().startswith(b"\x89PNG")
    else:
        entries = archive_entries(completed.stdout)
        assert list(entries)[-1] == "error"
        assert b"QC report failure after output" in entries["error"]
        assert entries["0001/images/0001_qc.png"].startswith(b"\x89PNG")
        assert "result_csv" not in entries
        assert not any(name.endswith("result_json") for name in entries)
