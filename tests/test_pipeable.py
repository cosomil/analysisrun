from __future__ import annotations

import json
import pickle
import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import BaseModel

from analysisrun.pipeable import (
    ManualInput,
    ProcessedInputs,
    create_image_analysis_results_input_model,
    entity_filter,
    image_analysis_result_spec,
    read_context,
)
from analysisrun.scanner import Fields
from analysisrun.tar import create_tar_from_dict, read_tar_as_dict

DATA_DIR = Path(__file__).parent / "testdata"
IMAGE_ANALYSIS_RESULT_CSV = DATA_DIR / "image_analysis_result.csv"
SAMPLES_CSV = DATA_DIR / "samples.csv"


class Params(BaseModel):
    threshold: int = 1


class ImageResults(NamedTuple):
    activity_spots: Fields = image_analysis_result_spec(
        description="Activity spots",
        cleansing=entity_filter("Activity Spots"),
    )


class RawImageResultsDf(NamedTuple):
    activity_spots: pd.DataFrame


class PreprocessedImageResultsDf(NamedTuple):
    activity_spots: pd.DataFrame


class PreprocessedImageResultsFields(NamedTuple):
    activity_spots: Fields


def _load_pickle_df(path: Path) -> BytesIO:
    df = pd.read_csv(path)
    return _dump_pickle(df)


def _dump_csv(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _dump_pickle(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    pickle.dump(df, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    return buf


def _dump_pickle_obj(value) -> BytesIO:
    buf = BytesIO()
    pickle.dump(value, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    return buf


def _force_interactivity(monkeypatch, value: str | None) -> None:
    """read_context と exit_with_error の両方で同じ interactivity を返すように固定する。"""

    import analysisrun.pipeable as pipeable
    import analysisrun.pipeable_io as pipeable_io

    monkeypatch.setattr(pipeable, "get_interactivity", lambda: value)
    monkeypatch.setattr(pipeable_io, "get_interactivity", lambda: value)


def _write_samples_csv(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    p = tmp_path / "samples.csv"
    df = pd.DataFrame(rows, columns=["data", "sample"])
    df.to_csv(p, index=False)
    return p


def _build_streaming_input_tar(
    rows: list[tuple[str, str]],
    params: Params | None = None,
) -> BytesIO:
    sample_buf = _dump_csv(pd.DataFrame(rows, columns=["data", "sample"]))
    image_buf = _dump_csv(pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV))
    return create_tar_from_dict(
        {
            "params": (params or Params()).model_dump_json(),
            "sample_names": sample_buf,
            "image_analysis_results/activity_spots": image_buf,
        }
    )


def test_create_image_analysis_results_input_model_requires_spec():
    class InvalidImageResults(NamedTuple):
        activity_spots: Fields

    with pytest.raises(ValueError):
        create_image_analysis_results_input_model(InvalidImageResults)


def test_read_context_showschema_outputs_streaming_input_schema(monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MODE", "showschema")

    stdout_buf = BytesIO()

    with pytest.raises(SystemExit) as excinfo:
        read_context(Params, ImageResults, stdout=stdout_buf)

    assert excinfo.value.code == 0

    schema = json.loads(stdout_buf.getvalue().decode("utf-8"))
    assert schema["schema_version"] == "1"
    assert schema["transport"] == {
        "type": "tar",
        "compression": ["tar", "tar.gz"],
        "path_separator": "/",
    }

    tar_entries = {entry["path"]: entry for entry in schema["tar_entries"]}
    assert tar_entries["params"]["required"] is True
    assert tar_entries["params"]["content_type"] == "application/json"
    assert tar_entries["params"]["pax_headers"] == {}
    assert tar_entries["params"]["json_schema"]["type"] == "object"
    assert tar_entries["params"]["json_schema"]["properties"]["threshold"]["type"] == (
        "integer"
    )

    assert tar_entries["sample_names"] == {
        "path": "sample_names",
        "required": True,
        "content_type": "text/csv",
        "description": "サンプル名CSVファイル（サンプル名とレーン番号の対応表）",
        "pax_headers": {"is_file": "true"},
    }
    assert tar_entries["image_analysis_results/activity_spots"] == {
        "path": "image_analysis_results/activity_spots",
        "required": True,
        "content_type": "text/csv",
        "description": "Activity spots",
        "pax_headers": {"is_file": "true"},
    }


def test_run_analysis_sequential_with_manual_input(monkeypatch):
    monkeypatch.setenv("PSEUDO_NBENV", "1")
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)

    manual_input = ManualInput(
        params=Params(threshold=2),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=SAMPLES_CSV,
    )
    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
    )

    def analyze(args):
        df = args.image_analysis_results.activity_spots.data
        assert "Total Well" not in set(df["Entity"])
        return pd.Series({"total_value": int(df["Value"].sum())})

    def postprocess(args):
        df = args.analysis_results.copy()
        df["norm"] = df["total_value"] / df["total_value"].max()
        return df

    result_df = ctx.run_analysis(analyze=analyze, postprocess=postprocess)

    raw = pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV)
    split = raw["Filename"].str.split("_000_", expand=True)
    raw["Data"] = split[1].str.split(".", expand=True)[0]
    expected = (
        raw[raw["Entity"] == "Activity Spots"]
        .groupby("Data")["Value"]
        .sum()
        .sort_index()
    )

    assert list(result_df["sample_name"]) == ["SampleA", "SampleB"]
    totals = result_df.sort_values("data").set_index("data")["total_value"]
    assert totals.to_dict() == expected.to_dict()
    assert "norm" in result_df
    assert result_df["norm"].max() == pytest.approx(1.0)


def test_parallel_entrypoint_streaming_outputs_tar(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(pipeable.os, "cpu_count", lambda: 2)

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        analysis_results: dict[str, BytesIO] = {}
        for data_name, sample_name in targets.items():
            on_image(data_name, "plot.png", b"fake-image", "png")
            analysis_results[data_name] = _dump_csv(
                pd.DataFrame(
                    [
                        {
                            "data": data_name,
                            "sample_name": sample_name,
                            "total_value": 2,
                        }
                    ]
                )
            )
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results=analysis_results,
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    stdout_buf = BytesIO()
    input_tar = _build_streaming_input_tar(
        [("0000", "SampleA"), ("0001", "SampleB")],
        Params(threshold=3),
    )
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(input_tar.getvalue()),
        stdout=stdout_buf,
    )
    assert ctx.mode == "parallel-entrypoint-streaming"

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert "result_csv" in tar_result
    assert "0000" in tar_result and "0001" in tar_result
    assert "images" in tar_result["0000"]
    assert "plot.png" in tar_result["0000"]["images"]
    assert "result_json" in tar_result["0000"]

    csv_df = pd.read_csv(tar_result["result_csv"], dtype=str)
    assert list(csv_df.sort_values("data")["data"]) == ["0000", "0001"]


def test_parallel_entrypoint_streaming_postprocess_print_goes_to_stderr(
    monkeypatch, tmp_path: Path, capsys
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        data_name, sample_name = next(iter(targets.items()))
        on_image(data_name, "plot.png", b"postprocess-image", "png")
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results={
                data_name: _dump_csv(
                    pd.DataFrame(
                        [
                            {
                                "data": data_name,
                                "sample_name": sample_name,
                                "total_value": 4,
                            }
                        ]
                    )
                )
            },
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    stdout_buf = BytesIO()
    input_tar = _build_streaming_input_tar([("0000", "SampleA")], Params())
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(input_tar.getvalue()),
        stdout=stdout_buf,
    )

    def postprocess(args):
        print("Debug: postprocess")
        df = args.analysis_results.copy()
        df["scaled"] = df["total_value"] * args.params.threshold
        return df

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(
            analyze=lambda _: pd.Series({"unused": 0}),
            postprocess=postprocess,
        )
    assert excinfo.value.code == 0

    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    csv_df = pd.read_csv(tar_result["result_csv"])
    assert list(csv_df["scaled"]) == [4]
    captured = capsys.readouterr()
    assert "Debug: postprocess" in captured.err


def test_parallel_entrypoint_streaming_writes_images_before_result_entries(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        data_name, sample_name = next(iter(targets.items()))
        on_image(data_name, "plot.png", b"streamed-image", "png")
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results={
                data_name: _dump_csv(
                    pd.DataFrame(
                        [
                            {
                                "data": data_name,
                                "sample_name": sample_name,
                                "total_value": 5,
                            }
                        ]
                    )
                )
            },
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    stdout_buf = BytesIO()
    input_tar = _build_streaming_input_tar([("0000", "SampleA")], Params())
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(input_tar.getvalue()),
        stdout=stdout_buf,
    )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    assert excinfo.value.code == 0
    names: list[str] = []
    with tarfile.open(fileobj=BytesIO(stdout_buf.getvalue()), mode="r|*") as tar:
        for member in tar:
            if member.isfile():
                names.append(member.name)

    assert "0000/images/plot.png" in names
    assert "result_csv" in names
    assert names.index("0000/images/plot.png") < names.index("result_csv")


def test_parallel_entrypoint_streaming_preserves_leading_zero_values(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        analysis_results: dict[str, BytesIO] = {}
        for data_name, sample_name in targets.items():
            on_image(data_name, "plot.png", b"leading-zero-image", "png")
            analysis_results[data_name] = _dump_csv(
                pd.DataFrame(
                    [
                        {
                            "data": data_name,
                            "sample_name": sample_name,
                            "barcode": "0012" if data_name == "0000" else "0100",
                        }
                    ]
                )
            )
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results=analysis_results,
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    stdout_buf = BytesIO()
    input_tar = _build_streaming_input_tar(
        [("0000", "SampleA"), ("0001", "SampleB")],
        Params(),
    )
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(input_tar.getvalue()),
        stdout=stdout_buf,
    )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    csv_df = pd.read_csv(tar_result["result_csv"], dtype=str)
    assert list(csv_df.sort_values("data")["barcode"]) == ["0012", "0100"]

    first = json.loads(tar_result["0000"]["result_json"].getvalue())
    second = json.loads(tar_result["0001"]["result_json"].getvalue())
    assert first["barcode"] == "0012"
    assert second["barcode"] == "0100"


def test_read_context_noninteractive_invalid_stdin_tar_outputs_error_tar(monkeypatch):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()
    with pytest.raises(SystemExit) as excinfo:
        read_context(
            Params, ImageResults, stdin=BytesIO(b"not a tar"), stdout=stdout_buf
        )

    assert excinfo.value.code == 2
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert (
        tar_result["error"]
        == "入力データの読み込みに失敗しました。入力形式を確認してください。"
    )


@pytest.mark.parametrize("method", ["analyze", "postprocess"])
def test_read_context_unsupported_distributed_mode_outputs_error_tar(
    monkeypatch, method: str
):
    monkeypatch.setenv("ANALYSISRUN_MODE", method)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()
    with pytest.raises(SystemExit) as excinfo:
        read_context(Params, ImageResults, stdin=BytesIO(b""), stdout=stdout_buf)

    assert excinfo.value.code == 2
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert tar_result["error"] == f"未対応のANALYSISRUN_MODEです: {method}"


def test_read_context_notebook_requires_manual_input(monkeypatch):
    """notebook 環境では manual_input 指定が必須。"""

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    _force_interactivity(monkeypatch, "notebook")

    with pytest.raises(RuntimeError) as excinfo:
        read_context(Params, ImageResults, manual_input=None)

    assert "Jupyter notebook環境ではmanual_inputの指定が必須" in str(excinfo.value)


def test_parallel_entrypoint_invokes_subprocess_and_saves_image(
    monkeypatch, tmp_path: Path
):
    """parallel-entrypoint で entrypoint 実行入力を組み立て、画像保存を行う。"""

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, "terminal")

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    samples_csv = _write_samples_csv(tmp_path, [("0000", "SampleA")])
    manual_input = ManualInput(
        params=Params(threshold=1),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=samples_csv,
    )

    called = {"count": 0}

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        called["count"] += 1
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"

        # 入力 tar の基本構造（targets/params）が入っていることだけ確認
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        assert targets == {"0000": "SampleA"}
        assert "params" in tar_in
        assert "image_analysis_results" in tar_in
        activity_spots = tar_in["image_analysis_results"]["activity_spots"]
        assert isinstance(activity_spots, BytesIO)
        restored_df = pickle.loads(activity_spots.getvalue())
        assert isinstance(restored_df, pd.DataFrame)
        assert not restored_df.empty

        on_image("0000", "plot.png", b"fake-image", "png")
        series_csv = b"total_value\n1\n"
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results={"0000": BytesIO(series_csv)},
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    out_dir = tmp_path / "out"
    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
        output_dir=out_dir,
    )

    df = ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))
    assert called["count"] == 1
    assert list(df["data"]) == ["0000"]
    assert list(df["sample_name"]) == ["SampleA"]
    assert (out_dir / "plot.png").exists()


def test_parallel_entrypoint_assigns_targets_evenly_in_order_with_core_limit(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, "terminal")

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)
    monkeypatch.setattr(pipeable.os, "cpu_count", lambda: 2)

    samples_csv = _write_samples_csv(
        tmp_path,
        [
            ("0000", "SampleA"),
            ("0001", "SampleB"),
            ("0002", "SampleC"),
            ("0003", "SampleD"),
            ("0004", "SampleE"),
        ],
    )
    manual_input = ManualInput(
        params=Params(threshold=1),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=samples_csv,
    )

    expected_rows = len(pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV))
    calls: list[list[tuple[str, str]]] = []

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"

        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        target_pairs = list(targets.items())
        calls.append(target_pairs)

        activity_spots = tar_in["image_analysis_results"]["activity_spots"]
        assert isinstance(activity_spots, BytesIO)
        restored_df = pickle.loads(activity_spots.getvalue())
        assert isinstance(restored_df, pd.DataFrame)
        assert len(restored_df) == expected_rows

        analysis_results: dict[str, BytesIO] = {}
        for data_name, sample_name in target_pairs:
            on_image(data_name, "plot.png", b"fake-image", "png")
            analysis_results[data_name] = _dump_csv(
                pd.DataFrame(
                    [
                        {
                            "data": data_name,
                            "sample_name": sample_name,
                            "total_value": 1,
                        }
                    ]
                )
            )
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results=analysis_results,
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
        output_dir=tmp_path / "out",
    )

    result_df = ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    assert len(calls) == 2
    calls_sorted = sorted(calls, key=lambda x: x[0][0])
    assert calls_sorted == [
        [("0000", "SampleA"), ("0001", "SampleB"), ("0002", "SampleC")],
        [("0003", "SampleD"), ("0004", "SampleE")],
    ]

    got = result_df.sort_values("data").reset_index(drop=True)
    assert list(got["data"]) == ["0000", "0001", "0002", "0003", "0004"]
    assert list(got["sample_name"]) == [
        "SampleA",
        "SampleB",
        "SampleC",
        "SampleD",
        "SampleE",
    ]
    assert list(got["total_value"]) == [1, 1, 1, 1, 1]


def test_parallel_entrypoint_error_tar_even_when_returncode_zero(
    monkeypatch, tmp_path: Path
):
    """returncode==0 でも tar に error があれば失敗扱い。"""

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, "terminal")

    import analysisrun.pipeable as pipeable

    stderr_buf = BytesIO()

    class _DummyStderr:
        def __init__(self, buf: BytesIO):
            self.buffer = buf

    monkeypatch.setattr(pipeable.sys, "stderr", _DummyStderr(stderr_buf))

    def _raise_original(code, message, stdout, stderr, exception=None):
        raise RuntimeError(message)

    monkeypatch.setattr(pipeable, "exit_with_error", _raise_original)

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    samples_csv = _write_samples_csv(tmp_path, [("0000", "SampleA")])
    manual_input = ManualInput(
        params=Params(threshold=1),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=samples_csv,
    )

    child_error = "error despite rc=0"

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results={},
            error=child_error,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
        output_dir=tmp_path / "out",
    )

    with pytest.raises(RuntimeError) as excinfo:
        ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    assert "エラーが発生しました" in str(excinfo.value)
    stderr_text = stderr_buf.getvalue().decode("utf-8")
    assert child_error in stderr_text


def test_parallel_entrypoint_error_tar_outputs_lane_message_and_saves_images(
    monkeypatch, tmp_path: Path
):
    """エラー tar でもレーンごとのメッセージを出力し、画像を保存する。"""

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, "terminal")

    import analysisrun.pipeable as pipeable

    stderr_buf = BytesIO()

    class _DummyStderr:
        def __init__(self, buf: BytesIO):
            self.buffer = buf

    monkeypatch.setattr(pipeable.sys, "stderr", _DummyStderr(stderr_buf))

    def _raise_original(code, message, stdout, stderr, exception=None):
        raise RuntimeError(message)

    monkeypatch.setattr(pipeable, "exit_with_error", _raise_original)

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    samples_csv = _write_samples_csv(
        tmp_path, [("0000", "SampleA"), ("0001", "SampleB")]
    )
    manual_input = ManualInput(
        params=Params(threshold=1),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=samples_csv,
    )

    child_error = "child error with image"

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        if "0000" in targets:
            on_image("0000", "error_plot.png", b"error-image", "png")
            return SimpleNamespace(
                returncode=1,
                stderr="",
                analysis_results={},
                error=child_error,
                tar_read_ok=True,
            )
        data_name, sample_name = next(iter(targets.items()))
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results={
                data_name: _dump_csv(
                    pd.DataFrame(
                        [
                            {
                                "data": data_name,
                                "sample_name": sample_name,
                                "total_value": 2,
                            }
                        ]
                    )
                )
            },
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    out_dir = tmp_path / "out"
    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
        output_dir=out_dir,
    )

    with pytest.raises(RuntimeError):
        ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    stderr_text = stderr_buf.getvalue().decode("utf-8")
    assert "0000 (SampleA)" in stderr_text
    assert child_error in stderr_text
    assert (out_dir / "error_plot.png").exists()


def test_run_analyze_seq_outputs_tar_with_multiple_targets_sequential(
    monkeypatch,
):
    """
    ANALYSISRUN_MODE=analyzeseq で複数ターゲットの解析がシーケンシャルに実行され、
    正しいフォーマットで tar 出力されることを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyzeseq")
    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "targets": json.dumps({"0000": "SampleA", "0001": "SampleB"}),
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_pickle_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    call_order = []

    def analyze(args):
        # 呼び出し順序を記録
        call_order.append((args.data_name, args.sample_name))

        df = args.image_analysis_results.activity_spots.data

        # 画像を出力
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        args.output(fig, "plot.png", "png")

        # 各ターゲットごとに異なる値を返す
        total = int(df["Value"].sum())
        return pd.Series(
            {
                "data": args.data_name,
                "sample_name": args.sample_name,
                "total_value": total,
            }
        )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze)

    assert excinfo.value.code == 0

    # シーケンシャル実行なので、呼び出し順序が保証される
    assert call_order == [("0000", "SampleA"), ("0001", "SampleB")]

    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))

    # Verify output structure: {data_name}/analysis_result
    assert "0000" in tar_result
    assert "0001" in tar_result
    assert "analysis_result" in tar_result["0000"]
    assert "analysis_result" in tar_result["0001"]

    # Verify analysis results
    result_0000 = pd.read_csv(
        tar_result["0000"]["analysis_result"],
        dtype={"data": str, "sample_name": str},
    )
    assert result_0000.iloc[0]["data"] == "0000"
    assert result_0000.iloc[0]["sample_name"] == "SampleA"
    assert result_0000.iloc[0]["total_value"] > 0

    result_0001 = pd.read_csv(
        tar_result["0001"]["analysis_result"],
        dtype={"data": str, "sample_name": str},
    )
    assert result_0001.iloc[0]["data"] == "0001"
    assert result_0001.iloc[0]["sample_name"] == "SampleB"
    assert result_0001.iloc[0]["total_value"] > 0

    # Verify output structure: {data_name}/images/{image_name}
    assert "images" in tar_result["0000"]
    assert "images" in tar_result["0001"]
    assert "plot.png" in tar_result["0000"]["images"]
    assert "plot.png" in tar_result["0001"]["images"]

    # Verify images
    assert isinstance(tar_result["0000"]["images"]["plot.png"], BytesIO)
    assert tar_result["0000"]["images"]["plot.png"].getbuffer().nbytes > 0
    assert isinstance(tar_result["0001"]["images"]["plot.png"], BytesIO)
    assert tar_result["0001"]["images"]["plot.png"].getbuffer().nbytes > 0


def test_run_analyze_seq_handles_target_failures_immediate(monkeypatch, capsys):
    """
    ANALYSISRUN_MODE=analyzeseq で一部のターゲットが失敗した場合、
    即座に処理が終了することを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyzeseq")
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "targets": json.dumps(
                {"0000": "SampleA", "0001": "SampleB", "0002": "SampleC"}
            ),
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_pickle_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    call_order = []

    def analyze(args):
        call_order.append(args.data_name)

        # 2番目のターゲット（0001）で失敗
        if args.data_name == "0001":
            raise ValueError(f"Intentional failure for {args.data_name}")

        df = args.image_analysis_results.activity_spots.data

        # 画像を出力
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        args.output(fig, "error_plot.png", "png")

        return pd.Series(
            {
                "data": args.data_name,
                "sample_name": args.sample_name,
                "total_value": int(df["Value"].sum()),
            }
        )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze)

    # Should exit with error code
    assert excinfo.value.code == 1

    # エラー発生時点で即座に終了するため、3番目のターゲット（0002）は処理されない
    assert call_order == ["0000", "0001"]

    # Verify error output in tar
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert "error" in tar_result
    error_msg = tar_result["error"]
    # エラーメッセージには、ターゲット名とサンプル名が含まれる
    assert "0001" in error_msg
    assert "SampleB" in error_msg
    assert "解析処理中にエラーが発生しました" in error_msg

    # 元の例外メッセージはstderrに出力される
    captured = capsys.readouterr()
    assert "Intentional failure for 0001" in captured.err

    # エラー前に生成された画像（0000のもの）は含まれるべき
    assert "0000" in tar_result
    assert "images" in tar_result["0000"]
    assert "error_plot.png" in tar_result["0000"]["images"]


def test_run_analyze_seq_with_print_statements_doesnt_corrupt_output(
    monkeypatch, capsys
):
    """
    analyzeseq モード中にprint文があっても標準出力が破損しないことを確認する。

    print文の出力が標準出力に混入するとtarフォーマットが破損してパースできなくなるため、
    print文が標準エラー出力に向かうことを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyzeseq")
    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "targets": json.dumps({"0000": "SampleA", "0001": "SampleB"}),
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_pickle_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    def analyze(args):
        # ユーザーコードにprint文が含まれているケース
        print(f"Debug: Analyzing {args.data_name}")
        df = args.image_analysis_results.activity_spots.data
        print(f"Debug: Processing {len(df)} rows")
        total = int(df["Value"].sum())
        print(f"Debug: Total value is {total}")
        return pd.Series({"total_value": total})

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze)

    assert excinfo.value.code == 0

    # 標準出力はtarフォーマットとして正常に読み込めるべき
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))

    # 結果は正しく取得できるべき
    assert "0000" in tar_result
    assert "0001" in tar_result
    assert "analysis_result" in tar_result["0000"]
    assert "analysis_result" in tar_result["0001"]

    # print文の出力は標準エラー出力に出ているべき
    captured = capsys.readouterr()
    assert "Debug: Analyzing 0000" in captured.err
    assert "Debug: Analyzing 0001" in captured.err
    assert "Debug: Processing" in captured.err
    assert "Debug: Total value is" in captured.err


def test_run_analysis_with_preprocess_sequential_with_manual_input(monkeypatch):
    monkeypatch.setenv("PSEUDO_NBENV", "1")
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)

    manual_input = ManualInput(
        params=Params(threshold=2),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=SAMPLES_CSV,
    )
    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
    )
    calls = {"count": 0}

    def preprocess(args):
        calls["count"] += 1
        df = args.image_analysis_results.activity_spots
        df["DoubleValue"] = df["Value"] * 2
        return ProcessedInputs(
            image_analysis_results=PreprocessedImageResultsDf(activity_spots=df),
            extra={"row_count": int(len(df)), "threshold": int(args.params.threshold)},
        )

    def analyze(args):
        df = args.image_analysis_results.activity_spots.data
        return pd.Series(
            {
                "total_value": int(df["DoubleValue"].sum()),
                "row_count": args.extra["row_count"],
            }
        )

    def postprocess(args):
        df = args.analysis_results.copy()
        df["pre_threshold"] = args.extra["threshold"]
        return df

    result_df = ctx.run_analysis_with_preprocess(
        raw_image_analysis_results=RawImageResultsDf,
        preprocessed_image_analysis_results_df=PreprocessedImageResultsDf,
        preprocessed_image_analysis_results_fields=PreprocessedImageResultsFields,
        preprocess=preprocess,
        analyze=analyze,
        postprocess=postprocess,
    )

    assert set(result_df["data"]) == {"0000", "0001"}
    assert all(result_df["pre_threshold"] == 2)
    assert all(result_df["row_count"] > 0)
    assert calls["count"] == 1


def test_run_analysis_with_preprocess_parallel_entrypoint_streaming_outputs_tar(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        preprocessed = pickle.loads(tar_in["preprocessed_data"].getvalue())
        assert preprocessed["multiplier"] == 3

        analysis_results: dict[str, BytesIO] = {}
        for data_name, sample_name in targets.items():
            on_image(data_name, "plot.png", b"preprocessed-image", "png")
            analysis_results[data_name] = _dump_csv(
                pd.DataFrame(
                    [
                        {
                            "data": data_name,
                            "sample_name": sample_name,
                            "total_value": 2,
                        }
                    ]
                )
            )
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results=analysis_results,
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    stdout_buf = BytesIO()
    input_tar = _build_streaming_input_tar(
        [("0000", "SampleA"), ("0001", "SampleB")],
        Params(),
    )
    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(input_tar.getvalue()),
        stdout=stdout_buf,
    )

    def preprocess(args):
        return ProcessedInputs(
            image_analysis_results=PreprocessedImageResultsDf(
                activity_spots=args.image_analysis_results.activity_spots
            ),
            extra={"multiplier": 3},
        )

    def postprocess(args):
        df = args.analysis_results.copy()
        df["scaled"] = df["total_value"] * args.extra["multiplier"]
        return df

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis_with_preprocess(
            raw_image_analysis_results=RawImageResultsDf,
            preprocessed_image_analysis_results_df=PreprocessedImageResultsDf,
            preprocessed_image_analysis_results_fields=PreprocessedImageResultsFields,
            preprocess=preprocess,
            analyze=lambda _: pd.Series({"unused": 0}),
            postprocess=postprocess,
        )

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    csv_df = pd.read_csv(tar_result["result_csv"])
    assert list(csv_df.sort_values("data")["scaled"]) == [6, 6]


def test_run_analysis_with_preprocess_parallel_entrypoint_collects_preprocessed_data(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, "terminal")

    import analysisrun.pipeable as pipeable

    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")
    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    samples_csv = _write_samples_csv(
        tmp_path, [("0000", "SampleA"), ("0001", "SampleB")]
    )
    manual_input = ManualInput(
        params=Params(threshold=1),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=samples_csv,
    )
    calls = {"count": 0}

    def fake_run_stream(entrypoint_path, tar_buf, mode, on_image):
        assert entrypoint_path == entrypoint
        assert mode == "analyzeseq"
        tar_in = read_tar_as_dict(tar_buf)
        targets = json.loads(tar_in["targets"])
        preprocessed_data = pickle.loads(tar_in["preprocessed_data"].getvalue())
        assert isinstance(preprocessed_data, dict)
        assert "multipliers" in preprocessed_data
        analysis_results: dict[str, BytesIO] = {}
        for data_name, sample_name in targets.items():
            on_image(data_name, "plot.png", b"preprocessed-image", "png")
            analysis_results[data_name] = _dump_csv(
                pd.DataFrame(
                    [
                        {
                            "data": data_name,
                            "sample_name": sample_name,
                            "total_value": 2,
                        }
                    ]
                )
            )
        return SimpleNamespace(
            returncode=0,
            stderr="",
            analysis_results=analysis_results,
            error=None,
            tar_read_ok=True,
        )

    monkeypatch.setattr(
        pipeable,
        "_run_entrypoint_with_tar_streaming",
        fake_run_stream,
    )

    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
        output_dir=tmp_path / "out",
    )

    def postprocess(args):
        df = args.analysis_results.copy()
        df["scaled"] = df.apply(
            lambda row: row["total_value"] * args.extra["multipliers"][row["data"]],
            axis=1,
        )
        return df

    def preprocess(args):
        calls["count"] += 1
        return ProcessedInputs(
            image_analysis_results=PreprocessedImageResultsDf(
                activity_spots=args.image_analysis_results.activity_spots
            ),
            extra={
                "multipliers": {
                    data_name: 5 + i * 2 for i, data_name in enumerate(args.targets)
                }
            },
        )

    result_df = ctx.run_analysis_with_preprocess(
        raw_image_analysis_results=RawImageResultsDf,
        preprocessed_image_analysis_results_df=PreprocessedImageResultsDf,
        preprocessed_image_analysis_results_fields=PreprocessedImageResultsFields,
        preprocess=preprocess,
        analyze=lambda _: pd.Series({"unused": 0}),
        postprocess=postprocess,
    ).sort_values("data")

    assert list(result_df["data"]) == ["0000", "0001"]
    assert list(result_df["scaled"]) == [10, 14]
    assert calls["count"] == 1
