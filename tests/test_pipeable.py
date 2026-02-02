from __future__ import annotations

import json
import sys
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


def _load_csv_df(path: Path) -> BytesIO:
    df = pd.read_csv(path)
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _dump_csv(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
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


def test_create_image_analysis_results_input_model_requires_spec():
    class InvalidImageResults(NamedTuple):
        activity_spots: Fields

    with pytest.raises(ValueError):
        create_image_analysis_results_input_model(InvalidImageResults)


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
    totals = result_df.sort_values("data_name").set_index("data_name")["total_value"]
    assert totals.to_dict() == expected.to_dict()
    assert "norm" in result_df
    assert result_df["norm"].max() == pytest.approx(1.0)


def test_run_analysis_only_outputs_tar(monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyze")
    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "data_name": "0000",
            "sample_name": "SampleA",
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_csv_df(
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
        df = args.image_analysis_results.activity_spots.data
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        args.output(fig, "plot.png", "png")
        return pd.Series({"total_value": int(df["Value"].sum())})

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze)

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))

    series_buf = tar_result["analysis_result"]
    assert isinstance(series_buf, BytesIO)
    df = pd.read_csv(series_buf, dtype={"data_name": str, "sample_name": str})
    series = df.iloc[0]
    filtered = pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV)
    filtered = filtered[filtered["Entity"] == "Activity Spots"]
    filtered = filtered[filtered["Filename"].str.contains("0000")]
    assert series["total_value"] == int(filtered["Value"].sum())
    assert series["data_name"] == "0000"
    assert series["sample_name"] == "SampleA"

    images = tar_result["images"]
    assert "plot.png" in images
    assert isinstance(images["plot.png"], BytesIO)
    assert images["plot.png"].getbuffer().nbytes > 0


def test_run_analysis_with_print_statements_doesnt_corrupt_output(monkeypatch, capsys):
    """
    解析処理中にprint文があっても標準出力が破損しないことを確認する。

    print文の出力が標準出力に混入するとtarフォーマットが破損してパースできなくなるため、
    print文が標準エラー出力に向かうことを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyze")
    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "data_name": "0000",
            "sample_name": "SampleA",
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_csv_df(
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
        print("Debug: Starting analysis")
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

    series_buf = tar_result["analysis_result"]
    assert isinstance(series_buf, BytesIO)
    df = pd.read_csv(series_buf, dtype={"data_name": str, "sample_name": str})
    series = df.iloc[0]

    # 結果は正しく取得できるべき
    filtered = pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV)
    filtered = filtered[filtered["Entity"] == "Activity Spots"]
    filtered = filtered[filtered["Filename"].str.contains("0000")]
    assert series["total_value"] == int(filtered["Value"].sum())

    # print文の出力は標準エラー出力に出ているべき
    captured = capsys.readouterr()
    assert "Debug: Starting analysis" in captured.err
    assert "Debug: Processing" in captured.err
    assert "Debug: Total value is" in captured.err


def test_run_postprocess_only_outputs_tar(monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MODE", "postprocess")
    stdout_buf = BytesIO()

    analysis_results = pd.DataFrame(
        [
            {"data_name": "0000", "sample_name": "SampleA", "total_value": 4},
            {"data_name": "0001", "sample_name": "SampleB", "total_value": 6},
        ]
    )

    tar_buf = create_tar_from_dict(
        {
            "analysis_results": {
                "0": _dump_csv(analysis_results.iloc[[0]]),
                "1": _dump_csv(analysis_results.iloc[[1]]),
            },
            "params": Params(threshold=5).model_dump_json(),
        }
    )

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    def analyze(args):
        raise RuntimeError("analyze should not be called in postprocess only mode")

    def postprocess(args):
        df = args.analysis_results.copy()
        df["scaled"] = df["total_value"] * args.params.threshold
        return df

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze, postprocess=postprocess)

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    csv_buf = tar_result["result_csv"]
    assert isinstance(csv_buf, BytesIO)
    csv_buf.seek(0)
    csv_df = pd.read_csv(csv_buf)
    assert list(csv_df["scaled"]) == [20, 30]

    json_entries = tar_result["result_json"]
    assert set(json_entries.keys()) == {"0000", "0001"}
    first = json.loads(json_entries["0000"].getvalue())
    assert first["scaled"] == "20"  # すべて文字列となる


def test_run_postprocess_with_print_statements_doesnt_corrupt_output(
    monkeypatch, capsys
):
    """
    後処理中にprint文があっても標準出力が破損しないことを確認する。

    print文の出力が標準出力に混入するとtarフォーマットが破損してパースできなくなるため、
    print文が標準エラー出力に向かうことを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "postprocess")
    stdout_buf = BytesIO()

    analysis_results = pd.DataFrame(
        [
            {"data_name": "0000", "sample_name": "SampleA", "total_value": 4},
            {"data_name": "0001", "sample_name": "SampleB", "total_value": 6},
        ]
    )
    tar_buf = create_tar_from_dict(
        {
            "analysis_results": {
                "0": _dump_csv(analysis_results.iloc[[0]]),
                "1": _dump_csv(pd.DataFrame(analysis_results.iloc[[1]])),
            },
            "params": Params(threshold=5).model_dump_json(),
        }
    )

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    def analyze(args):
        raise RuntimeError("analyze should not be called in postprocess only mode")

    def postprocess(args):
        print("Debug: Starting postprocess")
        df = args.analysis_results.copy()
        print(f"Debug: Processing {len(df)} results")
        df["scaled"] = df["total_value"] * args.params.threshold
        print("Debug: Scaled values calculated")
        return df

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=analyze, postprocess=postprocess)

    assert excinfo.value.code == 0

    # 標準出力はtarフォーマットとして正常に読み込めるべき
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    csv_buf = tar_result["result_csv"]
    assert isinstance(csv_buf, BytesIO)
    csv_buf.seek(0)
    csv_df = pd.read_csv(csv_buf)
    assert list(csv_df["scaled"]) == [20, 30]

    # print文の出力は標準エラー出力に出ているべき
    captured = capsys.readouterr()
    assert "Debug: Starting postprocess" in captured.err
    assert "Debug: Processing 2 results" in captured.err
    assert "Debug: Scaled values calculated" in captured.err


def test_run_postprocess_only_preserves_leading_zero_values(monkeypatch):
    """postprocess-only で leading-zero を含む列が落ちないことを確認する。"""

    monkeypatch.setenv("ANALYSISRUN_MODE", "postprocess")
    stdout_buf = BytesIO()

    analysis_results = pd.DataFrame(
        [
            {
                "data_name": "0000",
                "sample_name": "SampleA",
                "barcode": "0012",
                "total_value": 4,
            },
            {
                "data_name": "0001",
                "sample_name": "SampleB",
                "barcode": "0100",
                "total_value": 6,
            },
        ]
    )

    tar_buf = create_tar_from_dict(
        {
            "analysis_results": {
                "0": _dump_csv(analysis_results.iloc[[0]]),
                "1": _dump_csv(analysis_results.iloc[[1]]),
            },
            "params": Params(threshold=5).model_dump_json(),
        }
    )

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))

    csv_buf = tar_result["result_csv"]
    assert isinstance(csv_buf, BytesIO)
    csv_buf.seek(0)
    csv_df = pd.read_csv(csv_buf, dtype=str)
    assert list(csv_df["barcode"]) == ["0012", "0100"]

    json_entries = tar_result["result_json"]
    assert set(json_entries.keys()) == {"0000", "0001"}
    first = json.loads(json_entries["0000"].getvalue())
    second = json.loads(json_entries["0001"].getvalue())
    assert first["barcode"] == "0012"
    assert second["barcode"] == "0100"


def test_read_context_noninteractive_requires_method_outputs_error_tar(monkeypatch):
    """非対話かつ ANALYSISRUN_MODE 未指定なら、stdout に error tar を返して終了する。"""

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()
    with pytest.raises(SystemExit) as excinfo:
        read_context(Params, ImageResults, stdin=BytesIO(b""), stdout=stdout_buf)

    assert excinfo.value.code == 2
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert "error" in tar_result
    assert (
        tar_result["error"]
        == "ANALYSISRUN_MODE環境変数に実行モードが指定されていません。"
    )


@pytest.mark.parametrize("method", ["analyze", "postprocess"])
def test_read_context_invalid_stdin_tar_in_distributed_mode(monkeypatch, method: str):
    """analyze/postprocess モードで stdin の tar が壊れている場合は invalid usage として扱う。"""

    monkeypatch.setenv("ANALYSISRUN_MODE", method)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()
    with pytest.raises(SystemExit) as excinfo:
        read_context(
            Params,
            ImageResults,
            stdin=BytesIO(b"this is not a tar"),
            stdout=stdout_buf,
        )

    assert excinfo.value.code == 2
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert "error" in tar_result
    assert (
        tar_result["error"]
        == "入力データの読み込みに失敗しました。入力形式を確認してください。"
    )


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
    """parallel-entrypoint で subprocess を呼び、env 注入と画像保存を行う。"""

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

    def fake_run(cmd, *, input, stdout, stderr, env):
        called["count"] += 1
        assert cmd == [sys.executable, str(entrypoint)]
        assert env.get("ANALYSISRUN_MODE") == "analyze"

        # 入力 tar の基本構造（data_name/sample_name/params）が入っていることだけ確認
        tar_in = read_tar_as_dict(BytesIO(input))
        assert tar_in["data_name"] == "0000"
        assert tar_in["sample_name"] == "SampleA"
        assert "params" in tar_in
        assert "image_analysis_results" in tar_in

        series_csv = b"total_value\n1\n"
        tar_out = create_tar_from_dict(
            {
                "analysis_result": BytesIO(series_csv),
                "images/plot.png": BytesIO(b"fake-image"),
            }
        )
        return SimpleNamespace(returncode=0, stdout=tar_out.getvalue(), stderr=b"")

    monkeypatch.setattr(pipeable.subprocess, "run", fake_run)

    out_dir = tmp_path / "out"
    ctx = read_context(
        Params,
        ImageResults,
        manual_input=manual_input,
        output_dir=out_dir,
    )

    df = ctx.run_analysis(analyze=lambda _: pd.Series({"unused": 0}))
    assert called["count"] == 1
    assert list(df["data_name"]) == ["0000"]
    assert list(df["sample_name"]) == ["SampleA"]
    assert (out_dir / "plot.png").exists()


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
    error_tar = create_tar_from_dict({"error": child_error})

    def fake_run(cmd, *, input, stdout, stderr, env):
        return SimpleNamespace(returncode=0, stdout=error_tar.getvalue(), stderr=b"")

    monkeypatch.setattr(pipeable.subprocess, "run", fake_run)

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
    error_tar = create_tar_from_dict(
        {
            "error": child_error,
            "images/error_plot.png": BytesIO(b"error-image"),
        }
    )
    series_csv = b"total_value\n2\n"
    ok_tar = create_tar_from_dict({"analysis_result": BytesIO(series_csv)})

    def fake_run(cmd, *, input, stdout, stderr, env):
        tar_in = read_tar_as_dict(BytesIO(input))
        if tar_in["data_name"] == "0000":
            return SimpleNamespace(
                returncode=1, stdout=error_tar.getvalue(), stderr=b""
            )
        return SimpleNamespace(returncode=0, stdout=ok_tar.getvalue(), stderr=b"")

    monkeypatch.setattr(pipeable.subprocess, "run", fake_run)

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


def test_run_analyze_multi_outputs_tar_with_multiple_targets(
    monkeypatch, tmp_path: Path
):
    """
    ANALYSISRUN_MODE=analyzemulti で複数ターゲットの解析が並列実行され、
    正しいフォーマットで tar 出力されることを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyzemulti")
    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "targets": json.dumps({"0000": "SampleA", "0001": "SampleB"}),
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_csv_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )

    # Setup entrypoint mock
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")

    import analysisrun.pipeable as pipeable

    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    # Mock subprocess.run to simulate successful analyze mode execution
    def fake_run(cmd, *, input, stdout, stderr, env):
        tar_in = read_tar_as_dict(BytesIO(input))
        data_name = tar_in["data_name"]
        sample_name = tar_in["sample_name"]

        # Simulate analysis result
        series_df = pd.DataFrame(
            {
                "data_name": [data_name],
                "sample_name": [sample_name],
                "total_value": [100 if data_name == "0000" else 200],
            }
        )
        series_buf = BytesIO()
        series_df.to_csv(series_buf, index=False)
        series_buf.seek(0)

        # Simulate image output
        result_tar = create_tar_from_dict(
            {
                "analysis_result": series_buf,
                "images/plot.png": BytesIO(b"fake-image-data"),
            }
        )

        return SimpleNamespace(returncode=0, stdout=result_tar.getvalue(), stderr=b"")

    monkeypatch.setattr(pipeable.subprocess, "run", fake_run)

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=lambda args: pd.Series({"unused": 0}))

    assert excinfo.value.code == 0
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))

    # Verify output structure: {data_name}/analysis_result
    # read_tar_as_dict creates nested dicts for paths with "/"
    assert "0000" in tar_result
    assert "0001" in tar_result
    assert "analysis_result" in tar_result["0000"]
    assert "analysis_result" in tar_result["0001"]

    # Verify analysis results
    result_0000 = pd.read_csv(
        tar_result["0000"]["analysis_result"],
        dtype={"data_name": str, "sample_name": str},
    )
    assert result_0000.iloc[0]["total_value"] == 100
    assert result_0000.iloc[0]["data_name"] == "0000"
    assert result_0000.iloc[0]["sample_name"] == "SampleA"

    result_0001 = pd.read_csv(
        tar_result["0001"]["analysis_result"],
        dtype={"data_name": str, "sample_name": str},
    )
    assert result_0001.iloc[0]["total_value"] == 200
    assert result_0001.iloc[0]["data_name"] == "0001"
    assert result_0001.iloc[0]["sample_name"] == "SampleB"

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


def test_run_analyze_multi_handles_target_failures(monkeypatch, tmp_path: Path):
    """
    ANALYSISRUN_MODE=analyzemulti で一部のターゲットが失敗した場合、
    全体が失敗として扱われることを確認する。
    """
    monkeypatch.setenv("ANALYSISRUN_MODE", "analyzemulti")
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "targets": json.dumps({"0000": "SampleA", "0001": "SampleB"}),
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_csv_df(
                IMAGE_ANALYSIS_RESULT_CSV
            ),
        }
    )

    # Setup entrypoint mock
    entrypoint = tmp_path / "entry.py"
    entrypoint.write_text("# dummy\n")

    import analysisrun.pipeable as pipeable

    monkeypatch.setattr(pipeable, "get_entrypoint", lambda: entrypoint)

    # Mock subprocess.run to simulate failure for one target
    def fake_run(cmd, *, input, stdout, stderr, env):
        tar_in = read_tar_as_dict(BytesIO(input))
        data_name = tar_in["data_name"]

        if data_name == "0001":
            # Simulate failure
            error_tar = create_tar_from_dict({"error": "Target 0001 failed"})
            return SimpleNamespace(
                returncode=1, stdout=error_tar.getvalue(), stderr=b""
            )

        # Success for 0000
        series_df = pd.DataFrame(
            {"data_name": [data_name], "sample_name": ["SampleA"], "total_value": [100]}
        )
        series_buf = BytesIO()
        series_df.to_csv(series_buf, index=False)
        series_buf.seek(0)

        result_tar = create_tar_from_dict({"analysis_result": series_buf})
        return SimpleNamespace(returncode=0, stdout=result_tar.getvalue(), stderr=b"")

    monkeypatch.setattr(pipeable.subprocess, "run", fake_run)

    ctx = read_context(
        Params,
        ImageResults,
        stdin=BytesIO(tar_buf.getvalue()),
        stdout=stdout_buf,
    )

    with pytest.raises(SystemExit) as excinfo:
        ctx.run_analysis(analyze=lambda args: pd.Series({"unused": 0}))

    # Should exit with error code
    assert excinfo.value.code == 1

    # Verify error output in tar (summary message only)
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert "error" in tar_result
    error_msg = tar_result["error"]
    assert "複数ターゲットの解析中に" in error_msg
    assert "1件のエラーが発生しました" in error_msg
