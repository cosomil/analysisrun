from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import BaseModel

from analysisrun.pipeable import (
    ImageAnalysisResultSpec,
    ManualInput,
    create_image_analysis_results_input_model,
    entity_filter,
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
    activity_spots: Fields = ImageAnalysisResultSpec(
        description="Activity spots",
        cleansing=(entity_filter("Activity Spots"),),
    )


def _load_pickled_df(path: Path) -> BytesIO:
    df = pd.read_csv(path)
    buf = BytesIO()
    df.to_pickle(buf)
    buf.seek(0)
    return buf


def test_create_image_analysis_results_input_model_requires_spec():
    class InvalidImageResults(NamedTuple):
        activity_spots: Fields

    with pytest.raises(ValueError):
        create_image_analysis_results_input_model(InvalidImageResults)


def test_run_analysis_sequential_with_manual_input(monkeypatch):
    monkeypatch.setenv("PSEUDO_NBENV", "1")
    monkeypatch.delenv("ANALYSISRUN_METHOD", raising=False)

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
    monkeypatch.setenv("ANALYSISRUN_METHOD", "analyze")
    stdout_buf = BytesIO()

    tar_buf = create_tar_from_dict(
        {
            "data_name": "0000",
            "sample_name": "SampleA",
            "params": Params(threshold=3).model_dump_json(),
            "image_analysis_results/activity_spots": _load_pickled_df(
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
    series = pd.read_pickle(series_buf)
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


def test_run_postprocess_only_outputs_tar(monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_METHOD", "postprocess")
    stdout_buf = BytesIO()

    analysis_results = pd.DataFrame(
        [
            {"data_name": "0000", "sample_name": "SampleA", "total_value": 4},
            {"data_name": "0001", "sample_name": "SampleB", "total_value": 6},
        ]
    )
    buf = BytesIO()
    analysis_results.to_pickle(buf)
    buf.seek(0)

    tar_buf = create_tar_from_dict(
        {
            "analysis_results": buf,
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
    assert first["scaled"] == 20
