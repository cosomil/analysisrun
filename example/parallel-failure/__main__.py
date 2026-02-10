from pathlib import Path
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel

import analysisrun as ar
from analysisrun.scanner import Fields


class FailureParams(BaseModel):
    trigger: str = "always"


class FailureImageAnalysisResults(NamedTuple):
    activity_spots: Fields = ar.image_analysis_result_spec(
        description="Activity spots",
        cleansing=ar.entity_filter("Activity Spots"),
    )


def analyze(
    args: ar.AnalyzeArgs[FailureParams, FailureImageAnalysisResults],
) -> pd.Series:
    lane_name = args.image_analysis_results.activity_spots.data_name
    raise RuntimeError(f"intentional failure during lane analysis: {lane_name}")


project_root = Path(__file__).resolve().parents[2]
testdata_dir = project_root / "tests" / "testdata"
output_dir = Path(__file__).resolve().parent / "example-output"  # 何も出力されない

ctx = ar.read_context(
    FailureParams,
    FailureImageAnalysisResults,
    manual_input=ar.ManualInput(
        FailureParams(),
        {"activity_spots": testdata_dir / "image_analysis_result.csv"},
        sample_names=testdata_dir / "samples.csv",
    ),
    output_dir=output_dir,
)
ctx.run_analysis(analyze)
