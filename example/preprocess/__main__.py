from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel

import analysisrun as ar

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTDATA_DIR = REPO_ROOT / "tests" / "testdata"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class Params(BaseModel):
    threshold: int = 3


class ImageAnalysisResults(NamedTuple):
    activity_spots: pd.DataFrame = ar.image_analysis_result_spec(
        description="Activity spots",
        cleansing=ar.entity_filter("Activity Spots"),
    )


class PreprocessedImageAnalysisResults(NamedTuple):
    activity_spots: pd.DataFrame


@dataclass
class Extra:
    target_count: int


def preprocess(
    args: ar.PreprocessArgs[Params, ImageAnalysisResults],
) -> ar.ProcessedInputs[PreprocessedImageAnalysisResults, Extra]:
    df = args.image_analysis_results.activity_spots.copy()
    df["scaled_value"] = df["Value"] * args.params.threshold
    max_value = float(df["scaled_value"].max())
    df["scaled_ratio"] = df["scaled_value"] / max_value if max_value else 0.0

    return ar.ProcessedInputs(
        image_analysis_results=PreprocessedImageAnalysisResults(activity_spots=df),
        extra=Extra(target_count=int(len(df))),
    )


def analyze(
    args: ar.AnalyzeArgsWithPreprocess[
        Params,
        PreprocessedImageAnalysisResults,
        Extra,
    ],
) -> pd.Series:
    df = args.image_analysis_results.activity_spots
    fields = ar.scan_fields(df, args.data_name)

    return pd.Series(
        {
            "sample_name": args.sample_name,
            "spot_count": len(df),
            "scaled_value_sum": float(df["scaled_value"].sum()),
            "scaled_ratio_mean": float(df["scaled_ratio"].mean()),
            "target_count": args.extra.target_count,
            "image_analysis_method": fields.image_analysis_method,
        }
    )


def postprocess(
    args: ar.PostprocessArgsWithPreprocess[Params, Extra],
) -> pd.DataFrame:
    result = args.analysis_results.copy()
    result["threshold"] = args.params.threshold
    result["is_valid"] = result["scaled_value_sum"] >= result["threshold"]
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = ar.read_context(
        Params,
        ImageAnalysisResults,
        manual_input=ar.ManualInput(
            Params(threshold=3),
            {
                "activity_spots": TESTDATA_DIR / "image_analysis_result.csv",
            },
            sample_names=TESTDATA_DIR / "samples.csv",
        ),
        output_dir=OUTPUT_DIR,
    )

    result = ctx.run_analysis_with_preprocess(
        preprocessed_image_analysis_results=PreprocessedImageAnalysisResults,
        preprocess=preprocess,
        analyze=analyze,
        postprocess=postprocess,
    )
    result.to_csv(OUTPUT_DIR / "result.csv", index=False)
    print(result)


if __name__ == "__main__":
    main()
