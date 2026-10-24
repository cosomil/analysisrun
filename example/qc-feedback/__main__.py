"""全体の基準値と QC 判定を各レーンへフィードバックする例。"""

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

import analysisrun as ar

EXAMPLE_DIR = Path(__file__).resolve().parent


class Params(BaseModel):
    minimum_score: float = 0.8


class ImageResults(NamedTuple):
    measurements: pd.DataFrame = ar.image_analysis_result_spec(
        "Measurements", cleansing=ar.entity_filter("Activity Spots")
    )


@dataclass(frozen=True)
class Analysis:
    data: str
    sample_name: str
    measurement: float
    score: float
    analysis_ok: bool
    qc_ok: bool = False
    ok: bool = False


def calculate_reference(
    args: ar.SynthesizeArgs[Params, ImageResults, None, None],
) -> float:
    # QC/通常サンプルを区別せず、全レーンの全測定値を使用する。
    reference = float(
        pd.concat(
            [
                lane.measurements["Value"]
                for lane in args.image_analysis_results.values()
            ]
        ).median()
    )
    if not reference > 0:
        raise ValueError("reference must be positive")
    return reference


def analyze(args: ar.MapArgs[Params, ImageResults, None, float]) -> Analysis:
    measurement = float(args.image_analysis_results.measurements["Value"].mean())
    score = measurement / args.synthesized
    return Analysis(
        args.data_name,
        args.sample_name,
        measurement,
        score,
        score >= args.params.minimum_score,
    )


def evaluate_qc(
    args: ar.SynthesizeArgs[Params, ImageResults, Analysis, float],
) -> bool:
    qc = [value for value in args.values.values() if value.sample_name == "QC"]
    return bool(qc) and all(value.analysis_ok for value in qc)


def apply_qc_evaluation(
    args: ar.MapArgs[Params, ImageResults, Analysis, bool],
) -> Analysis:
    result = replace(
        args.value,
        qc_ok=args.synthesized,
        ok=args.value.analysis_ok and args.synthesized,
    )
    figure, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["relative score"], [result.score], color="green" if result.ok else "red")
    ax.axhline(args.params.minimum_score, color="black", linestyle="--")
    ax.set_title(
        f"{result.data}: {'PASS' if result.ok else 'FAIL'} (QC={result.qc_ok})"
    )
    figure.tight_layout()
    args.output(figure, f"{args.data_name}_qc.png", "qc", format="png")
    return result


def to_dataframe(
    args: ar.SynthesizeArgs[Params, ImageResults, Analysis, bool],
) -> pd.DataFrame:
    return pd.DataFrame(asdict(value) for value in args.values.values())


def main() -> pd.DataFrame:
    input_dir = Path(os.environ.get("QC_INPUT_DIR", EXAMPLE_DIR / "input"))
    output_dir = Path(os.environ.get("QC_OUTPUT_DIR", EXAMPLE_DIR / "output"))
    ctx = ar.read_context(
        Params,
        ImageResults,
        manual_input=ar.ManualInput(
            Params(),
            {"measurements": input_dir / "measurements.csv"},
            sample_names=input_dir / "samples.csv",
        ),
        output_dir=output_dir,
    )
    result = (
        ctx.pipeline()
        .synthesize(calculate_reference)
        .map(analyze)
        .synthesize(evaluate_qc)
        .map(apply_qc_evaluation)
        .finalize(to_dataframe)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "result.csv", index=False)
    return result


if __name__ == "__main__":
    main()
