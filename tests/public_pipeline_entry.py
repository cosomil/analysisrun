"""Real read_context/builder entrypoint for normal and TAR integration tests."""

import os
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from analysisrun import (
    ManualInput,
    entity_filter,
    image_analysis_result_spec,
    read_context,
)

IMAGE_ANALYSIS_RESULT_CSV = (
    Path(__file__).parent / "testdata" / "image_analysis_result.csv"
)
SAMPLES_CSV = Path(__file__).parent / "testdata" / "samples.csv"


class Params(BaseModel):
    threshold: int = 1


class ImageResults(NamedTuple):
    activity_spots: pd.DataFrame = image_analysis_result_spec(
        description="Activity spots",
        cleansing=entity_filter("Activity Spots"),
    )


class Prepared(NamedTuple):
    cells: pd.DataFrame


def prepare(args):
    return Prepared(args.image_analysis_results.activity_spots.assign(prepared=True))


def extract(args):
    assert args.value is args.synthesized is None
    if isinstance(args.image_analysis_results, Prepared):
        assert args.image_analysis_results.cells.prepared.all()
    else:
        assert isinstance(args.image_analysis_results, ImageResults)
    figure, ax = plt.subplots()
    ax.plot([0, 1])
    args.output(figure, f"{args.data_name}.png", "qc")
    plt.close(figure)
    return args.params.threshold


def qc(args):
    assert list(args.values) == ["0000", "0001"]
    return sum(args.values.values())


def correct(args):
    return args.value + args.synthesized


def finalize(args):
    return pd.DataFrame(
        {"data": list(args.values), "score": list(args.values.values())}
    )


def main():
    # In worker mode, these paths may be absent: only the parent loads input.
    manual = None
    if os.environ.get("ANALYSISRUN_INTERACTIVITY") == "terminal":
        manual = ManualInput(
            params=Params(threshold=2),
            sample_names=SAMPLES_CSV,
            image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        )
    ctx = read_context(
        Params, ImageResults, manual_input=manual, output_dir=os.environ["TEST_OUTPUT"]
    )
    builder = ctx.pipeline()
    if os.environ.get("TEST_PREPROCESS") == "1":
        builder = builder.preprocess(prepare)
    result = builder.map(extract).synthesize(qc).map(correct).finalize(finalize)
    result.to_json(Path(os.environ["TEST_OUTPUT"]) / "result.json", orient="records")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.public_pipeline_entry import main as entry

    entry()
