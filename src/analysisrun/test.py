from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field

from analysisrun.pipeable import (
    Analyzer,
    Cleansing,
    Fields,
    ImageAnalysisResult,
    Input,
    Postprocessor,
    VirtualFile,
    run_analysis,
)


class ImageAnalysisResultsImpl(NamedTuple):
    # activity_spots: VirtualFile = Field(description="画像解析結果csv")
    activity_spots: VirtualFile = ImageAnalysisResult(
        "画像解析結果csv", Cleansing(entity="Activity Spots")
    )
    # surrounding_spots: VirtualFile = Field(description="補強csv")


class ImageAnalysisResults(NamedTuple):
    activity_spots: Fields
    # surrounding_spots: Fields


class ParametersImpl(BaseModel):
    threshold: int = Field(
        description="解析に使用する閾値",
    )


class AAnalyzer(Analyzer[ParametersImpl, ImageAnalysisResults]):
    def analyze(self) -> pd.Series:
        a = self.image_analysis_results.activity_spots

        # 視野ごとの、閾値を超える数値の中央値を計算する
        medians = [f[f["Value"] > self.params.threshold]["Value"].median() for f in a]

        fig = plt.figure()
        ax = fig.subplots()
        ax.bar(range(len(medians)), medians, width=1, edgecolor="white", linewidth=0.7)
        self.output(fig, f"{self.sample_name}_median.png", "median")

        return pd.Series({f"field_{i}": v for i, v in enumerate(medians)})


class APostprocessor(Postprocessor[ParametersImpl]):
    def postprocess(self) -> pd.DataFrame:
        df = self.analysis_results
        df["threshold"] = self.params.threshold
        return df


if __name__ == "__main__":
    from pydantic import create_model

    M = create_model("Tmp", a=str, b=int)
    m = M(a="example", b=123)
    print(m)
    # result = run_analysis(
    #     parameters_type=ParametersImpl,
    #     image_analysis_result_input_type=ImageAnalysisResultsImpl,
    #     image_analysis_results_type=ImageAnalysisResults,
    #     analyzer=AAnalyzer,
    #     postprocessor=APostprocessor,
    #     manual_input=Input(
    #         image_analysis_results=ImageAnalysisResultsImpl(
    #             activity_spots=VirtualFile(
    #                 Path("./tests/testdata/image_analysis_result.csv")
    #             ),
    #         ),
    #         sample_names=VirtualFile(Path("./tests/testdata/samples.csv")),
    #         parameters=ParametersImpl(threshold=3000),
    #     ),
    #     output_dir="./src/analysisrun/testing/",
    # )
    # print(result)
