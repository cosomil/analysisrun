from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple

import pandas as pd

from analysisrun.cleansing import filter_by_entity
from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile
from analysisrun.pipeable import (
    Analyzer,
    Cleansing,
    ImageAnalysisResult,
    Input,
    Postprocessor,
    run_analysis,
)
from analysisrun.runner import AnalyzeArgs, ParallelRunner, PostprocessArgs
from analysisrun.scanner import Fields
from tests.testimpl import BenchParameters, analyze, postprocess

params = BenchParameters(
    points_per_field=60_000,  # 各視野の系列長（描画用にはサンプリング）
    burn_in=2_000,  # 収束捨て
    repeats=2,  # 追加の繰り返し計算ループ回数（計算量スケール用）
    poly_degree=512,  # 多項モーメントの最高次数
    tau=7,  # 遅延座標
    plot_sample=500,  # 散布図の描画サンプル数/視野（多すぎると画像が重い）
)


def assert_result_files(got_dir: Path):
    want_dir = Path(__file__).parent / "testresult"

    # got_dirに含まれるファイルとwant_dirに含まれるファイルのリストやデータが一致することを検証する。
    got_files = sorted([f.name for f in got_dir.iterdir() if f.is_file()])
    want_files = sorted([f.name for f in want_dir.iterdir() if f.is_file()])

    assert got_files == want_files, f"Files in {got_dir} do not match expected files."

    for filename in got_files:
        got_file = got_dir / filename
        want_file = want_dir / filename

        with open(got_file, "rb") as gf, open(want_file, "rb") as wf:
            got_data = gf.read()
            want_data = wf.read()
            assert got_data == want_data, (
                f"Contents of file {filename} do not match expected contents."
            )


def test_ParallelRunner():
    targets = read_dict("./tests/testdata/samples.csv", "data", "sample")
    df = pd.read_csv("./tests/testdata/image_analysis_result.csv")

    with TemporaryDirectory() as tempdir:
        os.chdir(tempdir)

        r = ParallelRunner(analyze, postprocess)
        result = r.run(
            ctx=params,
            target_data=list(targets),
            whole_data=filter_by_entity(df, "Activity Spots"),
        )
        result.to_csv("result.csv")

        assert_result_files(Path(tempdir))


class BenchImageAnalysisResultsInput(NamedTuple):
    activity_spots: VirtualFile = ImageAnalysisResult(
        "画像解析結果csv", Cleansing(entity="Activity Spots")
    )


def a() -> Any:
    pass


class BenchImageAnalysisResults(NamedTuple):
    activity_spots: Fields = a()


ss = BenchImageAnalysisResults()


class BenchAnalyzer(Analyzer[BenchParameters, BenchImageAnalysisResults]):
    def analyze(self):
        return analyze(
            AnalyzeArgs[BenchParameters](
                ctx=self.params,
                fields=self.image_analysis_results.activity_spots,
                data_for_enhancement=[],
                output=self.output,
            )
        )


class BenchPostprocessor(Postprocessor[BenchParameters]):
    def postprocess(self):
        return postprocess(
            PostprocessArgs[BenchParameters](
                ctx=self.params,
                analysis_results=self.analysis_results,
            )
        )


def test_run_analysis():
    p = Path(__file__).parent

    with TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        dir = Path(tempdir)

        result = run_analysis(
            parameters_type=BenchParameters,
            image_analysis_result_input_type=BenchImageAnalysisResultsInput,
            image_analysis_results_type=BenchImageAnalysisResults,
            analyzer=BenchAnalyzer,
            postprocessor=BenchPostprocessor,
            manual_input=Input(
                image_analysis_results=BenchImageAnalysisResultsInput(
                    activity_spots=VirtualFile(p / "testdata/image_analysis_result.csv")
                ),
                sample_names=VirtualFile(p / "testdata/samples.csv"),
                parameters=params,
            ),
            output_dir=dir,
        )
        result.to_csv(dir / "result.csv")

        assert_result_files(dir)
