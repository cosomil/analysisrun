from dataclasses import dataclass
from pathlib import Path
import sys
from typing import IO, Callable, Literal, Optional, Type

import pandas as pd
from pydantic import BaseModel

from analysisrun.__typing import NamedTupleLike, VirtualFileLike
from analysisrun.runner import Output
from analysisrun.scanner import Fields


@dataclass
class ManualInput[Params: BaseModel | None]:
    params: Params
    image_analysis_results: dict[str, VirtualFileLike]
    sample_names: VirtualFileLike


@dataclass
class AnalyzeArgs[
    Params: BaseModel | None,
    ImageAnalysisResults: NamedTupleLike[Fields],
]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    image_analysis_results: ImageAnalysisResults
    """
    解析対象となる画像解析結果の定義
    """
    output: Output
    """
    画像を保存するためのOutput実装
    """


@dataclass
class PostprocessArgs[Params: BaseModel | None]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    analysis_results: pd.DataFrame
    """
    各レーンの解析結果を格納したDataFrame
    """


@dataclass
class AnalysisContext[
    Params: BaseModel | None,
    ImageAnalysisResults: NamedTupleLike[Fields],
]:
    """
    数値解析のコンテキスト。
    解析の入力、実行モードなどを保持し、モードに応じて解析の実行を行う。
    """

    mode: Literal[
        "sequential", "parallel-entrypoint", "analysis-only", "postprocess-only"
    ]
    params: Params
    image_analysis_results: ImageAnalysisResults
    output: Output

    def run_analysis(
        self,
        analyze: Callable[[AnalyzeArgs[Params, ImageAnalysisResults]], pd.Series],
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        コンテキストに基づいて数値解析を実行する。

        TODO: 各実行モードの動作について、説明を追加したい。

        :param analyze: レーンごとに実行される解析処理の実装。
        :param postprocess: 後処理の実装（任意）。全レーンの解析結果をもとにさらに判定を加えたい場合に使用する。
        :return: 解析結果を格納したDataFrame。
        :raises SystemExit: 成功/失敗を問わず、分散環境で実行される場合にスローされる。
        :raises RuntimeError: ローカル実行でエラー発生時にスローされる。
        """
        # TODO: implement
        pass


def read_context[
    Params: BaseModel | None,
    ImageAnalysisResults: NamedTupleLike[Fields],
](
    params: Type[Params],
    image_analysis_results: Type[ImageAnalysisResults],
    manual_input: Optional[ManualInput[Params]] = None,
    stdin: IO[bytes] = sys.stdin.buffer,
    stdout: IO[bytes] = sys.stdout.buffer,
    output_dir: Optional[str | Path] = None,
) -> AnalysisContext[Params, ImageAnalysisResults]:
    """
    引数、環境変数、標準入力を読み取り、

    :param params: 解析全体に関わるパラメータを定義するクラス
    :param image_analysis_results: 解析対象となる画像解析結果を定義するクラス
    :param manual_input: 入力データ。Jupyter notebook環境で実行する際には必須
    :param stdin: 標準入力
    :param stdout: 標準出力
    :param output_dir: ローカル実行する際の画像データの出力先ディレクトリ。
        画像解析結果がファイルパスで与えられる場合にはそのファイルが存在するディレクトリ、それ以外はカレントディレクトリがデフォルト値となる。
    :return: AnalysisContextオブジェクト
    """
    # TODO: implement
    pass
