from typing import Callable, List, Optional, LiteralString, Protocol
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import matplotlib.figure as fig
import matplotlib.pyplot as plt

from . import scanner


class Output(Protocol):
    def image(
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs
    ) -> None:
        """
        matplotlib.figure.Figureを保存するための関数

        Parameters
        ----------
        fig
            保存するFigure
        name
            保存するファイル名
        image_type
            画像タイプ
            実際の画像保存処理のヒントとなります
        kwargs
            savefigに渡すキーワード引数
        """
        ...


class NotebookOutput:
    def image(
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs
    ) -> None:
        # 画像を指定の名前で保存し、さらにNotebook上に表示する。
        fig.savefig(name, **kwargs)
        plt.show(False)
        fig.clear()
        plt.close(fig)


@dataclass
class AnalyzeArgs[Context]:
    ctx: Context
    lane: scanner.LaneDataScanner
    output: Output


class NotebookRunner:
    """
    主にJupyter notebookでの使用を想定したrunner。
    """

    def __init__(
        self,
        whole_data: pd.DataFrame,
        target_data: List[str],
        viewpoints: Optional[List[int]] = None,
    ):
        """
        主にJupyter notebookでの使用を想定したrunner。

        Parameters
        ----------
        whole_data
            全データ
        target_data
            対象データのリスト
        viewpoints
            スキャン対象となる視野番号のリスト
        """
        viewpoints = (
            viewpoints if viewpoints is not None else [i + 1 for i in range(12)]
        )
        self.scanner = scanner.Scanner(
            whole_data=whole_data,
            target_data=target_data,
            viewpoints=viewpoints,
        )
        return

    def run[Context](
        self,
        ctx: Context,
        analyze: Callable[[AnalyzeArgs[Context]], pd.Series],
    ) -> pd.DataFrame:
        """
        各レーンごとに画像解析を実行する。
        レーンごとの解析結果を結合したDataFrameを返す。

        Parameters
        ----------
        ctx
            解析全体に関わる情報を格納するコンテキストオブジェクト。
            dataclassを使用するのが望ましい。
        analyze
            解析関数。
            解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。
        """

        output = NotebookOutput()
        results = [
            analyze(AnalyzeArgs(ctx, lane, output)) for lane in self.scanner.each_lane()
        ]
        return pd.DataFrame(results)


class ParallelRunner:
    """
    マルチプロセスで並列処理するrunner。
    """

    def __init__(
        self,
        whole_data: pd.DataFrame,
        target_data: List[str],
        viewpoints: Optional[List[int]] = None,
    ):
        """
        マルチプロセスで並列処理するrunner。

        Parameters
        ----------
        whole_data
            全データ
        target_data
            対象データのリスト
        viewpoints
            スキャン対象となる視野番号のリスト
        """
        viewpoints = (
            viewpoints if viewpoints is not None else [i + 1 for i in range(12)]
        )
        self.scanner = scanner.Scanner(
            whole_data=whole_data,
            target_data=target_data,
            viewpoints=viewpoints,
        )
        return

    def run[Context](
        self,
        ctx: Context,
        analyze: Callable[[AnalyzeArgs[Context]], pd.Series],
    ) -> pd.DataFrame:
        """
        各レーンごとに画像解析を実行する。
        レーンごとの解析結果を結合したDataFrameを返す。

        Parameters
        ----------
        ctx
            解析全体に関わる情報を格納するコンテキストオブジェクト。
            dataclassを使用するのが望ましい。
        analyze
            解析関数。
            解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。
        """

        output = NotebookOutput()
        with ProcessPoolExecutor() as executor:
            results = executor.map(
                analyze,
                [AnalyzeArgs(ctx, lane, output) for lane in self.scanner.each_lane()],
            )
        return pd.DataFrame(results)
