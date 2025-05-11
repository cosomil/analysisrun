from typing import Callable, List, LiteralString, Protocol, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import matplotlib.figure as fig
import matplotlib.pyplot as plt

from . import scanner


class Output(Protocol):
    """
    matplotlib.figure.Figureを保存する。

    Parameters
    ----------
    fig
        保存するFigure。
    name
        保存するファイル名。
    image_type
        画像タイプ。
        実際の画像保存処理のヒントとなります。
    kwargs
        savefigに渡すキーワード引数。
    """

    def __call__(
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs
    ) -> None: ...


class DefaultOutput:
    """
    matplotlib.figure.Figureを保存する。
    shoe=Trueの場合、保存後にNotebookへの表示を実行する。
    """

    def __init__(self, show: bool = False):
        self._show = show

    def __call__(
        self, fig: fig.Figure, name: str, image_type: LiteralString, **kwargs
    ) -> None:
        # 画像を指定の名前で保存し、さらにNotebook上に表示する。
        fig.savefig(name, **kwargs)
        if self._show:
            plt.show(False)
        fig.clear()
        plt.close(fig)


@dataclass
class AnalyzeArgs[Context]:
    ctx: Context
    """
    解析全体に関わる情報を格納するコンテキストオブジェクト。
    dataclassを使用するのが望ましい。
    """
    lane: scanner.LaneDataScanner
    """
    対象となるレーンのデータを探索するためのスキャナー。
    """
    output: Output
    """
    画像を保存するためのOutput実装。
    """


class NotebookRunner:
    """
    主にJupyter notebookでの使用を想定したrunner。
    """

    def __init__(
        self,
        whole_data: pd.DataFrame,
        target_data: List[str],
        viewpoints: Optional[List[int]] = None,
        output: Optional[Output] = None,
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

        self._scanner = scanner.Scanner(
            whole_data=whole_data,
            target_data=target_data,
            viewpoints=viewpoints or [i + 1 for i in range(12)],
        )
        self._output = output or DefaultOutput(show=True)
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

        results = [
            analyze(AnalyzeArgs(ctx, lane, self._output)) for lane in self._scanner
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
        output: Optional[Output] = None,
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

        self._scanner = scanner.Scanner(
            whole_data=whole_data,
            target_data=target_data,
            viewpoints=viewpoints or [i + 1 for i in range(12)],
        )
        self._output = output or DefaultOutput(show=False)
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

        with ProcessPoolExecutor() as executor:
            results = executor.map(
                analyze,
                [AnalyzeArgs(ctx, lane, self._output) for lane in self._scanner],
            )
        return pd.DataFrame(results)
