from typing import Callable, ParamSpec, Concatenate, List, Optional

import pandas as pd
import matplotlib.figure as fig

from . import scanner

SaveFigKwargs = ParamSpec("SaveFigKwargs")
OutputImage = Callable[Concatenate[fig.Figure, str, SaveFigKwargs], None]
"""
matplotlib.figure.Figureを保存するための関数

Parameters
----------
fig
    保存するFigure
name
    保存するファイル名
kwargs
    savefigに渡すキーワード引数
"""


class NotebookRunner:
    """
    主にJupyter notebookでの使用を想定したrunner。
    """

    def __init__(
        self,
        whole_data: pd.DataFrame,
        target_data: List[str],
        viewpoints: Optional[List[int]] = [],
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
        analyze: Callable[[Context, scanner.LaneDataScanner, OutputImage], pd.Series],
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
            analyze(ctx, lane, self._output_image) for lane in self.scanner.each_lane()
        ]
        return pd.DataFrame(results)

    @staticmethod
    def _output_image(fig: fig.Figure, name: str, **kwargs) -> None:
        # 画像を指定の名前で保存し、さらにNotebook上に表示する。
        fig.savefig(name, **kwargs)
        fig.show()
        fig.clf()
        fig.clear()
        return
