from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, LiteralString, Optional, Protocol, Iterable, Sequence

import matplotlib.figure as fig
import matplotlib.pyplot as plt
import pandas as pd

from .scanner import CleansedData, Fields, Lanes


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
    show=Trueの場合、保存後にNotebookへの表示を実行する。
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
    """
    fields: Fields
    """
    対象となるレーンのデータを視野ごとに探索するためのスキャナー。
    """
    fields_for_enhancement: list[Fields]
    """
    各データを別の観点から解析し、補強するためのスキャナーのリスト。
    """
    output: Output
    """
    画像を保存するためのOutput実装。
    """


@dataclass
class PostprocessArgs[Context]:
    ctx: Context
    """
    解析全体に関わる情報を格納するコンテキストオブジェクト。
    """
    analysis_results: pd.DataFrame
    """
    解析結果を格納したDataFrame。
    """


class NotebookRunner:
    """
    主にJupyter notebookでの使用を想定したrunner。
    """

    def __init__(
        self,
        target_data: list[str],
        whole_data: CleansedData,
        data_for_enhancement: list[CleansedData] = [],
        field_numbers: Optional[list[int]] = None,
        output: Optional[Output] = None,
    ):
        """
        主にJupyter notebookでの使用を想定したrunner。

        Parameters
        ----------
        target_data
            対象データのリスト
        whole_data
            クレンジングされた解析対象データ
        data_for_enhancement
            各データを別の観点から解析し、補強するためのデータのリスト。
        field_numbers
            スキャン対象となる視野番号のリスト
        """

        field_numbers = field_numbers or [i + 1 for i in range(12)]

        self._lanes = Lanes(
            whole_data=whole_data,
            target_data=target_data,
            field_numbers=field_numbers,
        )
        self._lanes_for_enhancement = [
            Lanes(
                target_data=target_data,
                whole_data=data,
                field_numbers=field_numbers,
            )
            for data in data_for_enhancement
        ]
        self._output = output or DefaultOutput(show=True)
        return

    def run[Context](
        self,
        ctx: Context,
        analyze: Callable[[AnalyzeArgs[Context]], pd.Series],
        postprocess: Optional[
            Callable[[PostprocessArgs[Context]], pd.DataFrame]
        ] = None,
    ) -> pd.DataFrame:
        """
        各レーンごとに数値解析を実行する。
        レーンごとの解析結果を結合したDataFrameを返す。

        Parameters
        ----------
        ctx
            解析全体に関わる情報を格納するコンテキストオブジェクト。
        analyze
            解析関数。
            解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。
        postprocess
            解析結果を後処理する関数。
            レーンごとの解析結果を結合したDataFrameを受け取り、総合して結果を更新することができる。
            更新したDataFrameは戻り値として返すこと。
        """

        results = pd.DataFrame(
            [
                analyze(
                    AnalyzeArgs[Context](
                        ctx=ctx,
                        fields=fields,
                        fields_for_enhancement=lane_for_enhancement,
                        output=self._output,
                    )
                )
                for fields, *lane_for_enhancement in __zip_unpacked(
                    self._lanes, self._lanes_for_enhancement
                )
            ]
        )
        if postprocess:
            postprocessed = postprocess(PostprocessArgs(ctx, results))
            if postprocessed is not None:
                return postprocessed
        return results


class ParallelRunner:
    """
    マルチプロセスで並列処理するrunner。
    """

    def __init__(
        self,
        target_data: list[str],
        whole_data: CleansedData,
        data_for_enhancement: list[CleansedData] = [],
        field_numbers: Optional[list[int]] = None,
        output: Optional[Output] = None,
    ):
        """
        マルチプロセスで並列処理するrunner。

        Parameters
        ----------
        target_data
            対象データのリスト
        whole_data
            クレンジングされた解析対象データ
        data_for_enhancement
            各データを別の観点から解析し、補強するためのデータのリスト。
        field_numbers
            スキャン対象となる視野番号のリスト
        """

        field_numbers = field_numbers or [i + 1 for i in range(12)]

        self._lanes = Lanes(
            target_data=target_data,
            whole_data=whole_data,
            field_numbers=field_numbers,
        )
        self._lanes_for_enhancement = [
            Lanes(
                target_data=target_data,
                whole_data=data,
                field_numbers=field_numbers,
            )
            for data in data_for_enhancement
        ]

        self._output = output or DefaultOutput(show=False)
        return

    def run[Context](
        self,
        ctx: Context,
        analyze: Callable[[AnalyzeArgs[Context]], pd.Series],
        postprocess: Optional[
            Callable[[PostprocessArgs[Context]], pd.DataFrame]
        ] = None,
    ) -> pd.DataFrame:
        """
        各レーンごとに数値解析を実行する。
        レーンごとの解析結果を結合したDataFrameを返す。

        Parameters
        ----------
        ctx
            解析全体に関わる情報を格納するコンテキストオブジェクト。
        analyze
            解析関数。
            解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。
        postprocess
            解析結果を後処理する関数。
            レーンごとの解析結果を結合したDataFrameを受け取り、総合して結果を更新することができる。
            更新したDataFrameは戻り値として返すこと。
        """

        with ProcessPoolExecutor() as executor:
            results = pd.DataFrame(
                executor.map(
                    analyze,
                    [
                        AnalyzeArgs[Context](
                            ctx=ctx,
                            fields=fields,
                            fields_for_enhancement=lane_for_enhancement,
                            output=self._output,
                        )
                        for fields, *lane_for_enhancement in __zip_unpacked(
                            self._lanes, self._lanes_for_enhancement
                        )
                    ],
                )
            )
        if postprocess:
            postprocessed = postprocess(PostprocessArgs(ctx, results))
            if postprocessed is not None:
                return postprocessed
        return results


def __zip_unpacked[T](
    main: Iterable[T], supplemental: Sequence[Iterable[T]]
) -> list[list[T]]:
    return [[x, *others] for x, *others in zip(main, *supplemental)]
