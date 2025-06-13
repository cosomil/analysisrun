from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import (
    Callable,
    Generator,
    Iterable,
    LiteralString,
    Optional,
    Protocol,
)

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
    解析全体に関わる情報を格納するコンテキストオブジェクト
    """
    fields: Fields
    """
    対象となるレーンのデータを視野ごとに探索するためのスキャナー
    """
    data_for_enhancement: list[Fields]
    """
    各データを別の観点から解析し、補強するためのスキャナーのリスト
    """
    output: Output
    """
    画像を保存するためのOutput実装
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


class NotebookRunner[Context]:
    """
    主にJupyter notebookでの使用を想定したrunner。
    """

    def __init__(
        self,
        analyze: Callable[[AnalyzeArgs[Context]], pd.Series],
        postprocess: Optional[
            Callable[[PostprocessArgs[Context]], pd.DataFrame]
        ] = None,
    ):
        """
        主にJupyter notebookでの使用を想定したrunner。

        Parameters
        ----------
        analyze
            解析関数

            解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。
        postprocess
            解析結果を後処理する関数

            レーンごとの解析結果を結合したDataFrameを受け取り、総合して結果を更新することができる。
            更新したDataFrameは戻り値として返すこと。
        """
        self._analyze = analyze
        self._postprocess = postprocess

    def run(
        self,
        ctx: Context,
        target_data: list[str],
        whole_data: CleansedData,
        data_for_enhancement: list[CleansedData] = [],
        field_numbers: Optional[list[int]] = None,
        output: Optional[Output] = None,
    ) -> pd.DataFrame:
        """
        各レーンごとに数値解析を実行し、解析結果を結合したDataFrameを返す

        Parameters
        ----------
        ctx
            解析全体に関わる情報を格納するコンテキストオブジェクト
        target_data
            対象データのリスト
        whole_data
            クレンジングされた解析対象データ
        data_for_enhancement
            各データを別の観点から解析し、補強するためのデータのリスト
        field_numbers
            スキャン対象となる視野番号のリスト
        output
            画像を保存するためのOutput実装
        """

        results = pd.DataFrame(
            [
                self._analyze(args)
                for args in __analysis_args_generator(
                    ctx,
                    target_data,
                    whole_data,
                    data_for_enhancement,
                    field_numbers,
                    output,
                )
            ]
        )

        return __apply_postprocess(ctx, results, self._postprocess)


class ParallelRunner[Context]:
    """
    マルチプロセスで並列処理するrunner。
    """

    def __init__(
        self,
        analyze: Callable[[AnalyzeArgs[Context]], pd.Series],
        postprocess: Optional[
            Callable[[PostprocessArgs[Context]], pd.DataFrame]
        ] = None,
    ):
        """
        マルチプロセスで並列処理するrunner。

        Parameters
        ----------
        analyze
            解析関数

            解析関数はグローバル変数を参照してはならず、関数のなかで宣言された変数とコンテキストオブジェクトに格納した変数のみを参照すること。
        postprocess
            解析結果を後処理する関数

            レーンごとの解析結果を結合したDataFrameを受け取り、総合して結果を更新することができる。
            更新したDataFrameは戻り値として返すこと。
        """
        self._analyze = analyze
        self._postprocess = postprocess

    def run(
        self,
        ctx: Context,
        target_data: list[str],
        whole_data: CleansedData,
        data_for_enhancement: list[CleansedData] = [],
        field_numbers: Optional[list[int]] = None,
        output: Optional[Output] = None,
    ) -> pd.DataFrame:
        """
        各レーンごとに数値解析を実行し、解析結果を結合したDataFrameを返す

        Parameters
        ----------
        ctx
            解析全体に関わる情報を格納するコンテキストオブジェクト
        target_data
            対象データのリスト
        whole_data
            クレンジングされた解析対象データ
        data_for_enhancement
            各データを別の観点から解析し、補強するためのデータのリスト
        field_numbers
            スキャン対象となる視野番号のリスト
        output
            画像を保存するためのOutput実装
        """

        with ProcessPoolExecutor() as executor:
            results = pd.DataFrame(
                executor.map(
                    self._analyze,
                    __analysis_args_generator(
                        ctx,
                        target_data,
                        whole_data,
                        data_for_enhancement,
                        field_numbers,
                        output,
                    ),
                )
            )

        return __apply_postprocess(ctx, results, self._postprocess)


def __analysis_args_generator[Context](
    ctx: Context,
    target_data: list[str],
    whole_data: CleansedData,
    data_for_enhancement: list[CleansedData] = [],
    field_numbers: Optional[list[int]] = None,
    output: Optional[Output] = None,
) -> Generator[AnalyzeArgs[Context]]:
    """
    各レーンごとの解析に使用する引数を生成するジェネレータ
    """

    field_numbers = field_numbers or [i + 1 for i in range(12)]

    lanes = Lanes(
        whole_data=whole_data,
        target_data=target_data,
        field_numbers=field_numbers,
    )
    lanes_for_enhancement = (
        Lanes(
            target_data=target_data,
            whole_data=data,
            field_numbers=field_numbers,
        )
        for data in data_for_enhancement
    )

    for fields, *lane_for_enhancement in __zip_unpacked(lanes, lanes_for_enhancement):
        yield AnalyzeArgs[Context](
            ctx=ctx,
            fields=fields,
            data_for_enhancement=lane_for_enhancement,
            output=output or DefaultOutput(show=False),
        )


def __apply_postprocess[Context](
    ctx: Context,
    results: pd.DataFrame,
    postprocess: Optional[Callable[[PostprocessArgs[Context]], pd.DataFrame]],
) -> pd.DataFrame:
    """
    解析結果の後処理を適用する

    postprocessがNoneの場合は、結果をそのまま返す。
    """

    if postprocess:
        postprocessed = postprocess(PostprocessArgs(ctx, results))
        if postprocessed is not None:
            return postprocessed
    return results


def __zip_unpacked[T](
    main: Iterable[T], supplemental: Iterable[Iterable[T]]
) -> list[list[T]]:
    return [[x, *others] for x, *others in zip(main, *supplemental)]
