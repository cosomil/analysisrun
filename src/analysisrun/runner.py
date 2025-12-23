from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import (
    Callable,
    Generator,
    Generic,
    Iterable,
    Optional,
    TypeVar,
)

import pandas as pd

from analysisrun.output import DefaultOutput, Output
from analysisrun.scanner import CleansedData, Fields, Lanes

Context = TypeVar("Context")


@dataclass
class AnalyzeArgs(Generic[Context]):
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
class PostprocessArgs(Generic[Context]):
    ctx: Context
    """
    解析全体に関わる情報を格納するコンテキストオブジェクト。
    """
    analysis_results: pd.DataFrame
    """
    解析結果を格納したDataFrame。
    """


class NotebookRunner(Generic[Context]):
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
                for args in _analysis_args_generator(
                    ctx,
                    target_data,
                    whole_data,
                    data_for_enhancement,
                    field_numbers,
                    output,
                )
            ]
        )

        return _apply_postprocess(ctx, results, self._postprocess)


class ParallelRunner(Generic[Context]):
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
                    _analysis_args_generator(
                        ctx,
                        target_data,
                        whole_data,
                        data_for_enhancement,
                        field_numbers,
                        output,
                    ),
                )
            )

        return _apply_postprocess(ctx, results, self._postprocess)


def _analysis_args_generator(
    ctx: Context,
    target_data: list[str],
    whole_data: CleansedData,
    data_for_enhancement: list[CleansedData] = [],
    field_numbers: Optional[list[int]] = None,
    output: Optional[Output] = None,
) -> Generator[AnalyzeArgs[Context], None, None]:
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

    for fields, *lane_for_enhancement in _zip_unpacked(lanes, lanes_for_enhancement):
        yield AnalyzeArgs[Context](
            ctx=ctx,
            fields=fields,
            data_for_enhancement=lane_for_enhancement,
            output=output or DefaultOutput(show=False),
        )


def _apply_postprocess(
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


def _zip_unpacked[T](
    main: Iterable[T], supplemental: Iterable[Iterable[T]]
) -> Generator[list[T], None, None]:
    return ([x, *others] for x, *others in zip(main, *supplemental))
