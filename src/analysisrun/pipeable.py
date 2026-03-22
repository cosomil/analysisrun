import json
import os
import pickle
import re
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from types import MappingProxyType
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Literal,
    LiteralString,
    Mapping,
    Optional,
    Protocol,
    Type,
)

import matplotlib.figure as fig
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, create_model

from analysisrun.env import get_entrypoint, get_interactivity
from analysisrun.typing import NamedTupleLike, VirtualFileLike
from analysisrun.cleansing import CleansedData, filter_by_entity
from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile, scan_model_input
from analysisrun.pipeable_io import (
    AnalyzeSeqInputModel,
    ExitCodes,
    exit_with_error,
    exit_with_error_streaming,
    redirect_stdout_to_stderr,
)
from analysisrun.scanner import Lanes, scan
from analysisrun.tar import FileIO, create_tar_from_dict, read_tar_as_dict


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


class NoParams(BaseModel):
    pass


@dataclass
class ManualInput[Params: BaseModel]:
    """
    ローカル実行時に使用される数値解析の入力データ
    """

    params: Params
    """
    解析全体に関わるパラメータ
    """
    image_analysis_results: dict[str, VirtualFileLike]
    """
    解析対象となる画像解析結果CSVデータセット
    """
    sample_names: VirtualFileLike
    """
    サンプル名CSVファイル（サンプル名とレーン番号の対応表）
    """

    model_config = {
        "arbitrary_types_allowed": True,
    }


CleansingFunc = Callable[[pd.DataFrame | CleansedData], CleansedData]
_ImageResultsSerialization = Literal["csv", "pickle"]


@dataclass
class _ImageAnalysisResultSpec:
    """
    画像解析データの説明とクレンジング処理を保持する。
    """

    description: str
    cleansing: tuple[CleansingFunc, ...] = field(default_factory=tuple)


def image_analysis_result_spec(
    description: str, cleansing: CleansingFunc | tuple[CleansingFunc, ...]
) -> Any:
    """
    画像解析結果フィールドの仕様を定義する。

    """
    return _ImageAnalysisResultSpec(
        description=description,
        cleansing=cleansing if isinstance(cleansing, tuple) else (cleansing,),
    )


def entity_filter(
    entity: str | Iterable[str],
) -> Callable[[pd.DataFrame | CleansedData], CleansedData]:
    """
    既存のfilter_by_entityと同等のフィルタリング処理を返す。
    """

    return lambda data: filter_by_entity(data, entity=entity)


def _get_image_analysis_specs[
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    image_analysis_results: Type[ImageAnalysisResults],
) -> dict[str, _ImageAnalysisResultSpec]:
    """
    ImageAnalysisResultsのデフォルト値に定義されたImageAnalysisResultSpecを取得する。
    """

    field_defaults = image_analysis_results._field_defaults
    specs: dict[str, _ImageAnalysisResultSpec] = {}

    for name in image_analysis_results._fields:  # type: ignore[attr-defined]
        spec = field_defaults.get(name)
        if not isinstance(spec, _ImageAnalysisResultSpec):
            raise ValueError(
                f"{image_analysis_results.__name__}.{name} must have ImageAnalysisResultSpec as default"
            )
        specs[name] = spec

    return specs


def _build_streaming_input_schema[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    params: Type[Params],
    image_analysis_results: Type[ImageAnalysisResults],
) -> dict[str, Any]:
    specs = _get_image_analysis_specs(image_analysis_results)
    tar_entries: list[dict[str, Any]] = [
        {
            "path": "params",
            "required": True,
            "content_type": "application/json",
            "description": "解析全体に関わるパラメータ",
            "pax_headers": {},
            "json_schema": params.model_json_schema(),
        },
        {
            "path": "sample_names",
            "required": True,
            "content_type": "text/csv",
            "description": "サンプル名CSVファイル（サンプル名とレーン番号の対応表）",
            "pax_headers": {"is_file": "true"},
        },
    ]
    for name, spec in specs.items():
        tar_entries.append(
            {
                "path": f"image_analysis_results/{name}",
                "required": True,
                "content_type": "text/csv",
                "description": spec.description,
                "pax_headers": {"is_file": "true"},
            }
        )

    return {
        "schema_version": "1",
        "transport": {
            "type": "tar",
            "compression": ["tar", "tar.gz"],
            "path_separator": "/",
        },
        "tar_entries": tar_entries,
    }


def create_image_analysis_results_input_model[
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](image_analysis_results: Type[ImageAnalysisResults]) -> Type[BaseModel]:
    """
    ImageAnalysisResults定義から動的にVirtualFile入力モデルを生成する。

    各フィールドのデフォルト値としてImageAnalysisResultSpecが設定されていることを前提とし、
    設定されていない場合はエラーを送出する。
    """

    specs = _get_image_analysis_specs(image_analysis_results)

    model_fields: dict[str, tuple[type, Any]] = {}
    for name, spec in specs.items():
        model_fields[name] = (VirtualFile, Field(description=spec.description))

    return create_model(
        "ImageAnalysisResultsInput",
        __config__={"arbitrary_types_allowed": True},
        **model_fields,  # type: ignore[arg-type]
    )


class InputModel[
    Params: BaseModel,
    ImageAnalysisResultsInput: BaseModel,
](BaseModel):
    """
    ローカル実行時に使用される数値解析の入力データモデル

    manual_inputのバリデーションやインタラクティブな入力のバリデーションに使用する
    """

    image_analysis_results: ImageAnalysisResultsInput = Field(
        description="画像解析結果CSVデータセット"
    )
    sample_names: VirtualFile = Field(
        description="サンプル名CSVファイル（サンプル名とレーン番号の対応表）"
    )
    params: Params = Field(description="解析全体に関わるパラメータ")

    model_config = {
        "arbitrary_types_allowed": True,
    }


@dataclass
class AnalyzeArgs[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    data_name: str
    """
    画像解析結果CSVのデータ名
    """
    sample_name: str
    """
    サンプル名
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
class PreprocessArgs[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    image_analysis_results: ImageAnalysisResults
    """
    cleansing済みの画像解析結果（DataFrame）
    """
    targets: Mapping[str, str]
    """
    解析対象データ。keyはdata_name、valueはsample_name
    """


@dataclass
class ProcessedInputs[
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
    Extra,
]:
    image_analysis_results: ImageAnalysisResults
    """
    preprocess後にanalyzeへ渡されるcleansing済み画像解析結果（DataFrame）
    """
    extra: Extra
    """
    preprocessで生成された追加データ
    """


@dataclass
class AnalyzeArgsWithPreprocess[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
    Extra,
]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    data_name: str
    """
    画像解析結果CSVのデータ名
    """
    sample_name: str
    """
    サンプル名
    """
    image_analysis_results: ImageAnalysisResults
    """
    解析対象となる画像解析結果の定義
    """
    output: Output
    """
    画像を保存するためのOutput実装
    """
    extra: Extra
    """
    preprocessで生成された追加データ
    """


@dataclass
class PostprocessArgs[Params: BaseModel]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    analysis_results: pd.DataFrame
    """
    各レーンの解析結果を格納したDataFrame
    """


@dataclass
class PostprocessArgsWithPreprocess[Params: BaseModel, Extra]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    analysis_results: pd.DataFrame
    """
    各レーンの解析結果を格納したDataFrame
    """
    extra: Extra
    """
    preprocessで生成された追加データ
    """


@dataclass
class _BaseState:
    stdout: IO[bytes]
    stderr: IO[bytes]


@dataclass
class _AnalyzeSeqState[ParamsT: BaseModel, ImageInputModelT: BaseModel](_BaseState):
    """
    複数ターゲットのシーケンシャル解析モード用のState

    ANALYSISRUN_MODE=analyzeseq で使用される。
    """

    parsed_input: AnalyzeSeqInputModel[ParamsT, ImageInputModelT]
    field_numbers: list[int]


@dataclass
class _SequentialState(_BaseState):
    cleansed_lanes: dict[str, Lanes]
    sample_pairs: list[tuple[str, str]]
    field_numbers: list[int]


@dataclass
class _ParallelState(_BaseState):
    raw_data: dict[str, pd.DataFrame]
    cleansed_lanes: dict[str, Lanes]
    sample_pairs: list[tuple[str, str]]
    output_dir: Path
    entrypoint: Path
    field_numbers: list[int]


@dataclass
class _ParallelStreamingState(_BaseState):
    raw_data: dict[str, pd.DataFrame]
    cleansed_lanes: dict[str, Lanes]
    sample_pairs: list[tuple[str, str]]
    entrypoint: Path
    field_numbers: list[int]


@dataclass
class AnalysisContext[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
]:
    """
    数値解析のコンテキスト。
    解析の入力、実行モードなどを保持し、モードに応じて解析の実行を行う。
    """

    params: Params
    image_analysis_results: Type[ImageAnalysisResults]
    output: Output
    state: (
        _AnalyzeSeqState[Params, BaseModel]
        | _SequentialState
        | _ParallelState
        | _ParallelStreamingState
    )

    @property
    def mode(
        self,
    ) -> Literal[
        "analyze-seq",
        "sequential",
        "parallel-entrypoint",
        "parallel-entrypoint-streaming",
    ]:
        match self.state:
            case _AnalyzeSeqState():
                return "analyze-seq"
            case _SequentialState():
                return "sequential"
            case _ParallelState():
                return "parallel-entrypoint"
            case _ParallelStreamingState():
                return "parallel-entrypoint-streaming"

    def run_analysis(
        self,
        analyze: Callable[[AnalyzeArgs[Params, ImageAnalysisResults]], pd.Series],
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        read_contextで読み込んだコンテキストに基づいて数値解析を実行する。

        - mode="analyze-seq": 複数レーンの解析をシーケンシャルに行い、各レーンの解析結果をまとめたtarデータを標準出力に書き出す。その後exitする。
        - mode="sequential": multiprocessingを活用できないJupyter notebook環境において、全レーンをシーケンシャルに処理し、DataFrameを返す。
        - mode="parallel-entrypoint": entrypointを並列起動し、各プロセスで複数レーンのanalyzeseqを実行して結果を集約する。集約後は必要に応じて後処理を加え、DataFrameを返す。
        - mode="parallel-entrypoint-streaming": entrypointを並列起動し、各プロセスで複数レーンのanalyzeseqを実行して結果を集約する。集約後はtarデータを標準出力に書き出し、その後exitする。

        :param analyze: レーンごとに実行される解析処理の実装。
        :param postprocess: 後処理の実装（任意）。全レーンの解析結果をもとにさらに判定を加えたい場合に使用する。
        :return: 解析結果を格納したDataFrame。
        :raises SystemExit: 成功/失敗を問わず、分散環境で実行される場合にスローされる。
        """
        stdout = self.state.stdout
        stderr = self.state.stderr

        try:
            match self.state:
                case _AnalyzeSeqState():
                    return self._run_analyze_seq(analyze, stdout, stderr)
                case _SequentialState():
                    return self._run_sequential(analyze, postprocess)
                case _ParallelState():
                    return self._run_parallel(postprocess)
                case _ParallelStreamingState():
                    return self._run_parallel_streaming(postprocess, stdout, stderr)
        except SystemExit:
            raise
        except Exception as exc:
            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                "解析処理中に不明なエラーが発生しました。開発者に確認してください。",
                stdout,
                stderr,
                exc,
            )

    def run_analysis_with_preprocess[
        PreprocessedImageAnalysisResults: NamedTupleLike[pd.DataFrame],
        Extra,
    ](
        self,
        preprocessed_image_analysis_results: Type[PreprocessedImageAnalysisResults],
        preprocess: Callable[
            [PreprocessArgs[Params, ImageAnalysisResults]],
            ProcessedInputs[PreprocessedImageAnalysisResults, Extra],
        ],
        analyze: Callable[
            [
                AnalyzeArgsWithPreprocess[
                    Params,
                    PreprocessedImageAnalysisResults,
                    Extra,
                ]
            ],
            pd.Series,
        ],
        postprocess: Optional[
            Callable[
                [PostprocessArgsWithPreprocess[Params, Extra]],
                pd.DataFrame,
            ]
        ] = None,
    ) -> pd.DataFrame:
        """
        read_contextで読み込んだコンテキストに基づいて、preprocess -> analyze -> postprocess の順で数値解析を実行する。

        - mode="analyze-seq": 複数レーンの前処理と解析をシーケンシャルに行い、各レーンの解析結果をまとめたtarデータを標準出力に書き出す。その後exitする。
        - mode="sequential": multiprocessingを活用できないJupyter notebook環境において、全レーンをシーケンシャルに処理し、DataFrameを返す。
        - mode="parallel-entrypoint": entrypointを並列起動し、各プロセスで複数レーンの前処理と解析を実行して結果を集約する。集約後は必要に応じて後処理を加え、DataFrameを返す。
        - mode="parallel-entrypoint-streaming": entrypointを並列起動し、各プロセスで複数レーンの前処理と解析を実行して結果を集約する。集約後はtarデータを標準出力に書き出し、その後exitする。

        :param preprocess: レーンごとに実行される前処理の実装。
        :param analyze: 前処理済みデータを受け取って実行される解析処理の実装。
        :param postprocess: 後処理の実装（任意）。全レーンの解析結果と前処理結果を受け取る。
        :return: 解析結果を格納したDataFrame。
        :raises SystemExit: 成功/失敗を問わず、分散環境で実行される場合にスローされる。
        """
        stdout = self.state.stdout
        stderr = self.state.stderr

        try:
            match self.state:
                case _AnalyzeSeqState():
                    return self._run_analyze_seq_with_preprocess(
                        preprocessed_image_analysis_results,
                        analyze,
                        stdout,
                        stderr,
                    )
                case _SequentialState():
                    return self._run_sequential_with_preprocess(
                        preprocessed_image_analysis_results,
                        preprocess,
                        analyze,
                        postprocess,
                    )
                case _ParallelState():
                    return self._run_parallel_with_preprocess(
                        preprocessed_image_analysis_results,
                        preprocess,
                        postprocess,
                    )
                case _ParallelStreamingState():
                    return self._run_parallel_streaming_with_preprocess(
                        preprocessed_image_analysis_results,
                        preprocess,
                        postprocess,
                        stdout,
                        stderr,
                    )
        except SystemExit:
            raise
        except Exception as exc:
            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                "解析処理中に不明なエラーが発生しました。開発者に確認してください。",
                stdout,
                stderr,
                exc,
            )

    def _run_analyze_seq(
        self,
        analyze: Callable[[AnalyzeArgs[Params, ImageAnalysisResults]], pd.Series],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> pd.DataFrame:
        """
        analyzeseqモードで、複数ターゲットの解析を単一プロセス内で順次実行し、結果をtar形式で標準出力に出力する。

        parallel-entrypoint 系モードで起動された子プロセスから利用される想定であり、
        各ターゲットは同一プロセス内で順次処理される。
        いずれかのターゲットでエラーが発生した場合は、即座に全体を失敗として扱う。

        出力フォーマット:
            {
                "{data_name}/analysis_result": BytesIO (pd.Series as CSV),
                "{data_name}/images/{image_name}": BytesIO (PNG/image files),
            }

        Raises
        ------
        SystemExit
            処理完了時（成功/失敗問わず）
        """
        assert isinstance(self.state, _AnalyzeSeqState)
        state = self.state
        parsed: AnalyzeSeqInputModel = state.parsed_input
        field_numbers = state.field_numbers

        # Get specs for cleansing
        specs = _get_image_analysis_specs(self.image_analysis_results)
        cleansed_data = _load_and_cleanse_image_results(
            parsed.image_analysis_results,
            specs,
            serialization="pickle",
        )

        # TARストリームを開く（パイプモード: w|）
        with tarfile.open(fileobj=stdout, mode="w|") as tar:
            # 各ターゲットをシーケンシャルに処理
            lanes_by_name = _build_lanes_by_result_name(
                cleansed_data,
                list(parsed.targets.keys()),
                field_numbers,
            )
            for data_name, sample_name in parsed.targets.items():
                # このターゲット専用のOutput（画像を{data_name}/images/配下に書き込む）
                output = _TarStreamingOutput(tar, prefix=f"{data_name}/images")

                try:
                    # 解析実行（画像は生成されるたびにTARに書き込まれる）
                    with redirect_stdout_to_stderr(stderr):
                        series = analyze(
                            AnalyzeArgs(
                                _copy_params(self.params),
                                data_name,
                                sample_name,
                                _build_lane_dataframe_namedtuple(
                                    self.image_analysis_results,
                                    lanes_by_name,
                                    data_name,
                                ),
                                output,
                            )
                        )
                    _ensure_result_annotations(series, data_name, sample_name)
                except Exception as exc:
                    exit_with_error_streaming(
                        ExitCodes.PROCESSING_ERROR,
                        f"ターゲット {data_name} ({sample_name}) の解析処理中にエラーが発生しました。",
                        tar,
                        stderr,
                        exc,
                    )

                # 解析結果をTARに追加（画像の後）
                result_csv = _serialize_series(series)
                result_csv.seek(0)
                result_data = result_csv.read()

                tar_info = tarfile.TarInfo(name=f"{data_name}/analysis_result")
                tar_info.size = len(result_data)
                tar_info.pax_headers = {"is_file": "true"}
                tar.addfile(tar_info, BytesIO(result_data))

        # 正常終了: TAR自動クローズ済み
        stdout.flush()
        sys.exit(0)

    def _run_analyze_seq_with_preprocess[
        PreprocessedImageAnalysisResults: NamedTupleLike[pd.DataFrame],
        Extra,
    ](
        self,
        preprocessed_image_analysis_results: Type[PreprocessedImageAnalysisResults],
        analyze: Callable[
            [
                AnalyzeArgsWithPreprocess[
                    Params,
                    PreprocessedImageAnalysisResults,
                    Extra,
                ]
            ],
            pd.Series,
        ],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _AnalyzeSeqState)
        state = self.state
        parsed: AnalyzeSeqInputModel = state.parsed_input
        field_numbers = state.field_numbers

        with redirect_stdout_to_stderr(stderr):
            if parsed.preprocessed_data is None:
                raise ValueError("preprocessed_data is required in analyzeseq mode")
            extra = _deserialize_preprocessed_data(parsed.preprocessed_data.unwrap())
            preprocessed_data = _load_image_results_raw(
                parsed.image_analysis_results,
                serialization="pickle",
            )
            lanes_by_name = _build_lanes_by_result_name(
                preprocessed_data,
                list(parsed.targets.keys()),
                field_numbers,
            )

        with tarfile.open(fileobj=stdout, mode="w|") as tar:
            for data_name, sample_name in parsed.targets.items():
                output = _TarStreamingOutput(tar, prefix=f"{data_name}/images")

                try:
                    lane_data = _build_lane_dataframe_namedtuple(
                        preprocessed_image_analysis_results,
                        lanes_by_name,
                        data_name,
                    )
                    with redirect_stdout_to_stderr(stderr):
                        series = analyze(
                            AnalyzeArgsWithPreprocess(
                                _copy_params(self.params),
                                data_name,
                                sample_name,
                                lane_data,
                                output,
                                extra,
                            )
                        )
                    _ensure_result_annotations(series, data_name, sample_name)
                except Exception as exc:
                    exit_with_error_streaming(
                        ExitCodes.PROCESSING_ERROR,
                        f"ターゲット {data_name} ({sample_name}) の解析処理中にエラーが発生しました。",
                        tar,
                        stderr,
                        exc,
                    )

                result_csv = _serialize_series(series)
                result_csv.seek(0)
                result_data = result_csv.read()

                tar_info = tarfile.TarInfo(name=f"{data_name}/analysis_result")
                tar_info.size = len(result_data)
                tar_info.pax_headers = {"is_file": "true"}
                tar.addfile(tar_info, BytesIO(result_data))

        stdout.flush()
        sys.exit(0)

    def _run_sequential(
        self,
        analyze: Callable[[AnalyzeArgs[Params, ImageAnalysisResults]], pd.Series],
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _SequentialState)
        state = self.state
        cleansed_lanes = state.cleansed_lanes
        sample_pairs = state.sample_pairs

        results: list[pd.Series] = []
        for data_name, sample_name in sample_pairs:
            lane_data = _build_lane_dataframe_namedtuple(
                self.image_analysis_results,
                cleansed_lanes,
                data_name,
            )
            series = analyze(
                AnalyzeArgs(
                    _copy_params(self.params),
                    data_name,
                    sample_name,
                    lane_data,
                    self.output,
                )
            )
            _ensure_result_annotations(series, data_name, sample_name)
            results.append(series)

        result_df = pd.DataFrame(results)
        if postprocess:
            postprocessed = postprocess(
                PostprocessArgs(_copy_params(self.params), result_df)
            )
            if postprocessed is not None:
                result_df = postprocessed
        return result_df

    def _run_sequential_with_preprocess[
        PreprocessedImageAnalysisResults: NamedTupleLike[pd.DataFrame],
        Extra,
    ](
        self,
        preprocessed_image_analysis_results: Type[PreprocessedImageAnalysisResults],
        preprocess: Callable[
            [PreprocessArgs[Params, ImageAnalysisResults]],
            ProcessedInputs[PreprocessedImageAnalysisResults, Extra],
        ],
        analyze: Callable[
            [
                AnalyzeArgsWithPreprocess[
                    Params,
                    PreprocessedImageAnalysisResults,
                    Extra,
                ]
            ],
            pd.Series,
        ],
        postprocess: Optional[
            Callable[
                [PostprocessArgsWithPreprocess[Params, Extra]],
                pd.DataFrame,
            ]
        ],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _SequentialState)
        state = self.state
        cleansed_lanes = state.cleansed_lanes
        sample_pairs = state.sample_pairs
        field_numbers = state.field_numbers
        processed_inputs = preprocess(
            _build_preprocess_args(
                self.params,
                self.image_analysis_results,
                cleansed_lanes,
                {data_name: sample_name for data_name, sample_name in sample_pairs},
            )
        )
        preprocessed_lanes = _build_lanes_by_result_name(
            _namedtuple_to_dict(processed_inputs.image_analysis_results),
            _target_names(sample_pairs),
            field_numbers,
        )

        results: list[pd.Series] = []
        for data_name, sample_name in sample_pairs:
            lane_data = _build_lane_dataframe_namedtuple(
                preprocessed_image_analysis_results,
                preprocessed_lanes,
                data_name,
            )
            series = analyze(
                AnalyzeArgsWithPreprocess(
                    _copy_params(self.params),
                    data_name,
                    sample_name,
                    lane_data,
                    self.output,
                    processed_inputs.extra,
                )
            )
            _ensure_result_annotations(series, data_name, sample_name)
            results.append(series)

        result_df = pd.DataFrame(results)
        if postprocess:
            postprocessed = postprocess(
                PostprocessArgsWithPreprocess(
                    _copy_params(self.params),
                    result_df,
                    processed_inputs.extra,
                )
            )
            if postprocessed is not None:
                result_df = postprocessed
        return result_df

    def _run_parallel(
        self,
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _ParallelState)
        state = self.state
        output_dir = state.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        image_write_lock = Lock()

        def _save_streamed_image(
            _data_name: str,
            image_name: str,
            image_bytes: bytes,
            _image_type: Optional[str],
        ) -> None:
            # parallel-entrypoint では従来仕様どおりファイル名のみで保存する
            with image_write_lock:
                _save_image_bytes_to_dir(image_name, image_bytes, output_dir)

        result_df, images_by_data = self._run_parallel_entrypoint(
            raw_data=state.raw_data,
            sample_pairs=state.sample_pairs,
            entrypoint=state.entrypoint,
            stdout=state.stdout,
            stderr=state.stderr,
            output_dir=output_dir,
            on_image=_save_streamed_image,
        )
        flat_images = _flatten_images_by_data(images_by_data)
        if flat_images:
            _save_images_to_dir(flat_images, output_dir)

        if postprocess:
            postprocessed = postprocess(
                PostprocessArgs(_copy_params(self.params), result_df)
            )
            if postprocessed is not None:
                result_df = postprocessed
        return result_df

    def _run_parallel_with_preprocess[
        PreprocessedImageAnalysisResults: NamedTupleLike[pd.DataFrame],
        Extra,
    ](
        self,
        preprocessed_image_analysis_results: Type[PreprocessedImageAnalysisResults],
        preprocess: Callable[
            [PreprocessArgs[Params, ImageAnalysisResults]],
            ProcessedInputs[PreprocessedImageAnalysisResults, Extra],
        ],
        postprocess: Optional[
            Callable[
                [PostprocessArgsWithPreprocess[Params, Extra]],
                pd.DataFrame,
            ]
        ],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _ParallelState)
        state = self.state
        output_dir = state.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        image_write_lock = Lock()

        def _save_streamed_image(
            _data_name: str,
            image_name: str,
            image_bytes: bytes,
            _image_type: Optional[str],
        ) -> None:
            # parallel-entrypoint では従来仕様どおりファイル名のみで保存する
            with image_write_lock:
                _save_image_bytes_to_dir(image_name, image_bytes, output_dir)

        processed_inputs = preprocess(
            _build_preprocess_args(
                self.params,
                self.image_analysis_results,
                state.cleansed_lanes,
                {
                    data_name: sample_name
                    for data_name, sample_name in state.sample_pairs
                },
            )
        )
        result_df, images_by_data = self._run_parallel_entrypoint(
            raw_data=_namedtuple_to_dict(processed_inputs.image_analysis_results),
            sample_pairs=state.sample_pairs,
            entrypoint=state.entrypoint,
            stdout=state.stdout,
            stderr=state.stderr,
            preprocessed_data=processed_inputs.extra,
            include_preprocessed_data=True,
            output_dir=output_dir,
            on_image=_save_streamed_image,
        )
        flat_images = _flatten_images_by_data(images_by_data)
        if flat_images:
            _save_images_to_dir(flat_images, output_dir)

        if postprocess:
            postprocessed = postprocess(
                PostprocessArgsWithPreprocess(
                    _copy_params(self.params),
                    result_df,
                    processed_inputs.extra,
                )
            )
            if postprocessed is not None:
                result_df = postprocessed
        return result_df

    def _run_parallel_streaming(
        self,
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _ParallelStreamingState)
        state = self.state
        with tarfile.open(fileobj=stdout, mode="w|gz") as tar:
            tar_lock = Lock()

            def _stream_image(
                data_name: str,
                image_name: str,
                image_bytes: bytes,
                image_type: Optional[str],
            ) -> None:
                with tar_lock:
                    _add_tar_binary_entry(
                        tar,
                        f"{data_name}/images/{image_name}",
                        image_bytes,
                        image_type=image_type,
                    )
                    stdout.flush()

            try:
                result_df, errors = self._run_parallel_entrypoint_streaming(
                    raw_data=state.raw_data,
                    sample_pairs=state.sample_pairs,
                    entrypoint=state.entrypoint,
                    preprocessed_data=None,
                    include_preprocessed_data=False,
                    on_image=_stream_image,
                )
                if errors:
                    _write_parallel_chunk_errors(stderr, errors)
                    lanes = [
                        f"- {data_name} ({sample_name})"
                        for data_name, sample_name, _, _ in errors
                    ]
                    exit_with_error_streaming(
                        ExitCodes.PROCESSING_ERROR,
                        "\n".join(["解析中にエラーが発生しました。", *lanes]),
                        tar,
                        stderr,
                    )

                if postprocess:
                    with redirect_stdout_to_stderr(stderr):
                        postprocessed = postprocess(
                            PostprocessArgs(_copy_params(self.params), result_df)
                        )
                    if postprocessed is not None:
                        result_df = postprocessed

                _write_parallel_streaming_result_entries(tar, result_df)
            except SystemExit:
                raise
            except Exception as exc:
                exit_with_error_streaming(
                    ExitCodes.PROCESSING_ERROR,
                    "解析処理中に不明なエラーが発生しました。開発者に確認してください。",
                    tar,
                    stderr,
                    exc,
                )

        stdout.flush()
        sys.exit(0)

    def _run_parallel_streaming_with_preprocess[
        PreprocessedImageAnalysisResults: NamedTupleLike[pd.DataFrame],
        Extra,
    ](
        self,
        preprocessed_image_analysis_results: Type[PreprocessedImageAnalysisResults],
        preprocess: Callable[
            [PreprocessArgs[Params, ImageAnalysisResults]],
            ProcessedInputs[PreprocessedImageAnalysisResults, Extra],
        ],
        postprocess: Optional[
            Callable[
                [PostprocessArgsWithPreprocess[Params, Extra]],
                pd.DataFrame,
            ]
        ],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _ParallelStreamingState)
        state = self.state
        with tarfile.open(fileobj=stdout, mode="w|gz") as tar:
            tar_lock = Lock()

            def _stream_image(
                data_name: str,
                image_name: str,
                image_bytes: bytes,
                image_type: Optional[str],
            ) -> None:
                with tar_lock:
                    _add_tar_binary_entry(
                        tar,
                        f"{data_name}/images/{image_name}",
                        image_bytes,
                        image_type=image_type,
                    )
                    stdout.flush()

            try:
                with redirect_stdout_to_stderr(stderr):
                    processed_inputs = preprocess(
                        _build_preprocess_args(
                            self.params,
                            self.image_analysis_results,
                            state.cleansed_lanes,
                            {
                                data_name: sample_name
                                for data_name, sample_name in state.sample_pairs
                            },
                        )
                    )

                result_df, errors = self._run_parallel_entrypoint_streaming(
                    raw_data=_namedtuple_to_dict(
                        processed_inputs.image_analysis_results
                    ),
                    sample_pairs=state.sample_pairs,
                    entrypoint=state.entrypoint,
                    preprocessed_data=processed_inputs.extra,
                    include_preprocessed_data=True,
                    on_image=_stream_image,
                )
                if errors:
                    _write_parallel_chunk_errors(stderr, errors)
                    lanes = [
                        f"- {data_name} ({sample_name})"
                        for data_name, sample_name, _, _ in errors
                    ]
                    exit_with_error_streaming(
                        ExitCodes.PROCESSING_ERROR,
                        "\n".join(["解析中にエラーが発生しました。", *lanes]),
                        tar,
                        stderr,
                    )

                if postprocess:
                    with redirect_stdout_to_stderr(stderr):
                        postprocessed = postprocess(
                            PostprocessArgsWithPreprocess(
                                _copy_params(self.params),
                                result_df,
                                processed_inputs.extra,
                            )
                        )
                    if postprocessed is not None:
                        result_df = postprocessed

                _write_parallel_streaming_result_entries(tar, result_df)
            except SystemExit:
                raise
            except Exception as exc:
                exit_with_error_streaming(
                    ExitCodes.PROCESSING_ERROR,
                    "解析処理中に不明なエラーが発生しました。開発者に確認してください。",
                    tar,
                    stderr,
                    exc,
                )

        stdout.flush()
        sys.exit(0)

    def _run_parallel_entrypoint_streaming(
        self,
        raw_data: dict[str, pd.DataFrame],
        sample_pairs: list[tuple[str, str]],
        entrypoint: Path,
        preprocessed_data: Any | None,
        include_preprocessed_data: bool,
        on_image: Callable[[str, str, bytes, Optional[str]], None],
    ) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]]]:
        if not sample_pairs:
            return pd.DataFrame(), []

        params_payload = self.params.model_dump_json()
        core_count = os.cpu_count() or 1
        process_count = min(len(sample_pairs), max(core_count, 1))
        target_chunks = _split_evenly_in_order(sample_pairs, process_count)

        def _run_chunk(
            targets: list[tuple[str, str]],
        ) -> tuple[list[pd.Series], Optional[tuple[str, str, str, str]]]:
            tar_buf = _build_analyze_seq_tar_buffer(
                params_payload,
                targets,
                raw_data,
                preprocessed_data=preprocessed_data,
                include_preprocessed_data=include_preprocessed_data,
            )
            result = _run_entrypoint_with_tar_streaming(
                entrypoint,
                tar_buf,
                "analyzeseq",
                on_image=on_image,
            )
            err_out = result.stderr

            if result.returncode != 0:
                err_msg = result.error or "不明なエラーが発生しました。"
                failed_data, failed_sample = _infer_failed_target(err_msg, targets)
                return [], (failed_data, failed_sample, err_msg, err_out)

            if not result.tar_read_ok:
                data_name, sample_name = targets[0]
                return (
                    [],
                    (
                        data_name,
                        sample_name,
                        f"{data_name}の解析結果の読み込みに失敗しました",
                        err_out,
                    ),
                )

            if result.error is not None:
                failed_data, failed_sample = _infer_failed_target(result.error, targets)
                return [], (failed_data, failed_sample, result.error, err_out)

            chunk_results: list[pd.Series] = []
            for data_name, sample_name in targets:
                series_buf = result.analysis_results.get(data_name)
                if not isinstance(series_buf, BytesIO):
                    return (
                        [],
                        (
                            data_name,
                            sample_name,
                            f"{data_name}の解析結果の形式が不正です。",
                            err_out,
                        ),
                    )
                series = _deserialize_series(series_buf)
                _ensure_result_annotations(series, data_name, sample_name)
                chunk_results.append(series)
            return chunk_results, None

        with ThreadPoolExecutor(max_workers=process_count) as executor:
            chunk_results = list(executor.map(_run_chunk, target_chunks))

        errors: list[tuple[str, str, str, str]] = []
        results: list[pd.Series] = []

        for series_list, err in chunk_results:
            if err:
                errors.append(err)
            else:
                results.extend(series_list)

        return pd.DataFrame(results), errors

    def _run_parallel_entrypoint(
        self,
        raw_data: dict[str, pd.DataFrame],
        sample_pairs: list[tuple[str, str]],
        entrypoint: Path,
        stdout: IO[bytes],
        stderr: IO[bytes],
        preprocessed_data: Any | None = None,
        include_preprocessed_data: bool = False,
        output_dir: Path | None = None,
        on_image: Optional[Callable[[str, str, bytes, Optional[str]], None]] = None,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, BytesIO]]]:
        if not sample_pairs:
            return pd.DataFrame(), {}

        params_payload = self.params.model_dump_json()
        core_count = os.cpu_count() or 1
        process_count = min(len(sample_pairs), max(core_count, 1))
        target_chunks = _split_evenly_in_order(sample_pairs, process_count)

        def _run_chunk(
            targets: list[tuple[str, str]],
        ) -> tuple[
            list[pd.Series],
            dict[str, dict[str, BytesIO]],
            Optional[tuple[str, str, str, str]],
        ]:
            tar_buf = _build_analyze_seq_tar_buffer(
                params_payload,
                targets,
                raw_data,
                preprocessed_data=preprocessed_data,
                include_preprocessed_data=include_preprocessed_data,
            )
            if on_image is not None:
                result = _run_entrypoint_with_tar_streaming(
                    entrypoint,
                    tar_buf,
                    "analyzeseq",
                    on_image=on_image,
                )
                err_out = result.stderr
                images_by_data: dict[str, dict[str, BytesIO]] = {}

                if result.returncode != 0:
                    err_msg = result.error or "不明なエラーが発生しました。"
                    failed_data, failed_sample = _infer_failed_target(err_msg, targets)
                    return (
                        [],
                        images_by_data,
                        (failed_data, failed_sample, err_msg, err_out),
                    )

                if not result.tar_read_ok:
                    data_name, sample_name = targets[0]
                    return (
                        [],
                        images_by_data,
                        (
                            data_name,
                            sample_name,
                            f"{data_name}の解析結果の読み込みに失敗しました",
                            err_out,
                        ),
                    )

                if result.error is not None:
                    failed_data, failed_sample = _infer_failed_target(
                        result.error, targets
                    )
                    return (
                        [],
                        images_by_data,
                        (failed_data, failed_sample, result.error, err_out),
                    )

                chunk_results: list[pd.Series] = []
                for data_name, sample_name in targets:
                    series_buf = result.analysis_results.get(data_name)
                    if not isinstance(series_buf, BytesIO):
                        return (
                            [],
                            images_by_data,
                            (
                                data_name,
                                sample_name,
                                f"{data_name}の解析結果の形式が不正です。",
                                err_out,
                            ),
                        )
                    series = _deserialize_series(series_buf)
                    _ensure_result_annotations(series, data_name, sample_name)
                    chunk_results.append(series)
                return chunk_results, images_by_data, None

            proc = _run_entrypoint_with_tar(entrypoint, tar_buf, "analyzeseq")
            err_out = proc.stderr.decode("utf-8", errors="ignore")
            tar_result = _try_read_tar_from_bytes(proc.stdout)
            images_by_data = (
                _extract_images_by_data_from_tar_dict(tar_result)
                if isinstance(tar_result, dict)
                else {}
            )

            if proc.returncode != 0:
                err_msg = (
                    str(tar_result["error"])
                    if isinstance(tar_result, dict) and "error" in tar_result
                    else "不明なエラーが発生しました。"
                )
                failed_data, failed_sample = _infer_failed_target(err_msg, targets)
                return (
                    [],
                    images_by_data,
                    (failed_data, failed_sample, err_msg, err_out),
                )

            if not isinstance(tar_result, dict):
                data_name, sample_name = targets[0]
                return (
                    [],
                    images_by_data,
                    (
                        data_name,
                        sample_name,
                        f"{data_name}の解析結果の読み込みに失敗しました",
                        err_out,
                    ),
                )

            if "error" in tar_result:
                err_msg = str(tar_result["error"])
                failed_data, failed_sample = _infer_failed_target(err_msg, targets)
                return (
                    [],
                    images_by_data,
                    (failed_data, failed_sample, err_msg, err_out),
                )

            chunk_results: list[pd.Series] = []
            for data_name, sample_name in targets:
                lane_result = tar_result.get(data_name)
                if not isinstance(lane_result, dict):
                    return (
                        [],
                        images_by_data,
                        (
                            data_name,
                            sample_name,
                            f"{data_name}の解析結果の形式が不正です。",
                            err_out,
                        ),
                    )
                series_buf = lane_result.get("analysis_result")
                if not isinstance(series_buf, BytesIO):
                    return (
                        [],
                        images_by_data,
                        (
                            data_name,
                            sample_name,
                            f"{data_name}の解析結果の形式が不正です。",
                            err_out,
                        ),
                    )
                series = _deserialize_series(series_buf)
                _ensure_result_annotations(series, data_name, sample_name)
                chunk_results.append(series)
            return chunk_results, images_by_data, None

        with ThreadPoolExecutor(max_workers=process_count) as executor:
            chunk_results = list(executor.map(_run_chunk, target_chunks))

        errors: list[tuple[str, str, str, str]] = []
        results: list[pd.Series] = []
        collected_images: dict[str, dict[str, BytesIO]] = {}

        for series_list, images_by_data, err in chunk_results:
            for data_name, images in images_by_data.items():
                lane_images = collected_images.setdefault(data_name, {})
                lane_images.update(images)
            if err:
                errors.append(err)
            else:
                results.extend(series_list)

        if errors:
            if output_dir is not None and collected_images:
                _save_images_to_dir(
                    _flatten_images_by_data(collected_images),
                    output_dir,
                )
            _write_parallel_chunk_errors(stderr, errors)
            lanes = [
                f"- {data_name} ({sample_name})"
                for data_name, sample_name, _, _ in errors
            ]
            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                "\n".join(["解析中にエラーが発生しました。", *lanes]),
                stdout,
                stderr,
            )

        return pd.DataFrame(results), collected_images


def read_context[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    params: Type[Params],
    image_analysis_results: Type[ImageAnalysisResults],
    manual_input: Optional[ManualInput[Params]] = None,
    stdin: Optional[IO[bytes]] = None,
    stdout: Optional[IO[bytes]] = None,
    output_dir: Optional[str | Path] = None,
) -> AnalysisContext[Params, ImageAnalysisResults]:
    """
    引数、環境変数、標準入力を読み取り、実行モードに応じたAnalysisContextを構築する。

    `ANALYSISRUN_MODE` と対話環境の有無に応じて、
    analyze-seq / sequential / parallel-entrypoint / parallel-entrypoint-streaming
    のいずれかのstateを持つコンテキストを返す。

    :param params: 解析全体に関わるパラメータを定義するクラス
    :param image_analysis_results: 解析対象となる画像解析結果を定義するクラス
    :param manual_input: 入力データ。Jupyter notebook環境で実行する際には必須
    :param stdin: 標準入力
    :param stdout: 標準出力
    :param output_dir: ローカル実行する際の画像データの出力先ディレクトリ。
        画像解析結果がファイルパスで与えられる場合にはそのファイルが存在するディレクトリ、それ以外はカレントディレクトリがデフォルト値となる。
    :return: AnalysisContextオブジェクト
    """
    _stdin: IO[bytes] = stdin or sys.stdin.buffer
    _stdout: IO[bytes] = stdout or sys.stdout.buffer
    _stderr: IO[bytes] = sys.stderr.buffer
    interactivity = get_interactivity()
    env_mode = os.getenv("ANALYSISRUN_MODE")

    try:
        specs = _get_image_analysis_specs(image_analysis_results)
        image_analysis_results_input_model = create_image_analysis_results_input_model(
            image_analysis_results
        )
    except Exception as exc:
        exit_with_error(
            ExitCodes.INVALID_USAGE,
            "ImageAnalysisResultsの定義が不正です。各フィールドにimage_analysis_result_spec(...)を設定してください。",
            _stdout,
            _stderr,
            exc,
        )
    RuntimeInputModel = InputModel[params, image_analysis_results_input_model]
    field_numbers = [i + 1 for i in range(12)]
    output_dir_path = Path(output_dir) if output_dir else None

    if env_mode == "analyzeseq":
        # 複数ターゲットのシーケンシャル解析モード
        try:
            tar_dict = read_tar_as_dict(_stdin)
            tar_dict["params"] = _maybe_load_json(tar_dict.get("params"))
            tar_dict["targets"] = _maybe_load_json(tar_dict.get("targets"))

            AnalyzeSeqInput = AnalyzeSeqInputModel[
                params, image_analysis_results_input_model
            ]
            parsed = AnalyzeSeqInput(**tar_dict)
        except Exception as exc:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "入力データの読み込みに失敗しました。入力形式を確認してください。",
                _stdout,
                _stderr,
                exc,
            )
        return AnalysisContext[Params, ImageAnalysisResults](
            params=parsed.params,
            image_analysis_results=image_analysis_results,
            output=_NullOutput(),
            state=_AnalyzeSeqState(
                stdout=_stdout,
                stderr=_stderr,
                parsed_input=parsed,
                field_numbers=field_numbers,
            ),
        )

    elif env_mode == "showschema":
        schema = _build_streaming_input_schema(params, image_analysis_results)
        _stdout.write(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
        _stdout.flush()
        raise SystemExit(0)

    elif env_mode is not None:
        exit_with_error(
            ExitCodes.INVALID_USAGE,
            f"未対応のANALYSISRUN_MODEです: {env_mode}",
            _stdout,
            _stderr,
        )

    mode: Literal["sequential", "parallel-entrypoint", "parallel-entrypoint-streaming"]
    if interactivity is None:
        mode = "parallel-entrypoint-streaming"
        try:
            tar_dict = read_tar_as_dict(_stdin)
            tar_dict["params"] = _maybe_load_json(tar_dict.get("params"))
            runtime_input = RuntimeInputModel(**tar_dict)
        except Exception as exc:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "入力データの読み込みに失敗しました。入力形式を確認してください。",
                _stdout,
                _stderr,
                exc,
            )
    elif manual_input is not None:
        if interactivity == "notebook":
            mode = "sequential"
        else:
            mode = "parallel-entrypoint"
        try:
            iar_input = image_analysis_results_input_model(
                **manual_input.image_analysis_results
            )
            runtime_input = RuntimeInputModel(
                image_analysis_results=iar_input,
                sample_names=manual_input.sample_names,  # type: ignore
                params=manual_input.params,
            )
        except ValidationError as exc:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "manual_inputの形式が正しくありません。",
                _stdout,
                _stderr,
                exc,
            )
    else:
        mode = "sequential" if interactivity == "notebook" else "parallel-entrypoint"
        if mode == "sequential":
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "Jupyter notebook環境ではmanual_inputの指定が必須です。",
                _stdout,
                _stderr,
            )

        # CLI等ではインタラクティブな入力を通じてRuntimeInputModelを組み立てる。
        runtime_input = scan_model_input(RuntimeInputModel)

    raw_data = _load_image_results_raw(
        runtime_input.image_analysis_results,
        serialization="csv",
    )
    sample_pairs = [
        (data, sample)
        for data, sample in read_dict(
            runtime_input.sample_names.unwrap(),
            key="data",
            value="sample",
        ).items()
    ]
    if not sample_pairs:
        exit_with_error(
            ExitCodes.INVALID_USAGE,
            "サンプル名CSVファイルが空です。",
            _stdout,
            _stderr,
        )

    cleansed_data = {
        name: _apply_cleansing_pipeline(df, specs[name])
        for name, df in raw_data.items()
    }
    target_names = _target_names(sample_pairs)
    cleansed_lanes = _build_lanes_by_result_name(
        cleansed_data,
        target_names,
        field_numbers,
    )

    if mode == "sequential":
        if output_dir_path is None:
            output_dir_path = _derive_output_dir(runtime_input.image_analysis_results)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_impl = _FileOutput(output_dir_path)
        return AnalysisContext[Params, ImageAnalysisResults](
            params=runtime_input.params,
            image_analysis_results=image_analysis_results,
            output=output_impl,
            state=_SequentialState(
                stdout=_stdout,
                stderr=_stderr,
                cleansed_lanes=cleansed_lanes,
                sample_pairs=sample_pairs,
                field_numbers=field_numbers,
            ),
        )

    entrypoint = get_entrypoint()
    if entrypoint is None:
        exit_with_error(
            ExitCodes.INVALID_USAGE,
            "エントリーポイントとなるスクリプトのパス取得に失敗したため、並列実行させることができません。",
            _stdout,
            _stderr,
        )

    if mode == "parallel-entrypoint":
        if output_dir_path is None:
            output_dir_path = _derive_output_dir(runtime_input.image_analysis_results)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_impl = _FileOutput(output_dir_path)
        return AnalysisContext[Params, ImageAnalysisResults](
            params=runtime_input.params,
            image_analysis_results=image_analysis_results,
            output=output_impl,
            state=_ParallelState(
                stdout=_stdout,
                stderr=_stderr,
                raw_data=raw_data,
                cleansed_lanes=cleansed_lanes,
                sample_pairs=sample_pairs,
                output_dir=output_dir_path,
                entrypoint=entrypoint,
                field_numbers=field_numbers,
            ),
        )

    return AnalysisContext[Params, ImageAnalysisResults](
        params=runtime_input.params,
        image_analysis_results=image_analysis_results,
        output=_NullOutput(),
        state=_ParallelStreamingState(
            stdout=_stdout,
            stderr=_stderr,
            raw_data=raw_data,
            cleansed_lanes=cleansed_lanes,
            sample_pairs=sample_pairs,
            entrypoint=entrypoint,
            field_numbers=field_numbers,
        ),
    )


class _TarStreamingOutput(Output):
    """
    画像をストリームでTARに即座に書き込むOutputクラス。

    画像が生成されるたびにTARエントリとして追加し、
    メモリに蓄積しない。
    """

    def __init__(self, tar: tarfile.TarFile, prefix: str = "images"):
        self.tar = tar
        self.prefix = prefix

    def __call__(self, fig, name: str, image_type: str, **kwargs) -> None:
        # 画像をバッファに保存
        buf = FileIO({"image_type": image_type})
        fig.savefig(buf, **kwargs)
        buf.seek(0)
        image_data = buf.read()

        # TARエントリとして即座に書き込み
        tar_info = tarfile.TarInfo(name=f"{self.prefix}/{name}")
        tar_info.size = len(image_data)
        tar_info.pax_headers = {"is_file": "true", "image_type": image_type}
        self.tar.addfile(tar_info, BytesIO(image_data))

        # メモリ解放
        fig.clear()
        plt.close(fig)


class _FileOutput(Output):
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, fig, name: str, image_type: str, **kwargs) -> None:
        path = self.base_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **kwargs)
        fig.clear()
        plt.close(fig)


class _NullOutput(Output):
    def __call__(self, fig, name: str, image_type: str, **kwargs) -> None:
        fig.clear()
        plt.close(fig)


def _load_image_results_raw(
    image_analysis_results_model: BaseModel,
    *,
    serialization: _ImageResultsSerialization,
) -> dict[str, pd.DataFrame]:
    """
    ImageAnalysisResultsInputから各データをDataFrameとして読み込む。

    クレンジングは行わず、生データを返す。
    """

    raw: dict[str, pd.DataFrame] = {}
    for name in type(image_analysis_results_model).model_fields:
        vfile = getattr(image_analysis_results_model, name)
        if serialization == "csv":
            raw[name] = _deserialize_dataframe_csv(vfile.unwrap())
        else:
            raw[name] = _deserialize_dataframe_pickle(vfile.unwrap())
    return raw


def _build_preprocess_args[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    params: Params,
    image_analysis_results_type: Type[ImageAnalysisResults],
    cleansed_lanes: Mapping[str, Lanes],
    targets: dict[str, str],
) -> PreprocessArgs[Params, ImageAnalysisResults]:
    copied = {name: lanes.whole_data.copy() for name, lanes in cleansed_lanes.items()}
    return PreprocessArgs(
        params=_copy_params(params),
        image_analysis_results=_build_dataframe_namedtuple(
            image_analysis_results_type,
            MappingProxyType(copied),
        ),
        targets=targets,
    )


def _copy_params[Params: BaseModel](params: Params) -> Params:
    return params.model_copy(deep=True)


def _build_dataframe_namedtuple[
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    image_analysis_results_type: Type[ImageAnalysisResults],
    values: Mapping[str, pd.DataFrame],
) -> ImageAnalysisResults:
    expected_fields = image_analysis_results_type._fields  # type: ignore[attr-defined]
    actual_fields = tuple(values.keys())
    if set(expected_fields) != set(actual_fields):
        raise ValueError(
            f"image_analysis_results must have the same fields as "
            f"{image_analysis_results_type.__name__}: "
            f"expected {expected_fields}, got {actual_fields}"
        )
    return image_analysis_results_type(
        **{name: values[name] for name in expected_fields}
    )


def _namedtuple_to_dict(value: NamedTupleLike) -> dict[str, Any]:
    if not isinstance(value, tuple) or not hasattr(value, "_fields"):
        raise TypeError("value must be a NamedTuple-like instance")
    return {name: getattr(value, name) for name in value._fields}


def _target_names(sample_pairs: list[tuple[str, str]]) -> list[str]:
    return list(dict.fromkeys(data_name for data_name, _ in sample_pairs))


def _apply_cleansing_pipeline(
    data: pd.DataFrame, spec: _ImageAnalysisResultSpec
) -> pd.DataFrame:
    cleansed: pd.DataFrame | CleansedData = data
    for fn in spec.cleansing:
        cleansed = fn(cleansed)

    if isinstance(cleansed, CleansedData):
        return cleansed._data

    raise TypeError(  # type: ignore
        f"cleansing function must return a CleansedData, got {type(cleansed)!r}"
    )


def _load_and_cleanse_image_results(
    image_analysis_results_model: BaseModel,
    specs: dict[str, _ImageAnalysisResultSpec],
    *,
    serialization: _ImageResultsSerialization,
) -> dict[str, pd.DataFrame]:
    cleansed_data: dict[str, pd.DataFrame] = {}
    raw_data = _load_image_results_raw(
        image_analysis_results_model,
        serialization=serialization,
    )
    for name, spec in specs.items():
        df = raw_data[name]
        cleansed_data[name] = _apply_cleansing_pipeline(df, spec)
    return cleansed_data


def _build_lanes_by_result_name(
    image_results: Mapping[str, pd.DataFrame],
    target_names: list[str],
    field_numbers: list[int],
) -> dict[str, Lanes]:
    return {
        name: scan(
            whole_data=df,
            target_data=target_names,
            field_numbers=field_numbers,
        )
        for name, df in image_results.items()
    }


def _build_lane_dataframe_namedtuple[
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    image_analysis_results_type: Type[ImageAnalysisResults],
    lanes_by_name: Mapping[str, Lanes],
    data_name: str,
) -> ImageAnalysisResults:
    lane_data = {
        name: lanes.get(data_name).data.copy() for name, lanes in lanes_by_name.items()
    }
    return _build_dataframe_namedtuple(image_analysis_results_type, lane_data)


def _ensure_result_annotations(series: pd.Series, data_name: str, sample_name: str):
    if "data" not in series:
        series["data"] = data_name
    if "sample_name" not in series:
        series["sample_name"] = sample_name


def _serialize_series(series: pd.Series) -> BytesIO:
    # CSVはDataFrame向けのため、1行DataFrameとして保存する
    df = series.to_frame().T
    return _serialize_dataframe(df)


def _serialize_dataframe(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _deserialize_dataframe_csv(value: Any) -> pd.DataFrame:
    if isinstance(value, BytesIO):
        value.seek(0)
    return pd.read_csv(value, dtype=_CSV_DTYPE)


def _deserialize_dataframe_pickle(value: Any) -> pd.DataFrame:
    """BytesIO/PathからDataFrameを復元する（DataFrame pickle専用）。"""

    try:
        if isinstance(value, Path):
            loaded = pd.read_pickle(value)
        elif isinstance(value, BytesIO):
            value.seek(0)
            loaded = pickle.load(value)
        else:
            loaded = pd.read_pickle(value)
    except Exception as exc:
        raise RuntimeError(
            "画像解析結果データの読み込みに失敗しました。DataFrameのpickleデータを指定してください。"
        ) from exc

    if not isinstance(loaded, pd.DataFrame):
        raise RuntimeError(
            "画像解析結果データはDataFrameのpickleデータを指定してください。"
        )
    return loaded


def _deserialize_dataframe_with_leading_zeroes(value: BytesIO) -> pd.DataFrame:
    """BytesIO/PathからDataFrameを復元する（CSV専用）。"""

    # まずはすべて文字列として一度読み込み、「leading-zeroが含まれる（文字列として読み込むべきと思われる）列」だけを str に固定する。
    value.seek(0)
    initial = pd.read_csv(value, dtype=str)

    str_dtypes = _CSV_DTYPE.copy()
    for col in initial.columns:
        if initial[col].str.match(r"^\s*[+-]?0\d+", na=False).any():
            str_dtypes[col] = str

    value.seek(0)
    return pd.read_csv(value, dtype=str_dtypes)


def _deserialize_series(buf: BytesIO) -> pd.Series:
    df = _deserialize_dataframe_with_leading_zeroes(buf)
    if df.shape[0] != 1:
        raise RuntimeError("解析結果の行数が不正です。")
    return df.iloc[0]


def _serialize_preprocessed_data(value: Any) -> BytesIO:
    buf = BytesIO()
    pickle.dump(value, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    return buf


def _deserialize_preprocessed_data(value: Any) -> Any:
    try:
        if isinstance(value, Path):
            with open(value, "rb") as f:
                return pickle.load(f)
        if isinstance(value, BytesIO):
            value.seek(0)
            return pickle.load(value)
        raise RuntimeError("前処理済みデータの形式が不正です。")
    except Exception as exc:
        raise RuntimeError(
            "前処理済みデータの読み込みに失敗しました。pickleデータを指定してください。"
        ) from exc


_CSV_DTYPE: dict[str, type] = {
    "Entity": str,
    "Filename": str,
    "data": str,
    "sample_name": str,
}


def _maybe_load_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _derive_output_dir(image_analysis_results_model: BaseModel) -> Path:
    for value in image_analysis_results_model.model_dump(mode="python").values():
        if isinstance(value, Path):
            return value.parent
    return Path.cwd()


def _extract_images_by_data_from_tar_dict(
    tar_dict: dict[str, Any],
) -> dict[str, dict[str, BytesIO]]:
    images_by_data: dict[str, dict[str, BytesIO]] = {}
    for data_name, node in tar_dict.items():
        if data_name == "error" or not isinstance(node, dict):
            continue
        images = node.get("images")
        if not isinstance(images, dict):
            continue
        lane_images: dict[str, BytesIO] = {}
        for image_name, value in images.items():
            if isinstance(value, BytesIO):
                lane_images[image_name] = value
        if lane_images:
            images_by_data[str(data_name)] = lane_images
    return images_by_data


def _flatten_images_by_data(
    images_by_data: dict[str, dict[str, BytesIO]],
) -> dict[str, BytesIO]:
    images: dict[str, BytesIO] = {}
    for lane_images in images_by_data.values():
        for image_name, value in lane_images.items():
            images[image_name] = value
    return images


def _build_analyze_seq_tar_buffer(
    params_payload: str,
    targets: list[tuple[str, str]],
    image_results: dict[str, pd.DataFrame],
    preprocessed_data: Any | None = None,
    *,
    include_preprocessed_data: bool = False,
) -> BytesIO:
    targets_payload = json.dumps(
        {data_name: sample_name for data_name, sample_name in targets}
    )
    payload: dict[str, Any] = {
        "targets": targets_payload,
        "params": params_payload,
    }
    if include_preprocessed_data:
        payload["preprocessed_data"] = _serialize_preprocessed_data(preprocessed_data)
    for name, df in image_results.items():
        payload[f"image_analysis_results/{name}"] = _serialize_dataframe_pickle(df)
    return create_tar_from_dict(payload)


def _split_evenly_in_order[T](items: list[T], chunks: int) -> list[list[T]]:
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    if not items:
        return []

    chunk_count = min(chunks, len(items))
    base, extra = divmod(len(items), chunk_count)

    result: list[list[T]] = []
    start = 0
    for idx in range(chunk_count):
        size = base + (1 if idx < extra else 0)
        end = start + size
        result.append(items[start:end])
        start = end
    return result


def _infer_failed_target(
    err_msg: str, targets: list[tuple[str, str]]
) -> tuple[str, str]:
    match = re.search(r"ターゲット\s+([^\s]+)\s+\(([^)]+)\)", err_msg)
    if match:
        return match.group(1), match.group(2)
    return targets[0]


def _serialize_dataframe_pickle(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    pd.to_pickle(df, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    return buf


def _save_images_to_dir(images: dict[str, BytesIO], output_dir: Path) -> None:
    for name, buf in images.items():
        buf.seek(0)
        _save_image_bytes_to_dir(name, buf.read(), output_dir)


def _save_image_bytes_to_dir(name: str, image_bytes: bytes, output_dir: Path) -> None:
    path = output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)


def _run_entrypoint_with_tar(
    entrypoint: Path,
    tar_buf: BytesIO,
    mode: str,
) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["ANALYSISRUN_MODE"] = mode
    return subprocess.run(
        [sys.executable, str(entrypoint)],
        input=tar_buf.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


@dataclass
class _AnalyzeSeqStreamingResult:
    returncode: int
    stderr: str
    analysis_results: dict[str, BytesIO]
    error: Optional[str]
    tar_read_ok: bool


def _run_entrypoint_with_tar_streaming(
    entrypoint: Path,
    tar_buf: BytesIO,
    mode: str,
    on_image: Callable[[str, str, bytes, Optional[str]], None],
) -> _AnalyzeSeqStreamingResult:
    env = os.environ.copy()
    env["ANALYSISRUN_MODE"] = mode

    proc = subprocess.Popen(
        [sys.executable, str(entrypoint)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    stderr_chunks = bytearray()

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        stderr_chunks.extend(proc.stderr.read())

    stderr_reader = Thread(target=_drain_stderr, daemon=True)
    stderr_reader.start()

    if proc.stdin is not None:
        try:
            proc.stdin.write(tar_buf.getvalue())
            proc.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()

    analysis_results: dict[str, BytesIO] = {}
    error_message: Optional[str] = None
    tar_read_ok = True

    if proc.stdout is not None:
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    file_obj = tar.extractfile(member)
                    if file_obj is None:
                        continue
                    content = file_obj.read()
                    if member.name == "error":
                        error_message = content.decode("utf-8").strip()
                        continue

                    parts = member.name.split("/")
                    if len(parts) == 2 and parts[1] == "analysis_result":
                        analysis_results[parts[0]] = BytesIO(content)
                        continue
                    if len(parts) >= 3 and parts[1] == "images":
                        image_name = "/".join(parts[2:])
                        image_type = member.pax_headers.get("image_type")
                        on_image(parts[0], image_name, content, image_type)
        except Exception:
            tar_read_ok = False
        finally:
            proc.stdout.close()

    returncode = proc.wait()
    stderr_reader.join()
    stderr_text = stderr_chunks.decode("utf-8", errors="ignore")

    return _AnalyzeSeqStreamingResult(
        returncode=returncode,
        stderr=stderr_text,
        analysis_results=analysis_results,
        error=error_message,
        tar_read_ok=tar_read_ok,
    )


def _try_read_tar_from_bytes(data: bytes) -> dict[str, Any] | None:
    try:
        return read_tar_as_dict(BytesIO(data))
    except Exception:
        return None


def _add_tar_binary_entry(
    tar: tarfile.TarFile,
    name: str,
    content: bytes,
    *,
    image_type: Optional[str] = None,
) -> None:
    tar_info = tarfile.TarInfo(name=name)
    tar_info.size = len(content)
    pax_headers = {"is_file": "true"}
    if image_type is not None:
        pax_headers["image_type"] = image_type
    tar_info.pax_headers = pax_headers
    tar.addfile(tar_info, BytesIO(content))


def _write_parallel_streaming_result_entries(
    tar: tarfile.TarFile,
    result_df: pd.DataFrame,
) -> None:
    csv_buf = BytesIO()
    result_df = result_df.astype(str)
    result_df.to_csv(csv_buf, index=False)
    _add_tar_binary_entry(tar, "result_csv", csv_buf.getvalue())

    for idx, row in result_df.iterrows():
        data_name = row["data"] if "data" in row else idx
        json_bytes = row.to_json().encode("utf-8")
        _add_tar_binary_entry(tar, f"{str(data_name)}/result_json", json_bytes)


def _write_parallel_chunk_errors(
    stderr: IO[bytes],
    errors: list[tuple[str, str, str, str]],
) -> None:
    for data_name, sample_name, err_msg, err_out in errors:
        header = f"\033[1;31m=====> * {data_name} ({sample_name}) ====================>\033[0m"
        footer = f"\033[1;31m<==================== {data_name} ({sample_name}) * <=====\033[0m"
        stderr.write(
            "\n".join(
                [header, err_msg.strip(), "", err_out.strip(), footer, "", ""]
            ).encode("utf-8")
        )
    stderr.flush()
