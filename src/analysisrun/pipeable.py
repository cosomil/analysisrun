import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Literal,
    LiteralString,
    Optional,
    Protocol,
    Type,
)

import matplotlib.figure as fig
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, create_model

from analysisrun.__env import get_entrypoint, get_interactivity
from analysisrun.__typing import NamedTupleLike, VirtualFileLike
from analysisrun.cleansing import CleansedData, filter_by_entity
from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile, scan_model_input
from analysisrun.pipeable_io import (
    AnalysisInputModel,
    ExitCodes,
    PostprocessInputModel,
    exit_with_error,
    list_from_dict,
    redirect_stdout_to_stderr,
)
from analysisrun.scanner import Fields, Lanes
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
    ImageAnalysisResults: NamedTupleLike[Fields],
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


def create_image_analysis_results_input_model[
    ImageAnalysisResults: NamedTupleLike[Fields],
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
class _BaseState:
    stdout: IO[bytes]
    stderr: IO[bytes]


@dataclass
class _AnalysisState[ParamsT: BaseModel, ImageInputModelT: BaseModel](_BaseState):
    parsed_input: AnalysisInputModel[ParamsT, ImageInputModelT]
    specs: dict[str, _ImageAnalysisResultSpec]
    field_numbers: list[int]


@dataclass
class _PostprocessState[ParamsT: BaseModel](_BaseState):
    parsed_input: PostprocessInputModel[ParamsT]


@dataclass
class _SequentialState(_BaseState):
    cleansed_data: dict[str, CleansedData]
    sample_pairs: list[tuple[str, str]]
    field_numbers: list[int]


@dataclass
class _ParallelState(_BaseState):
    raw_data: dict[str, pd.DataFrame]
    sample_pairs: list[tuple[str, str]]
    output_dir: Path
    entrypoint: Path
    field_numbers: list[int]


@dataclass
class AnalysisContext[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[Fields],
]:
    """
    数値解析のコンテキスト。
    解析の入力、実行モードなどを保持し、モードに応じて解析の実行を行う。
    """

    params: Params
    image_analysis_results: Type[ImageAnalysisResults]
    output: Output
    state: (
        _AnalysisState[Params, BaseModel]
        | _PostprocessState[Params]
        | _SequentialState
        | _ParallelState
    )

    @property
    def mode(
        self,
    ) -> Literal[
        "analysis-only", "postprocess-only", "sequential", "parallel-entrypoint"
    ]:
        match self.state:
            case _AnalysisState():
                return "analysis-only"
            case _PostprocessState():
                return "postprocess-only"
            case _SequentialState():
                return "sequential"
            case _ParallelState():
                return "parallel-entrypoint"

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
        """
        stdout = self.state.stdout
        stderr = self.state.stderr

        try:
            match self.state:
                case _AnalysisState():
                    return self._run_analysis_only(analyze, stdout, stderr)
                case _PostprocessState():
                    return self._run_postprocess_only(postprocess, stdout, stderr)
                case _SequentialState():
                    return self._run_local_sequential(analyze, postprocess)
                case _ParallelState():
                    return self._run_local_parallel(postprocess)
        except SystemExit:
            raise
        except Exception as exc:
            message = (
                "後処理でエラーが発生しました。入力データやパラメータを確認してください。"
                if isinstance(self.state, _PostprocessState)
                else "解析処理中にエラーが発生しました。入力データやパラメータを確認してください。"
            )
            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                message,
                stdout,
                stderr,
                exc,
            )

    def _run_analysis_only(
        self,
        analyze: Callable[[AnalyzeArgs[Params, ImageAnalysisResults]], pd.Series],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _AnalysisState)
        state = self.state
        parsed = state.parsed_input
        specs = state.specs
        field_numbers = state.field_numbers

        output = _TarCollectingOutput()
        try:
            cleansed_data = _load_and_cleanse_image_results(
                parsed.image_analysis_results,
                specs,
            )
            lanes = _build_fields_namedtuple(
                self.image_analysis_results,
                cleansed_data,
                parsed.data_name,
                field_numbers,
            )
            with redirect_stdout_to_stderr(stderr):
                series = analyze(AnalyzeArgs(parsed.params, lanes, output))
            _ensure_result_annotations(series, parsed.data_name, parsed.sample_name)
        except Exception as exc:
            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                "解析処理中にエラーが発生しました。",
                stdout,
                stderr,
                exc,
            )

        tar_data = {"analysis_result": _serialize_series(series)}
        tar_data.update({f"images/{name}": buf for name, buf in output.images.items()})
        tar_buf = create_tar_from_dict(tar_data)
        stdout.write(tar_buf.getvalue())
        stdout.flush()
        sys.exit(0)

    def _run_postprocess_only(
        self,
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]],
        stdout: IO[bytes],
        stderr: IO[bytes],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _PostprocessState)
        state = self.state
        parsed = state.parsed_input

        analysis_results_inputs = list_from_dict(parsed.analysis_results)
        analysis_results: list[pd.DataFrame] = []
        for analyisis_results_input in analysis_results_inputs:
            b = analyisis_results_input.unwrap()
            assert isinstance(b, BytesIO)
            analysis_results.append(_deserialize_dataframe_with_leading_zeroes(b))

        concatenated_analysis_results = pd.concat(analysis_results, ignore_index=True)

        try:
            with redirect_stdout_to_stderr(stderr):
                result_df = (
                    postprocess(
                        PostprocessArgs(parsed.params, concatenated_analysis_results)
                    )
                    if postprocess
                    else concatenated_analysis_results
                )
            if result_df is None:
                result_df = concatenated_analysis_results
        except Exception as exc:
            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                "後処理でエラーが発生しました。",
                stdout,
                stderr,
                exc,
            )

        tar_buf = create_tar_from_dict(_build_postprocess_tar_entries(result_df))
        stdout.write(tar_buf.getvalue())
        stdout.flush()
        sys.exit(0)

    def _run_local_sequential(
        self,
        analyze: Callable[[AnalyzeArgs[Params, ImageAnalysisResults]], pd.Series],
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _SequentialState)
        state = self.state
        cleansed_data = state.cleansed_data
        sample_pairs = state.sample_pairs
        field_numbers = state.field_numbers

        results: list[pd.Series] = []
        for data_name, sample_name in sample_pairs:
            lanes = _build_fields_namedtuple(
                self.image_analysis_results,
                cleansed_data,
                data_name,
                field_numbers,
            )
            series = analyze(AnalyzeArgs(self.params, lanes, self.output))
            _ensure_result_annotations(series, data_name, sample_name)
            results.append(series)

        result_df = pd.DataFrame(results)
        if postprocess:
            postprocessed = postprocess(PostprocessArgs(self.params, result_df))
            if postprocessed is not None:
                result_df = postprocessed
        return result_df

    def _run_local_parallel(
        self,
        postprocess: Optional[Callable[[PostprocessArgs[Params]], pd.DataFrame]],
    ) -> pd.DataFrame:
        assert isinstance(self.state, _ParallelState)
        state = self.state
        raw_data = state.raw_data
        sample_pairs = state.sample_pairs
        output_dir = state.output_dir
        entrypoint = state.entrypoint

        params_payload = self.params.model_dump_json()
        output_dir.mkdir(parents=True, exist_ok=True)

        def _build_tar_buffer(data_name: str, sample_name: str) -> BytesIO:
            payload: dict[str, Any] = {
                "data_name": data_name,
                "sample_name": sample_name,
                "params": params_payload,
            }
            for name, df in raw_data.items():
                payload[f"image_analysis_results/{name}"] = _serialize_dataframe(df)
            return create_tar_from_dict(payload)

        def _run_lane(args: tuple[str, str]) -> pd.Series:
            data_name, sample_name = args
            tar_buf = _build_tar_buffer(data_name, sample_name)
            env = os.environ.copy()
            env["ANALYSISRUN_METHOD"] = "analyze"
            proc = subprocess.run(
                [sys.executable, str(entrypoint)],
                input=tar_buf.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            if proc.returncode != 0:
                err_msg = proc.stderr.decode(errors="ignore")
                try:
                    error_tar = read_tar_as_dict(BytesIO(proc.stdout))
                    if isinstance(error_tar, dict) and "error" in error_tar:
                        err_msg = error_tar["error"]
                except Exception:
                    pass
                raise RuntimeError(
                    f"{data_name}の解析プロセスでエラーが発生しました: {err_msg}"
                )

            tar_result = read_tar_as_dict(BytesIO(proc.stdout))
            if "error" in tar_result:
                raise RuntimeError(
                    f"{data_name}の解析プロセスでエラーが発生しました: {tar_result['error']}"
                )

            series_buf = tar_result.get("analysis_result")
            if not isinstance(series_buf, BytesIO):
                raise RuntimeError(f"{data_name}の解析結果の形式が不正です。")
            series = _deserialize_series(series_buf)
            _ensure_result_annotations(series, data_name, sample_name)

            images = _extract_images_from_tar_dict(tar_result)
            _save_images_to_dir(images, output_dir)
            return series

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_run_lane, sample_pairs))

        result_df = pd.DataFrame(results)
        if postprocess:
            postprocessed = postprocess(PostprocessArgs(self.params, result_df))
            if postprocessed is not None:
                result_df = postprocessed
        return result_df


def read_context[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[Fields],
](
    params: Type[Params],
    image_analysis_results: Type[ImageAnalysisResults],
    manual_input: Optional[ManualInput[Params]] = None,
    stdin: Optional[IO[bytes]] = None,
    stdout: Optional[IO[bytes]] = None,
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
    _stdin: IO[bytes] = stdin or sys.stdin.buffer
    _stdout: IO[bytes] = stdout or sys.stdout.buffer
    _stderr: IO[bytes] = sys.stderr.buffer
    interactivity = get_interactivity()
    method = os.getenv("ANALYSISRUN_METHOD")

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
    LocalInputModel = InputModel[params, image_analysis_results_input_model]
    AnalysisInput = AnalysisInputModel[params, image_analysis_results_input_model]
    PostprocessInput = PostprocessInputModel[params]

    field_numbers = [i + 1 for i in range(12)]
    output_dir_path = Path(output_dir) if output_dir else None
    cleansed_data: dict[str, CleansedData] | None = None
    raw_data: dict[str, pd.DataFrame] | None = None
    sample_pairs: list[tuple[str, str]] | None = None
    params_value: Params | None = None
    output_impl: Output = _NullOutput()
    ctx_state: (
        _AnalysisState[Params, BaseModel]
        | _PostprocessState[Params]
        | _SequentialState
        | _ParallelState
    )

    if method == "analyze":
        try:
            tar_dict = read_tar_as_dict(_stdin)
            tar_dict["params"] = _maybe_load_json(tar_dict.get("params"))
            parsed = AnalysisInput(**tar_dict)
            params_value = parsed.params
        except Exception as exc:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "入力データの読み込みに失敗しました。入力形式を確認してください。",
                _stdout,
                _stderr,
                exc,
            )
        ctx_state = _AnalysisState(
            stdout=_stdout,
            stderr=_stderr,
            parsed_input=parsed,
            specs=specs,
            field_numbers=field_numbers,
        )
    elif method == "postprocess":
        try:
            tar_dict = read_tar_as_dict(_stdin)
            tar_dict["params"] = _maybe_load_json(tar_dict.get("params"))
            parsed = PostprocessInput(**tar_dict)
            params_value = parsed.params
        except Exception as exc:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "入力データの読み込みに失敗しました。入力形式を確認してください。",
                _stdout,
                _stderr,
                exc,
            )
        ctx_state = _PostprocessState(
            stdout=_stdout,
            stderr=_stderr,
            parsed_input=parsed,
        )
    else:
        if interactivity is None:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "ANALYSISRUN_METHOD環境変数に実行モードが指定されていません。",
                _stdout,
                _stderr,
            )

        mode = "sequential" if interactivity == "notebook" else "parallel-entrypoint"
        if mode == "sequential" and manual_input is None:
            exit_with_error(
                ExitCodes.INVALID_USAGE,
                "Jupyter notebook環境ではmanual_inputの指定が必須です。",
                _stdout,
                _stderr,
            )

        if manual_input is not None:
            try:
                iar_input = image_analysis_results_input_model(
                    **manual_input.image_analysis_results
                )
                local_input = LocalInputModel(
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
            local_input = scan_model_input(LocalInputModel)

        params_value = local_input.params
        if output_dir_path is None:
            output_dir_path = _derive_output_dir(local_input.image_analysis_results)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_impl = _FileOutput(output_dir_path)

        raw_data = _load_image_results_raw(
            local_input.image_analysis_results,
        )
        if mode == "sequential":
            cleansed_data = {
                name: _apply_cleansing_pipeline(df, specs[name])
                for name, df in raw_data.items()
            }
        sample_pairs = [
            (data, sample)
            for data, sample in read_dict(
                local_input.sample_names.unwrap(),
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

        if mode == "sequential":
            assert cleansed_data is not None
            ctx_state = _SequentialState(
                stdout=_stdout,
                stderr=_stderr,
                cleansed_data=cleansed_data,
                sample_pairs=sample_pairs,
                field_numbers=field_numbers,
            )
        else:
            assert raw_data is not None and output_dir_path is not None
            entrypoint = get_entrypoint()
            if entrypoint is None:
                exit_with_error(
                    ExitCodes.INVALID_USAGE,
                    "エントリーポイントとなるスクリプトのパス取得に失敗したため、並列実行させることができません。",
                    _stdout,
                    _stderr,
                )
            ctx_state = _ParallelState(
                stdout=_stdout,
                stderr=_stderr,
                raw_data=raw_data,
                sample_pairs=sample_pairs,
                output_dir=output_dir_path,
                entrypoint=entrypoint,
                field_numbers=field_numbers,
            )

    ctx = AnalysisContext[Params, ImageAnalysisResults](
        params=params_value,
        image_analysis_results=image_analysis_results,
        output=output_impl,
        state=ctx_state,
    )
    return ctx


class _TarCollectingOutput(Output):
    def __init__(self):
        self.images: dict[str, BytesIO] = {}

    def __call__(self, fig, name: str, image_type: str, **kwargs) -> None:
        buf = FileIO({"image_type": image_type})
        fig.savefig(buf, **kwargs)
        buf.seek(0)
        self.images[name] = buf
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
) -> dict[str, pd.DataFrame]:
    """
    ImageAnalysisResultsInputから各データをDataFrameとして読み込む。

    クレンジングは行わず、生データを返す。
    """

    raw: dict[str, pd.DataFrame] = {}
    for name in type(image_analysis_results_model).model_fields:
        vfile = getattr(image_analysis_results_model, name)
        raw[name] = _deserialize_dataframe(vfile.unwrap())
    return raw


def _apply_cleansing_pipeline(
    data: pd.DataFrame | CleansedData, spec: _ImageAnalysisResultSpec
) -> CleansedData:
    cleansed: pd.DataFrame | CleansedData = data
    for fn in spec.cleansing:
        cleansed = fn(cleansed)

    if isinstance(cleansed, CleansedData):
        return cleansed
    if isinstance(cleansed, pd.DataFrame):
        return CleansedData(_data=cleansed)


def _load_and_cleanse_image_results(
    image_analysis_results_model: BaseModel,
    specs: dict[str, _ImageAnalysisResultSpec],
) -> dict[str, CleansedData]:
    cleansed_data: dict[str, CleansedData] = {}
    raw_data = _load_image_results_raw(
        image_analysis_results_model,
    )
    for name, spec in specs.items():
        df = raw_data[name]
        cleansed_data[name] = _apply_cleansing_pipeline(df, spec)
    return cleansed_data


def _build_fields_namedtuple[
    ImageAnalysisResults: NamedTupleLike[Fields],
](
    image_analysis_results_type: Type[ImageAnalysisResults],
    cleansed_data: dict[str, CleansedData],
    data_name: str,
    field_numbers: list[int],
) -> ImageAnalysisResults:
    lanes: dict[str, Fields] = {}
    for name, cleansed in cleansed_data.items():
        lane_iter = Lanes(
            whole_data=cleansed,
            target_data=[data_name],
            field_numbers=field_numbers,
        )
        lanes[name] = next(iter(lane_iter))
    return image_analysis_results_type(**lanes)


def _ensure_result_annotations(series: pd.Series, data_name: str, sample_name: str):
    if "data_name" not in series:
        series["data_name"] = data_name
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


def _deserialize_dataframe(value: Any) -> pd.DataFrame:
    """BytesIO/PathからDataFrameを復元する（CSV専用）。"""

    if isinstance(value, BytesIO):
        value.seek(0)
    return pd.read_csv(value, dtype=_CSV_DTYPE)


def _deserialize_dataframe_with_leading_zeroes(value: BytesIO) -> pd.DataFrame:
    """BytesIO/PathからDataFrameを復元する（CSV専用）。"""

    # まずはすべて文字列として一度読み込む
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


_CSV_DTYPE: dict[str, type] = {
    "Entity": str,
    "Filename": str,
    "data_name": str,
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


def _extract_images_from_tar_dict(tar_dict: dict[str, Any]) -> dict[str, BytesIO]:
    images_entry = tar_dict.get("images")
    if isinstance(images_entry, dict):
        return {
            name: value
            for name, value in images_entry.items()
            if isinstance(value, BytesIO)
        }
    return {}


def _save_images_to_dir(images: dict[str, BytesIO], output_dir: Path) -> None:
    for name, buf in images.items():
        buf.seek(0)
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.read())


def _build_postprocess_tar_entries(result_df: pd.DataFrame) -> dict[str, BytesIO]:
    csv_buf = BytesIO()
    result_df = result_df.astype(str)
    result_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    entries: dict[str, BytesIO] = {"result_csv": csv_buf}
    for idx, row in result_df.iterrows():
        data_name = row["data_name"] if "data_name" in row else idx
        json_buf = BytesIO()
        json_buf.write(row.to_json().encode("utf-8"))
        json_buf.seek(0)
        entries[f"result_json/{str(data_name)}"] = json_buf
    return entries
