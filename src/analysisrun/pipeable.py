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
        read_contextで読み込んだコンテキストに基づいて数値解析を実行する。

        - mode="analysis-only": 1レーンの解析だけを行い、結果をまとめたtarデータを標準出力に書き出す。その後exitする。
        - mode="postprocess-only": 各レーンの解析結果を結合し、後処理を行った結果をtarデータとして標準出力に書き出す。その後exitする。
        - mode="sequential": multiprocessingを活用できないJupyter notebook環境において、全レーンをシーケンシャルに処理し、DataFrameを返す。
        - mode="parallel-entrypoint": entrypointを並列起動し、各レーンのanalysis-onlyの結果を集約、後処理を加えてDataFrameを返す。

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
                "後処理中に不明なエラーが発生しました。開発者に確認してください。"
                if isinstance(self.state, _PostprocessState)
                else "解析処理中に不明なエラーが発生しました。開発者に確認してください。"
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

        # 分散/サブプロセス実行では、画像はファイル保存できないため tar 内に格納して返す（完全な分散実行と同様の挙動）。
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
                series = analyze(
                    AnalyzeArgs(
                        parsed.params,
                        parsed.data_name,
                        parsed.sample_name,
                        lanes,
                        output,
                    )
                )
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

        # postprocess は複数 lane の結果（CSV）を受け取り、まず結合してから任意の後処理を適用する。
        analysis_results_inputs = list_from_dict(parsed.analysis_results)
        analysis_results: list[pd.DataFrame] = []
        for analyisis_results_input in analysis_results_inputs:
            b = analyisis_results_input.unwrap()
            assert isinstance(b, BytesIO)
            # 先頭ゼロを含む値（例: "0012"）が落ちないように、列ごとに dtype を再推定する。
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
            series = analyze(
                AnalyzeArgs(self.params, data_name, sample_name, lanes, self.output)
            )
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

        # サブプロセスへの入力は、標準入力を経由してtarデータで渡される。そのため、paramsはJSONとしてシリアライズする。
        params_payload = self.params.model_dump_json()
        output_dir.mkdir(parents=True, exist_ok=True)

        # データ名の重複排除
        target_data: list[str] = []
        seen_data: set[str] = set()
        for data_name, _ in sample_pairs:
            if data_name not in seen_data:
                seen_data.add(data_name)
                target_data.append(data_name)

        # データ（レーン）ごとに画像解析データを分割する
        lane_data_by_dataset: dict[str, dict[str, pd.DataFrame]] = {}
        for name, df in raw_data.items():
            lane_scanner = Lanes(
                whole_data=CleansedData(_data=df.copy()),
                target_data=target_data,
                field_numbers=state.field_numbers,
            )
            lane_data_by_dataset[name] = {
                fields.data_name: fields.data.drop(
                    columns=["ImageAnalysisMethod", "Data"], errors="ignore"
                )
                for fields in lane_scanner
            }

        def _build_tar_buffer(data_name: str, sample_name: str) -> BytesIO:
            payload: dict[str, Any] = {
                "data_name": data_name,
                "sample_name": sample_name,
                "params": params_payload,
            }
            for name, per_lane in lane_data_by_dataset.items():
                lane_df = per_lane.get(data_name)
                if lane_df is None:
                    lane_df = raw_data[name].iloc[0:0]
                payload[f"image_analysis_results/{name}"] = _serialize_dataframe(
                    lane_df
                )
            return create_tar_from_dict(payload)

        def _run_lane(
            args: tuple[str, str],
        ) -> tuple[str, str, Optional[pd.Series], Optional[tuple[str, str]]]:
            data_name, sample_name = args
            tar_buf = _build_tar_buffer(data_name, sample_name)
            env = os.environ.copy()
            # entrypoint 側は ANALYSISRUN_MODE を見てanalysis-onlyモードで動作する。
            env["ANALYSISRUN_MODE"] = "analyze"
            proc = subprocess.run(
                [sys.executable, str(entrypoint)],
                input=tar_buf.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            tar_result: dict[str, Any] | None = None
            if proc.returncode != 0:
                err_msg = ""
                err_out = proc.stderr.decode(errors="ignore")
                if not err_msg:
                    err_msg = "不明なエラーが発生しました。"
                try:
                    # 子プロセスが `exit_with_error(...)` した場合、標準出力からエラーの内容を読み取ることができる。
                    error_tar = read_tar_as_dict(BytesIO(proc.stdout))
                    if isinstance(error_tar, dict):
                        tar_result = error_tar
                        if "error" in error_tar:
                            err_msg = error_tar["error"]
                except Exception:
                    pass
                if tar_result is not None:
                    images = _extract_images_from_tar_dict(tar_result)
                    _save_images_to_dir(images, output_dir)
                return data_name, sample_name, None, (err_msg, err_out)

            try:
                tar_result = read_tar_as_dict(BytesIO(proc.stdout))
            except Exception as exc:
                return (
                    data_name,
                    sample_name,
                    None,
                    (f"{data_name}の解析結果の読み込みに失敗しました", str(exc)),
                )

            if "error" in tar_result:
                images = _extract_images_from_tar_dict(tar_result)
                _save_images_to_dir(images, output_dir)
                return (
                    data_name,
                    sample_name,
                    None,
                    (str(tar_result["error"]), ""),
                )

            series_buf = tar_result.get("analysis_result")
            if not isinstance(series_buf, BytesIO):
                return (
                    data_name,
                    sample_name,
                    None,
                    (f"{data_name}の解析結果の形式が不正です。", ""),
                )
            series = _deserialize_series(series_buf)
            _ensure_result_annotations(series, data_name, sample_name)

            images = _extract_images_from_tar_dict(tar_result)
            _save_images_to_dir(images, output_dir)
            return data_name, sample_name, series, None

        with ThreadPoolExecutor() as executor:
            lane_results = list(executor.map(_run_lane, sample_pairs))

        errors: list[tuple[str, str, str, str]] = []
        results: list[pd.Series] = []
        for data_name, sample_name, series, err_msg in lane_results:
            if err_msg:
                errors.append((data_name, sample_name, err_msg[0], err_msg[1]))
            elif series is not None:
                results.append(series)

        if errors:
            lanes = []
            for data_name, sample_name, err_msg, err_out in errors:
                header = f"\033[1;31m=====> * {data_name} ({sample_name}) ====================>\033[0m"
                footer = f"\033[1;31m<==================== {data_name} ({sample_name}) * <=====\033[0m"
                state.stderr.write(
                    "\n".join(
                        [header, err_msg.strip(), "", err_out.strip(), footer, "", ""]
                    ).encode("utf-8")
                )
                lanes.append(f"- {data_name} ({sample_name})")
            state.stderr.flush()

            exit_with_error(
                ExitCodes.PROCESSING_ERROR,
                "\n".join(["解析中にエラーが発生しました。", *lanes]),
                state.stdout,
                state.stderr,
            )

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
    # 分散/サブプロセス実行では、呼び出し側が環境変数で実行フェーズを指定する。
    mode = os.getenv("ANALYSISRUN_MODE")

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

    # 画像解析結果は 1..12 lane を前提としている（現状は固定）。
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

    if mode == "analyze":
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
    elif mode == "postprocess":
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
                "ANALYSISRUN_MODE環境変数に実行モードが指定されていません。",
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

        # 分散実行でないあ場合Pythonオブジェクトとして入力を渡すことができるので、まずそれを優先して検証する。
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
            # CLI等ではインタラクティブな入力を通じてLocalInputModelを組み立てる。
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
