import os
import subprocess
import sys
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import (
    IO,
    Annotated,
    Any,
    BinaryIO,
    Callable,
    Generator,
    Iterable,
    Literal,
    LiteralString,
    NamedTuple,
    Optional,
    TextIO,
    Type,
)

import pandas as pd
from matplotlib import figure as fig
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, create_model

from analysisrun.__env import get_entrypoint, get_interactivity
from analysisrun.__pipeable_io import (
    EXIT_CODE_INVALID_USAGE,
    EXIT_CODE_PROCESSING_ERROR,
    Cleansing,  # noqa: F401
    ErrorResult,
    ImageAnalysisResult,  # noqa: F401
    ParallelAnalysisOutput,
    create_parallel_analysis_input,
    create_parallel_analysis_output,
    exit_with_error,
    extract_image_analysis_results,
    read_parallel_analysis_input,
    read_parallel_analysis_output,
)
from analysisrun.__tar import read_tar_as_dict
from analysisrun.__typing import NamedTupleLike
from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile, scan_model_input
from analysisrun.output import DefaultOutput, Output
from analysisrun.runner import AnalyzeArgs, PostprocessArgs
from analysisrun.scanner import Fields, Lanes

ENV_METHOD = "ANALYSISRUN_METHOD"
ENV_IMAGE_OUTPUT_DESTINATION = "ANALYSISRUN_IMAGE_OUTPUT_DESTINATION"


class Input[
    Parameters: BaseModel | None,
    ImageAnalysisResultInput: BaseModel,
](BaseModel):
    """
    解析の入力を格納するモデル。
    """

    image_analysis_results: ImageAnalysisResultInput
    sample_names: VirtualFile
    parameters: Parameters

    model_config = {
        "arbitrary_types_allowed": True,
    }


@dataclass
class Analyzer[
    Parameters: BaseModel | None,
    ImageAnalysisResults: NamedTupleLike[Fields],
](metaclass=ABCMeta):
    """
    各レーンの数値解析を行うAnalyzerの基底クラス。
    このクラスを継承することで数値解析メソッドを実装します。
    """

    params: Parameters
    """
    数値解析のパラメータ
    """
    sample_name: str
    """
    対象となるレーンのデータを視野ごとに探索するためのスキャナー
    """
    image_analysis_results: ImageAnalysisResults
    """
    各データを別の観点から解析し、補強するためのスキャナーのリスト
    """
    output: Output
    """
    画像を保存するためのOutput実装
    """

    @abstractmethod
    def analyze(self) -> pd.Series:
        pass


class _PostprocessInput[Parameters: BaseModel | None](BaseModel):
    analysis_results: list[VirtualFile] = Field(description="解析結果リスト")
    parameters: Parameters


@dataclass
class Postprocessor[
    Parameters: BaseModel | None,
](metaclass=ABCMeta):
    """
    すべてのレーンの解析結果を総合し、後処理（評価、調整など）を行うPostprocessorの基底クラス。
    後処理が必要な場合のみこのクラスを継承して実装する。
    """

    params: Parameters
    """
    数値解析のパラメータ
    """
    analysis_results: pd.DataFrame
    """
    解析結果を格納したDataFrame。
    """

    @abstractmethod
    def postprocess(self) -> pd.DataFrame:
        pass


class InMemoryOutput(Output):
    """
    解析で生成される画像をメモリ上に保存するOutput実装。
    メモリに蓄積した画像データは、tar形式でシリアライズして外部に出力される。
    """

    def __init__(self, store: dict[str, BytesIO]):
        self._store = store

    def __call__(
        self,
        fig: fig.Figure,
        name: str,
        image_type: LiteralString,
        **kwargs: Any,
    ) -> None:
        buf = BytesIO()
        fig.savefig(buf, **kwargs)
        fig.clear()
        plt.close(fig)
        buf.seek(0)
        self._store[name] = buf


def run_analysis[
    Parameters: BaseModel | None,
    ImageAnalysisResultInput: NamedTupleLike[VirtualFile],
    ImageAnalysisResults: NamedTupleLike[Fields],
](
    parameters_type: Type[Parameters],
    image_analysis_result_input_type: Type[ImageAnalysisResultInput],
    image_analysis_results_type: Type[ImageAnalysisResults],
    analyzer: Type[Analyzer[Parameters, ImageAnalysisResults]],
    postprocessor: Optional[Type[Postprocessor[Parameters]]] = None,
    manual_input: Optional[Input[Parameters, ImageAnalysisResultInput]] = None,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    # 解析する視野の番号
    field_numbers = [i + 1 for i in range(12)]

    local_input = manual_input
    parallel = False
    interactivity = get_interactivity()
    match interactivity:
        case "notebook":
            if local_input is None:
                raise exit_with_error(
                    EXIT_CODE_INVALID_USAGE,
                    "notebookで実行する場合、manual_inputを指定してください。",
                )
        case "terminal":
            if local_input is None:
                # ユーザーからインタラクティブに入力を受け取る。
                local_input = scan_model_input(
                    Input[Parameters, image_analysis_result_input_type]
                )
            parallel = True

    # ローカルで直列/並列実行する
    if local_input and interactivity:
        if output_dir is None:
            # 数値解析結果ファイルの置き場所を作業ディレクトリとする。
            output_dir = str(local_input.image_analysis_results[0].parent)
        output_dir = Path(output_dir).absolute()

        sample_names = read_dict(local_input.sample_names, "data", "sample")

        lanes = _make_lanes(
            local_input.image_analysis_results,
            field_numbers,
            sample_names,
        )

        results: list[pd.Series] = []
        if not parallel:
            # 直列実行
            output = DefaultOutput(
                show=interactivity == "notebook", parent_dir=output_dir
            )

            def analyze(
                fields: tuple[Fields, ...],
                parameters: Parameters,
                sample_names: dict[str, str],
            ):
                f = fields[0]
                try:
                    a = analyzer(
                        params=parameters,
                        sample_name=sample_names.get(f.data_name, f.data_name),
                        image_analysis_results=image_analysis_results_type(*(fields)),
                        output=output,
                    )
                    return a.analyze()
                except Exception as e:
                    raise exit_with_error(
                        EXIT_CODE_INVALID_USAGE,
                        f"'{f.data_name}'の解析中にエラーが発生しました。",
                        e,
                    )

            results = [
                analyze(fields, local_input.parameters, sample_names)
                for fields in _zip_unpacked(lanes)
            ]
        else:
            # 並列実行
            entrypoint = get_entrypoint()
            assert entrypoint is not None, (
                "エントリポイントの取得に失敗したため実行できません"
            )
            with ThreadPoolExecutor() as executor:
                outputs = executor.map(
                    _run_parallel_analysis,
                    [
                        _RunParallelAnalysisArgs(
                            sys.executable,
                            entrypoint,
                            str(output_dir),
                            fields,
                            local_input.parameters,
                            sample_names,
                        )
                        for fields in _zip_unpacked(lanes)
                    ],
                )
                for output in outputs:
                    data_name = output[0]
                    return_code = output[1]
                    result = output[2]
                    match result:
                        case ParallelAnalysisOutput():
                            results.append(result.analysis_result)
                        case ErrorResult():
                            raise exit_with_error(
                                return_code,
                                f"数値解析の実行中にエラーが発生しました[{data_name}]\n{result.error}",
                            )

        if postprocessor is not None:
            try:
                p = postprocessor(
                    params=local_input.parameters,
                    analysis_results=pd.DataFrame(results),
                )
                return p.postprocess()
            except Exception as e:
                raise exit_with_error(
                    EXIT_CODE_INVALID_USAGE,
                    "解析結果のpostprocess処理中にエラーが発生しました。",
                    e,
                )
        return pd.DataFrame(results)

    # 標準入力から入力を受け取る。
    # 以下のパターンを想定する。
    # - 並列処理: analyze
    #     ローカルでの並列処理時に、tarを利用して別プロセスで解析を実行するパターン。
    #     標準入力からレーンのデータを受け取り、解析結果を標準出力する。
    #     ただし、各画像は標準出力に流すのではなく直接画像を保存する。
    # - 分散処理: analyze
    #     分散環境での処理時に、tarを利用して特定のレーンの解析を実行するパターン。
    #     画像と解析結果のすべてを標準出力に書き出す。
    # - 分散処理: postprocess
    #     分散環境での処理時に、tarを利用して解析結果の後処理を実行するパターン。
    #     解析結果のすべてを標準入力から受け取り、後処理結果を標準出力に書き出す。

    # 処理メソッドの判定(analyze/postprocess)
    method = os.environ.get(ENV_METHOD)
    match method:
        case "analyze":
            # 画像出力方法の判定(標準出力/直接保存)
            output_images: dict[str, BytesIO] = {}
            output_destination = os.environ.get(ENV_IMAGE_OUTPUT_DESTINATION)
            match output_destination:
                case None:
                    raise exit_with_error(
                        EXIT_CODE_INVALID_USAGE,
                        f"{ENV_IMAGE_OUTPUT_DESTINATION} not set",
                    )
                case "stdout":
                    # 分散処理: analyze
                    output = InMemoryOutput(store=output_images)
                case _:
                    # 並列処理: analyze
                    # 画像の保存先となるディレクトリ
                    _dir = Path(output_destination)
                    if not _dir.is_dir():
                        raise exit_with_error(
                            EXIT_CODE_INVALID_USAGE,
                            f"{ENV_IMAGE_OUTPUT_DESTINATION} is not a directory",
                        )
                    output = DefaultOutput(parent_dir=_dir)

            try:
                _in = read_parallel_analysis_input(
                    sys.stdin.buffer,
                    parameters_type,
                    image_analysis_result_input_type,
                    field_numbers,
                )
            except Exception as e:
                raise exit_with_error(
                    EXIT_CODE_PROCESSING_ERROR,
                    "画像解析結果の読み込みに失敗しました。",
                    e,
                )
            try:
                fields = next(_zip_unpacked(_in.image_analysis_results))
                a = analyzer(
                    params=_in.params,
                    sample_name=_in.sample_name,
                    image_analysis_results=image_analysis_results_type(*(fields)),
                    output=output,
                )
                analysis_result = a.analyze()
            except Exception as e:
                raise exit_with_error(
                    EXIT_CODE_PROCESSING_ERROR,
                    repr(e),
                    e,
                )

            _out = create_parallel_analysis_output(analysis_result, output_images)
            sys.stdout.buffer.write(_out.getvalue())
            sys.stdout.buffer.flush()

        case "postprocess":
            # 分散処理: postprocess

            # FIXME: エラーの原因となるフィールドの出力
            _in = _PostprocessInput[Parameters](**read_tar_as_dict(sys.stdin.buffer))

            pass
        case None:
            raise exit_with_error(
                EXIT_CODE_INVALID_USAGE,
                f"{ENV_METHOD} not set",
            )
        case _:
            raise exit_with_error(
                EXIT_CODE_INVALID_USAGE,
                f"invalid value found in {ENV_METHOD}",
            )

    # DataFrameを返さずに直接終了する
    sys.exit(0)


def _zip_unpacked[T](
    data: tuple[Iterable[T], ...],
) -> Generator[tuple[T, ...], None, None]:
    return (li for li in zip(*data))


@dataclass
class _RunParallelAnalysisArgs:
    executable: str
    entrypoint: Path
    output_destination: str
    fields: tuple[Fields, ...]
    parameters: Optional[BaseModel]
    sample_names: dict[str, str]


def _run_parallel_analysis(
    args: _RunParallelAnalysisArgs,
):
    data_name = args.fields[0].data_name
    print(f"数値解析を開始[{data_name}]")
    input_data = create_parallel_analysis_input(
        args.parameters,
        args.sample_names,
        args.fields,
    )

    proc = subprocess.Popen(
        [args.executable, args.entrypoint],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        shell=False,
        env={
            ENV_METHOD: "analyze",
            ENV_IMAGE_OUTPUT_DESTINATION: args.output_destination,
        },
    )
    out, err = proc.communicate(input_data.getvalue())
    print(err)

    output = read_parallel_analysis_output(BytesIO(out))

    # FIXME: エラーハンドリング
    #  エラーメッセージとしては最初に発生したものだけ表示し、すべてのスタックトレースはその前にすべて表示する

    return (data_name, proc.returncode, output)


def _make_lanes(
    image_analysis_results: NamedTupleLike[VirtualFile],
    field_numbers: list[int],
    sample_names: dict[str, str],
):
    try:
        return tuple(
            Lanes(
                target_data=list(sample_names),
                whole_data=data,
                field_numbers=field_numbers,
            )
            for data in extract_image_analysis_results(
                image_analysis_results, pd.read_csv
            )
        )
    except Exception as e:
        raise exit_with_error(
            EXIT_CODE_INVALID_USAGE,
            "画像解析結果の読み込みに失敗しました。",
            e,
        )


@dataclass
class Context[
    Parameters: BaseModel | None,
    ImageAnalysisResults: NamedTupleLike[Fields],
]:
    class _SequentialAnalysisInput:
        targets: dict[str, str]
        params: Parameters

    class _ParallelAnalysisInput:
        targets: dict[str, str]
        params: Parameters

    class _DistributedPostprocessInput:
        params: Parameters

    input: (
        _SequentialAnalysisInput | _ParallelAnalysisInput | _DistributedPostprocessInput
    )

    def run(
        self,
        analyze: Callable[[AnalyzeArgs[Parameters]], pd.Series],
        postprocess: Optional[
            Callable[[PostprocessArgs[Parameters]], pd.DataFrame]
        ] = None,
    ) -> pd.DataFrame:
        # FIXME: ここの条件分岐を整理して、Contextに何が揃っていなければならないのか検討する
        match self.input:
            case Context._SequentialAnalysisInput() as analysis_input:
                pass
            case Context._ParallelAnalysisInput() as analysis_input:
                pass
            case Context._DistributedPostprocessInput() as postprocess_input:
                pass

        pass


def _get_distributed_method():
    match os.getenv(ENV_METHOD):
        case "analyze":
            return "distributed-analyze"
        case "postprocess":
            return "distributed-postprocess"
        case None:
            return None
        case _:
            raise exit_with_error(
                EXIT_CODE_INVALID_USAGE,
                f"invalid value found in {ENV_METHOD}",
            )


class ManualInput[
    Parameters: BaseModel | None,
]:
    """
    解析の入力を格納するモデル。
    """

    image_analysis_results: dict[str, VirtualFile]
    sample_names: VirtualFile
    parameters: Parameters


def _create_image_analysis_result_input_model(base: Type[NamedTupleLike[Fields]]):
    # FIXME: cleansingをどう設定するのか？
    # FIXME: データの流れをまず概観する
    fields: dict[str, Any] = {
        f: (VirtualFile, Annotated[VirtualFile, Field(description="画像解析結果")])
        for f in base._fields
    }
    return create_model(
        "ImageAnalysisResultInput",
        **fields,
    )


def load_context[
    Parameters: BaseModel,
    ImageAnalysisResults: NamedTupleLike[Fields],
](
    parameters: Type[Parameters],
    image_analysis_results: Type[ImageAnalysisResults],
    manual_input: Optional[ManualInput[Parameters]] = None,
    stdin: BinaryIO = sys.stdin.buffer,
    stdout: BinaryIO = sys.stdout.buffer,
) -> Context[Parameters, ImageAnalysisResults]:
    # 解析する視野の番号
    field_numbers = [i + 1 for i in range(12)]

    distributed_method = _get_distributed_method()
    interactivity = get_interactivity()

    # 入力の読み取りと実行モードの決定。
    ImageAnalysisResultInput = _create_image_analysis_result_input_model(
        image_analysis_results
    )
    In = Input[parameters, ImageAnalysisResultInput]
    # FIXME: エラーハンドリング

    _input = In(**manual_input.__dict__) if manual_input is not None else None
    mode: Optional[
        Literal[
            "sequential", "parallel", "distributed-analyze", "distributed-postprocess"
        ]
    ] = None
    match distributed_method:
        case "distributed-analyze":
            try:
                _in = read_parallel_analysis_input(
                    sys.stdin.buffer,
                    parameters,
                    ImageAnalysisResultInput,
                    field_numbers,
                )
            except Exception as e:
                raise exit_with_error(
                    EXIT_CODE_PROCESSING_ERROR,
                    "画像解析結果の読み込みに失敗しました。",
                    e,
                )
            mode = "distributed-analyze"

        case "distributed-postprocess":
            assert _input is not None
            mode = "distributed-postprocess"
            # FIXME: ここどうなるんだ…？

        case None:
            match interactivity:
                case "terminal":
                    if _input is None:
                        # ユーザーからインタラクティブに入力を受け取る。
                        ImageAnalysisResultInput = (
                            _create_image_analysis_result_input_model(
                                image_analysis_results
                            )
                        )
                        _input = scan_model_input(In)
                    mode = "parallel"
                case "notebook" | None:
                    if _input is None:
                        raise exit_with_error(
                            EXIT_CODE_INVALID_USAGE,
                            "manual_inputを指定してください。",
                        )
                    mode = "sequential" if interactivity == "notebook" else "parallel"

            # FIXME: エラーハンドリング
            target_samples = read_dict(_input.sample_names, "data", "sample")

    return Context[Parameters, ImageAnalysisResults](
        mode=mode, params=_input.parameters, targets=target_samples
    )
