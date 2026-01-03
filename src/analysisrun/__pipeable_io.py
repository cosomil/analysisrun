from io import BytesIO
import json
import sys
import traceback
from typing import IO, Annotated, Any, Callable, Optional, Type
from dataclasses import dataclass

import cowsay
import pandas as pd
from pydantic import BaseModel, Field

from analysisrun.__env import get_interactivity
from analysisrun.__tar import create_tar_from_dict, read_tar_as_dict
from analysisrun.__typing import NamedTupleLike
from analysisrun.cleansing import CleansedData, filter_by_entity
from analysisrun.interactive import VirtualFile
from analysisrun.scanner import Fields, Lanes

EXIT_CODE_PROCESSING_ERROR = 1
EXIT_CODE_INVALID_USAGE = 2


class ErrorResult(BaseModel):
    """
    標準出力を通じて返されるエラー情報。
    """

    error: str = Field(description="エラーメッセージ")


def exit_with_error(
    code: int, message: str, exception: Optional[Exception] = None
) -> SystemExit:
    """
    解析の異常終了を行う。
    標準出力にエラー情報を出力するほか、スタックトレースやユーザーフレンドリーな
    エラーメッセージを標準エラー出力に出力する。

    Parameters
    ----------
    code
        終了コード
    message
        エラーメッセージ
    exception
        発生した例外オブジェクト

    Returns
    -------
    SystemExit
        与えられた終了コードをもつ例外。これをraiseすることでプロセスを終了する。
    """
    interactivity = get_interactivity()
    if interactivity is None:
        out = create_tar_from_dict(ErrorResult(error=message).model_dump())
        sys.stdout.buffer.write(out.getvalue())
        sys.stdout.buffer.flush()
        if exception is not None:
            print(exception, file=sys.stderr)
            traceback.print_exception(exception, file=sys.stderr)
    else:
        if exception is not None:
            traceback.print_exception(exception, file=sys.stderr)
        print(cowsay.get_output_string("cow", message), file=sys.stderr)
    return SystemExit(code)


@dataclass
class Cleansing:
    """
    画像解析結果データのクレンジング設定。
    """

    entity: str

    def _apply(self, df: pd.DataFrame) -> CleansedData:
        return filter_by_entity(df, self.entity)


def ImageAnalysisResult(description: str, cleansing: Cleansing, **kwargs) -> Any:
    """
    画像解析結果入力用のアノテーションを作成する。
    PydanticのFieldアノテーションのラッパー。

    Parameters
    ----------
    description
        フィールドの説明
    cleansing
        クレンジング設定
    **kwargs
        Fieldに渡す追加のキーワード引数
    """
    return Annotated[
        VirtualFile,
        Field(description=description, **kwargs),
        cleansing,
    ]


def _extract_cleansing(a: Any) -> Cleansing:
    if isinstance(a, tuple):
        if len(a) > 1:
            v = a[1]
            if isinstance(v, Cleansing):
                return v
    raise ValueError("failed to extract Cleansing.")


def _extract_virtualfile(v: Any) -> VirtualFile:
    if isinstance(v, VirtualFile):
        return v
    if isinstance(v, BytesIO):  # tar形式で受け取った入力はこのケースに該当する
        return VirtualFile(v)  # type: ignore
    raise ValueError("failed to extract VirtualFile.")


def extract_image_analysis_results(
    v: NamedTupleLike[VirtualFile], read: Callable[[VirtualFile], pd.DataFrame]
):
    """
    入力として受け取った画像解析結果を読み込み、さらにクレンジングを施して返す。
    """
    d = v._asdict()
    try:
        filtered = tuple(
            (
                _extract_cleansing(v._field_defaults[f].__metadata__),
                _extract_virtualfile(d[f]),
            )
            for f in v._fields
        )
    except Exception as e:
        raise exit_with_error(
            EXIT_CODE_INVALID_USAGE,
            "image_analysis_result_input_typeの属性にはVirtualFile型を使用し、さらにデフォルト値にImageAnalysisResultを使用してください。",
            e,
        )

    try:
        return tuple(f[0]._apply(read(f[1])) for f in filtered)
    except Exception as e:
        raise exit_with_error(
            EXIT_CODE_INVALID_USAGE,
            "画像解析結果の読み込みに失敗しました。",
            e,
        )

@dataclass
class _ImageAnalysisResult:
    data: pd.DataFrame
    cleansing: Cleansing


def extract_image_analysis_results2(model: Type[BaseModel], v: BaseModel) -> dict[str, _ImageAnalysisResult]:
    """
    入力として受け取った画像解析結果を読み込み、さらにクレンジングを施して返す。
    """
    d = v.model_dump()
    results = {}
    for f in model.model_fields:
        try:
            cleansing = _extract_cleansing(
                model.model_fields[f].default.__metadata__
            )
            virtualfile = _extract_virtualfile(d[f])
        except Exception as e:
            raise exit_with_error(
                EXIT_CODE_INVALID_USAGE,
                "image_analysis_result_input_typeの属性にはVirtualFile型を使用し、さらにデフォルト値にImageAnalysisResultを使用してください。",
                e,
            )

        try:
            data = _read_pickle_from_virtualfile(virtualfile)
            results[f] = _ImageAnalysisResult(data=data, cleansing=cleansing)
        except Exception as e:
            raise exit_with_error(
                EXIT_CODE_INVALID_USAGE,
                "画像解析結果の読み込みに失敗しました。",
                e,
            )
    return results


def create_parallel_analysis_input(
    params: Optional[BaseModel],
    sample_names: dict[str, str],
    fields: dict[str, Fields],
):
    """
    数値解析プロセスに渡すための入力を作成するユーティリティ。
    """
    data_name = next(iter(fields.values())).data_name
    input_data: dict[str, Any] = {
        "params": params.model_dump_json() if params is not None else None,
        "data_name": data_name,
        "sample_name": sample_names.get(data_name, data_name),
    }
    for key, f in fields.items():
        # DataFrameをBytesIOに変換する。
        # tarのエントリには"is_file"ヘッダーが付与される。
        data = BytesIO()
        f.data.to_pickle(data)
        input_data[f"image_analysis_results.{key}"] = data
    return create_tar_from_dict(input_data)




@dataclass
class _RawParallelAnalysisInput[Parameters: BaseModel | None]:
    params: Parameters
    sample_name: str
    image_analysis_results: tuple[Lanes, ...]


@dataclass
class _RawParallelAnalysisInput2[Parameters: BaseModel | None]:
    params: Parameters
    sample_name: str
    image_analysis_results: dict[str, _ImageAnalysisResult]


def _read_pickle_from_virtualfile(v: VirtualFile) -> pd.DataFrame:
    """
    tarからデシリアライズしたVirtualFile(実際にはBytesIO)から、画像解析結果のDataFrameを読み込む。
    """
    return pd.read_pickle(BytesIO(v.read()))


def read_parallel_analysis_input[
    Parameters: BaseModel | None,
    ImageAnalysisResultInput: BaseModel,
](
    _in: IO[bytes],
    parameters_type: Type[Parameters],
    image_analysis_result_input_type: Type[ImageAnalysisResultInput],
    field_numbers: list[int],
):
    """
    tar形式でシリアライズされた数値解析の入力データを読み取り、
    さらにクレンジング処理まで施して返す。
    """
    params = None
    data_name = None
    sample_name = None
    image_analysis_results = {}
    for name, value in read_tar_as_dict(_in).items():
        match name:
            case "params":
                assert isinstance(value, str), "params must be string"
                params = json.loads(value)
            case "data_name":
                assert isinstance(value, str), "data_name must be string"
                data_name = value
            case "sample_name":
                assert isinstance(value, str), "sample_name must be string"
                sample_name = value
            case "image_analysis_results":
                assert isinstance(value, dict), "image_analysis_results must be dict"
                image_analysis_results = value

    if parameters_type is not type(None):
        assert params is not None, "params is required in the input."
        params = parameters_type(**params)
    else:
        params = None

    assert data_name is not None, "data_name is required in the input."
    assert sample_name is not None, "sample_name is required in the input."

    assert len(image_analysis_results) > 0, (
        "No image analysis results found in the output."
    )
    image_analysis_result_input = image_analysis_result_input_type(*(image_analysis_results))
    results = extract_image_analysis_results(
        image_analysis_result_input, _read_pickle_from_virtualfile
    )

    return _RawParallelAnalysisInput2[Parameters](
        params=params,  # type: ignore
        sample_name=sample_name,
        image_analysis_results=tuple(
            Lanes(whole_data=data, target_data=[data_name], field_numbers=field_numbers)
            for data in results
        ),
    )


def create_parallel_analysis_output(
    analysis_result: pd.Series, images: dict[str, BytesIO] = {}
) -> BytesIO:
    """
    数値解析結果をtar形式にシリアライズする。
    """
    analysis_result_data = BytesIO()
    analysis_result.to_pickle(analysis_result_data)

    output_data = {
        "analysis_result": analysis_result_data,
        "images": images,
    }
    return create_tar_from_dict(output_data)


class _RawParallelAnalysisOutput(BaseModel):
    analysis_result: BytesIO = Field(description="解析結果")
    images: dict[str, BytesIO] = Field(
        default={},
        description="生成された画像データの辞書。キーがファイル名、値がバイト列",
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }


@dataclass
class ParallelAnalysisOutput:
    """
    1レーン分の画像解析結果。
    """

    analysis_result: pd.Series
    """解析結果"""
    images: dict[str, BytesIO]
    """生成された画像データの辞書。キーがファイル名、値がBytesIOオブジェクト"""


def read_parallel_analysis_output(_in: IO[bytes]):
    """
    tar形式でシリアライズされた数値解析の出力データを読み取る。
    """
    d = read_tar_as_dict(_in)
    try:
        raw = _RawParallelAnalysisOutput(**d)
        return ParallelAnalysisOutput(
            analysis_result=pd.read_pickle(raw.analysis_result),
            images=raw.images,
        )
    except Exception:
        try:
            return ErrorResult(**d)
        except Exception:
            return ErrorResult(error="Invalid output format.")
