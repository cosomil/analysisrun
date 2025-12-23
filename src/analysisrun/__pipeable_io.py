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

EXIT_CODE_INVALID_USAGE = 2


class ErrorResult(BaseModel):
    error: str = Field(description="エラーメッセージ")


def exit_with_error(
    code: int, message: str, exception: Optional[Exception] = None
) -> SystemExit:
    interactivity = get_interactivity()
    if interactivity is None:
        out = create_tar_from_dict(ErrorResult(error=message).model_dump())
        sys.stdout.buffer.write(out.getvalue())
        sys.stdout.buffer.flush()
        # FIXME: 例外の情報を標準エラー出力に流してもよいのでは？
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
    entity: str

    def _apply(self, df: pd.DataFrame) -> CleansedData:
        return filter_by_entity(df, self.entity)


def ImageAnalysisResult(description: str, cleansing: Cleansing, **kwargs) -> Any:
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


def create_parallel_analysis_input(
    params: Optional[BaseModel],
    sample_names: dict[str, str],
    fields: tuple[Fields, ...],
):
    data_name = fields[0].data_name
    input_data: dict[str, Any] = {
        "params": params.model_dump_json() if params is not None else None,
        "data_name": data_name,
        "sample_name": sample_names.get(data_name, data_name),
    }
    for idx, f in enumerate(fields):
        # DataFrameをBytesIOに変換する。
        # tarのエントリには"is_file"ヘッダーが付与される。
        data = BytesIO()
        f.data.to_pickle(data)
        input_data[f"image_analysis_results[{idx}]"] = data
    return create_tar_from_dict(input_data)


@dataclass
class RawParallelAnalysisInput[Parameters: BaseModel | None]:
    params: Parameters
    sample_name: str
    image_analysis_results: tuple[Lanes, ...]


def _read_pickle(v: VirtualFile):
    return pd.read_pickle(BytesIO(v.read()))


def read_parallel_analysis_input[Parameters: BaseModel | None](
    _in: IO[bytes],
    parameters_type: Type[Parameters],
    image_analysis_result_input_type: Type[NamedTupleLike[VirtualFile]],
    field_numbers: list[int],
):
    params = None
    data_name = None
    sample_name = None
    image_analysis_results = []
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
            case _ if name.startswith("image_analysis_results["):
                image_analysis_results.append(value)

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
    image_analysis_input = image_analysis_result_input_type(*(image_analysis_results))
    results = extract_image_analysis_results(image_analysis_input, _read_pickle)

    return RawParallelAnalysisInput[Parameters](
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
    analysis_result_data = BytesIO()
    analysis_result.to_pickle(analysis_result_data)

    output_data = {
        "analysis_result": analysis_result_data,
        "images": images,
    }
    return create_tar_from_dict(output_data)


class RawParallelAnalysisOutput(BaseModel):
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
    analysis_result: pd.Series
    """解析結果"""
    images: dict[str, BytesIO]
    """生成された画像データの辞書。キーがファイル名、値がBytesIOオブジェクト"""


def read_parallel_analysis_output(_in: IO[bytes]):
    d = read_tar_as_dict(_in)
    try:
        raw = RawParallelAnalysisOutput(**d)
        return ParallelAnalysisOutput(
            analysis_result=pd.read_pickle(raw.analysis_result),
            images=raw.images,
        )
    except Exception:
        try:
            return ErrorResult(**d)
        except Exception:
            return ErrorResult(error="Invalid output format.")
