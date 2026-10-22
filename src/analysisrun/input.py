"""External CSV/Params input protocol and cleansing."""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .cleansing import CleansedData, filter_by_entity
from .helper import read_dict
from .interactive import VirtualFile
from .scanner import scan
from .typing import NamedTupleLike, VirtualFileLike


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


CleansingFunc = Callable[[pd.DataFrame | CleansedData], CleansedData]
NonEmptyCleansingPipeline = tuple[CleansingFunc, *tuple[CleansingFunc, ...]]
CleansingPipeline = NonEmptyCleansingPipeline | list[CleansingFunc]


@dataclass
class _ImageAnalysisResultSpec:
    """
    画像解析データの説明とクレンジング処理を保持する。
    """

    description: str
    cleansing: NonEmptyCleansingPipeline


def image_analysis_result_spec(
    description: str,
    cleansing: CleansingFunc | CleansingPipeline,
) -> Any:
    """
    画像解析結果フィールドの仕様を定義する。

    """
    normalized_cleansing = (
        tuple(cleansing) if isinstance(cleansing, (tuple, list)) else (cleansing,)
    )
    if len(normalized_cleansing) == 0:
        raise ValueError("cleansing must contain at least one function")

    return _ImageAnalysisResultSpec(
        description=description,
        cleansing=normalized_cleansing,
    )


def entity_filter(
    entity: str | Iterable[str],
) -> Callable[[pd.DataFrame | CleansedData], CleansedData]:
    """
    既存のfilter_by_entityと同等のフィルタリング処理を返す。
    """

    return lambda data: filter_by_entity(data, entity=entity)


def dropna(column: str | Iterable[str]) -> CleansingFunc:
    """
    指定した列のいずれかに欠損値がある行を除去するクレンジング処理を返す。

    Parameters
    ----------
    column
        欠損値を確認する列名。複数指定も可能。
    """

    subset = (column,) if isinstance(column, str) else tuple(column)
    if not subset:
        raise ValueError("column must contain at least one column")

    def _dropna(data: pd.DataFrame | CleansedData) -> CleansedData:
        target_data = data if isinstance(data, pd.DataFrame) else data._data
        return CleansedData(_data=target_data.dropna(subset=list(subset)))

    return _dropna


def _get_image_analysis_specs[
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    image_analysis_results: type[ImageAnalysisResults],
) -> dict[str, _ImageAnalysisResultSpec]:
    """
    ImageAnalysisResultsのデフォルト値に定義されたImageAnalysisResultSpecを取得する。
    """

    field_defaults = image_analysis_results._field_defaults
    specs: dict[str, _ImageAnalysisResultSpec] = {}

    for name in image_analysis_results._fields:  # type: ignore[attr-defined]
        spec = field_defaults.get(name)
        if not isinstance(spec, _ImageAnalysisResultSpec):
            raise TypeError(
                f"{image_analysis_results.__name__}.{name} must have ImageAnalysisResultSpec as default"
            )
        specs[name] = spec

    return specs


def _build_streaming_input_schema[
    Params: BaseModel,
    ImageAnalysisResults: NamedTupleLike[pd.DataFrame],
](
    params: type[Params],
    image_analysis_results: type[ImageAnalysisResults],
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
](image_analysis_results: type[ImageAnalysisResults]) -> type[BaseModel]:
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


def _apply_cleansing_pipeline(
    data: pd.DataFrame, spec: _ImageAnalysisResultSpec
) -> pd.DataFrame:
    cleansed: pd.DataFrame | CleansedData = data
    for fn in spec.cleansing:
        cleansed = fn(cleansed)

    if isinstance(cleansed, CleansedData):
        return cleansed._data

    raise TypeError(
        f"cleansing function must return a CleansedData, got {type(cleansed)!r}"
    )


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


def _load_image_results_raw(
    image_analysis_results_model: BaseModel,
) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(
            getattr(image_analysis_results_model, name).unwrap(), dtype=_CSV_DTYPE
        )
        for name in type(image_analysis_results_model).model_fields
    }


def _prepare_inputs[I: NamedTupleLike[pd.DataFrame]](
    inputs_type: type[I],
    image_files: BaseModel,
    sample_names: VirtualFile,
    specs: dict[str, _ImageAnalysisResultSpec],
) -> tuple[I, dict[str, str]]:
    """Apply input cleansing and Filename normalization once, before user stages."""
    targets = read_dict(sample_names.unwrap(), key="data", value="sample")
    if not targets:
        raise ValueError("サンプル名CSVファイルが空です。")
    raw = _load_image_results_raw(image_files)
    prepared = {
        name: scan(
            _apply_cleansing_pipeline(frame, specs[name]), list(targets)
        ).whole_data
        for name, frame in raw.items()
    }
    return inputs_type(**prepared), targets
