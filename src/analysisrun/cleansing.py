"""
数値解析対象として意図されていないデータの混入を防ぐためのクレンジング処理と、クレンジング済みであることを表すデータ型を提供します。
"""

import pandas as pd
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class CleansedData:
    """
    データクレンジング処理後の解析対象データ
    """

    _data: pd.DataFrame


def filter_by_entity(
    data: pd.DataFrame | CleansedData,
    entity: str | Iterable[str] = "Activity Spots",
) -> CleansedData:
    """
    指定されたエンティティ名でデータをフィルタリングする。

    Parameters
    ----------
    data
        解析対象データ
    entity
        数値解析の対象となるエンティティ名(Entity列)
    """

    target_data = data if isinstance(data, pd.DataFrame) else data._data
    if isinstance(entity, str):
        target_data = target_data[target_data["Entity"] == entity]
    else:
        target_data = target_data[target_data["Entity"].isin(entity)]
    return CleansedData(_data=target_data)
