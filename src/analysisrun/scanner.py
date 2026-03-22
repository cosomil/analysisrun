from typing import List, Optional

import pandas as pd

from .cleansing import CleansedData


def _raise_missing_columns(context: str, missing_columns: set[str]) -> None:
    raise ValueError(f"{context}: missing {sorted(missing_columns)}")


def _default_field_numbers(field_numbers: Optional[List[int]]) -> List[int]:
    return field_numbers or [i + 1 for i in range(12)]


def _normalize_lane_source_data(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = {"MultiPointIndex"} - set(data.columns)
    if missing_columns:
        _raise_missing_columns("Lanes requires MultiPointIndex column", missing_columns)

    if {"ImageAnalysisMethod", "Data"}.issubset(data.columns):
        return data

    if data.empty:
        return data.assign(ImageAnalysisMethod="", Data="")

    missing_columns = {"Filename"} - set(data.columns)
    if missing_columns:
        _raise_missing_columns(
            "Lanes requires Filename when ImageAnalysisMethod/Data are absent",
            missing_columns,
        )

    split_data = data["Filename"].str.split("_000_", n=1, expand=True)
    return data.assign(
        ImageAnalysisMethod=split_data[0],
        Data=split_data[1].str.split(".", n=1).str[0],
    )


def _build_fields(name: str, data: pd.DataFrame, field_numbers: List[int]) -> "Fields":
    missing_columns = {"MultiPointIndex"} - set(data.columns)
    if missing_columns:
        _raise_missing_columns("Fields requires MultiPointIndex column", missing_columns)

    if data.empty:
        image_analysis_method = ""
    else:
        methods = data["ImageAnalysisMethod"].dropna().astype(str).unique()
        if len(methods) != 1 or methods[0] == "":
            raise ValueError(
                "Fields requires a unique non-empty image_analysis_method "
                f"for data_name={name!r}"
            )
        image_analysis_method = methods[0]

    return Fields(
        name=name,
        image_analysis_method=image_analysis_method,
        data=data,
        field_numbers=field_numbers,
    )


class Fields:
    """
    レーンのデータを視野ごとにスキャンする
    """

    def __init__(
        self,
        name: str,
        image_analysis_method: str,
        data: pd.DataFrame,
        field_numbers: List[int],
        skip_empty_fields: bool = False,
    ) -> None:
        """
        レーンのデータを視野ごとにスキャンする

        Parameters
        ----------
        name
            データ名
        image_analysis_method
            画像解析メソッド
        data
            対象データ
        field_numbers
            スキャン対象となる視野番号のリスト
        skip_empty_fields
            データのない視野をスキップするかどうか
        """

        missing_columns = {"MultiPointIndex"} - set(data.columns)
        if missing_columns:
            _raise_missing_columns(
                "Fields requires MultiPointIndex column", missing_columns
            )

        self.data_name = name
        self.image_analysis_method = image_analysis_method
        self.data = data
        self.field_numbers = field_numbers
        self.__skip_empty_fields = skip_empty_fields
        return

    def skip_empty_fields(self):
        """
        データのない視野をスキップするスキャナーを作成する。

        `field_numbers` は維持したまま、該当データがある視野だけを
        走査したいときに使う。
        """
        return Fields(
            name=self.data_name,
            image_analysis_method=self.image_analysis_method,
            data=self.data,
            field_numbers=self.field_numbers,
            skip_empty_fields=True,
        )

    def __iter__(self):
        return (
            d
            for v in self.field_numbers
            if len(d := self.data[self.data.MultiPointIndex == v]) > 0
            or not self.__skip_empty_fields
        )


class Lanes:
    """
    データ全体をレーンごとにスキャンするヘルパー。

    `Filename` 列から `ImageAnalysisMethod` と `Data` を導出する。
    `ImageAnalysisMethod` 列と `Data` 列が既に存在する場合は、それらを再計算せず既存値をそのまま使う。

    Examples
    --------
    >>> lanes = Lanes(
    ...     whole_data=data,
    ...     target_data=["data1", "data2"],
    ...     field_numbers=[1, 2],
    ... )
    >>> [lane.data_name for lane in lanes]
    ['data1', 'data2']
    """

    def __init__(
        self,
        whole_data: CleansedData,
        target_data: List[str],
        field_numbers: List[int],
    ) -> None:
        """
        データ全体をレーンごとにスキャンする。

        Parameters
        ----------
        whole_data
            解析対象データ
        target_data
            対象データ名のリスト
        field_numbers
            スキャン対象となる視野番号のリスト
        """

        self.whole_data = _normalize_lane_source_data(whole_data._data)
        self.target_data = target_data
        self.field_numbers = field_numbers
        self._target_data_set = set(target_data)
        self._empty_lane_data = self.whole_data.iloc[0:0]
        # Data列から一度だけ索引を構築し、レーンごとの再フィルタリングを避ける
        self._lane_data_by_name = {
            name: lane_data
            for name, lane_data in self.whole_data.groupby("Data", sort=False)
            if name in self._target_data_set
        }
        return

    def get(self, data_name: str) -> Fields:
        if data_name not in self._target_data_set:
            raise ValueError(f"{data_name} not found in lanes")

        data = self._lane_data_by_name.get(data_name, self._empty_lane_data)
        return _build_fields(data_name, data, self.field_numbers)

    def __iter__(self):
        return (self.get(name) for name in self.target_data)


def scan(
    whole_data: pd.DataFrame,
    target_data: List[str],
    field_numbers: Optional[List[int]] = None,
) -> Lanes:
    """
    データ全体をレーンごとにスキャンする。

    `Lanes` を直接組み立てなくても、手元の `DataFrame` を
    レーン単位・視野単位で順に処理したいときに使える。

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "Filename": [
    ...             "method1_000_data1.csv",
    ...             "method1_000_data1.csv",
    ...             "method2_000_data2.csv",
    ...         ],
    ...         "MultiPointIndex": [1, 2, 1],
    ...         "Value": [10, 20, 30],
    ...     }
    ... )
    >>> for lane in scan(df, target_data=["data1", "data2"], field_numbers=[1, 2]):
    ...     print(lane.data_name, lane.image_analysis_method)
    ...     for field in lane.skip_empty_fields():
    ...         print(field["Value"].sum())
    data1 method1
    10
    20
    data2 method2
    30

    Parameters
    ----------
    whole_data
        解析対象データ
    target_data
        対象データ名のリスト
    field_numbers
        スキャン対象となる視野番号のリスト（指定しない場合は1から12までの視野が対象）
    """

    return Lanes(
        whole_data=CleansedData(_data=whole_data),
        target_data=target_data,
        field_numbers=_default_field_numbers(field_numbers),
    )


def scan_fields(
    data: pd.DataFrame,
    data_name: str,
    field_numbers: Optional[List[int]] = None,
) -> Fields:
    """
    正規化済みDataFrameから単一レーンのFieldsを復元する。

    Parameters
    ----------
    data
        正規化済みDataFrame。`Data` 列、`ImageAnalysisMethod` 列、
        `MultiPointIndex` 列を含む必要がある。
    data_name
        復元対象のデータ名。
    field_numbers
        スキャン対象となる視野番号のリスト（指定しない場合は1から12までの視野が対象）
    """

    required_columns = {"Data", "ImageAnalysisMethod", "MultiPointIndex"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "scan_fields requires a normalized DataFrame with Data, "
            "ImageAnalysisMethod, and MultiPointIndex columns: "
            f"missing {sorted(missing_columns)}"
        )

    lane_data = data[data["Data"] == data_name]
    return _build_fields(data_name, lane_data, _default_field_numbers(field_numbers))
