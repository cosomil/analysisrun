from typing import List, Optional

import pandas as pd

from .cleansing import CleansedData


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

        data = whole_data._data
        # 既に必要な派生列があれば再代入せずそのまま使う
        if {"ImageAnalysisMethod", "Data"}.issubset(data.columns):
            self.whole_data = data
        # 入力データを直接変更せず、派生列を持つDataFrameを作る
        elif data.empty:
            self.whole_data = data.assign(ImageAnalysisMethod="", Data="")
        else:
            split_data = data["Filename"].str.split("_000_", n=1, expand=True)
            self.whole_data = data.assign(
                ImageAnalysisMethod=split_data[0],
                Data=split_data[1].str.split(".", n=1).str[0],
            )
        self.target_data = target_data
        self.field_numbers = field_numbers
        return

    def __iter__(self):
        return (
            Fields(
                name=name,
                data=(data := self.whole_data[self.whole_data["Data"] == name]),
                image_analysis_method=""
                if len(data) == 0
                else data.iloc[0, :].loc["ImageAnalysisMethod"],
                field_numbers=self.field_numbers,
            )
            for name in self.target_data
        )


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
        field_numbers=field_numbers or [i + 1 for i in range(12)],
    )
