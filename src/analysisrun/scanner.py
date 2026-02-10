from typing import List

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
        データのない視野をスキップするスキャナーを作成する
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
    データ全体をレーンごとにスキャンする
    """

    def __init__(
        self,
        whole_data: CleansedData,
        target_data: List[str],
        field_numbers: List[int],
    ) -> None:
        """
        データ全体をレーンごとにスキャンする

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
        split_data = data["Filename"].str.split("_000_", expand=True)
        # データが空の場合、以下の列の値を""にする
        if len(data) == 0:
            data["ImageAnalysisMethod"] = ""
            data["Data"] = ""
        else:
            data["ImageAnalysisMethod"] = split_data[0]
            data["Data"] = split_data[1].str.split(".", expand=True)[0]

        self.whole_data = data
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
