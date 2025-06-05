from typing import List

import pandas as pd


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
        self, whole_data: pd.DataFrame, target_data: List[str], field_numbers: List[int]
    ) -> None:
        """
        データ全体をレーンごとにスキャンする

        Parameters
        ----------
        whole_data
            データ全体
        target_data
            対象データ名のリスト
        field_numbers
            スキャン対象となる視野番号のリスト
        """

        whole_data["Data"] = whole_data["Filename"].apply(
            lambda x: str(x).split("_000_")[1].split(".")[0]
        )
        whole_data["ImageAnalysisMethod"] = whole_data["Filename"].apply(
            lambda x: str(x).split("_000_")[0]
        )
        whole_data = whole_data[whole_data["Entity"] == "Activity Spots"]
        self.whole_data = whole_data
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
