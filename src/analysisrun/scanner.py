from typing import Generator, List, Callable

import pandas as pd


Filter = Callable[[pd.DataFrame], pd.Series]


class Views:
    """
    レーンのデータを視野ごとにスキャンする
    """

    def __init__(
        self,
        name: str,
        image_analysis_method: str,
        data: pd.DataFrame,
        viewpoints: List[int],
        skip_empty_viewpoints: bool = False,
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
        viewpoints
            スキャン対象となる視野番号のリスト
        skip_empty_viewpoints
            データのない視野をスキップするかどうか
        """

        self.data_name = name
        self.image_analysis_method = image_analysis_method
        self.data = data
        self.viewpoints = viewpoints
        self.__skip_empty_viewpoints = skip_empty_viewpoints
        return

    def skip_empty_viewpoints(self):
        """
        データのない視野をスキップするスキャナーを作成する
        """
        return Views(
            name=self.data_name,
            image_analysis_method=self.image_analysis_method,
            data=self.data,
            viewpoints=self.viewpoints,
            skip_empty_viewpoints=True,
        )

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        return (
            d
            for v in self.viewpoints
            if len(d := self.data[self.data.MultiPointIndex == v]) > 0
            or not self.__skip_empty_viewpoints
        )


class Lanes:
    """
    データ全体をレーンごとにスキャンする
    """

    def __init__(
        self, whole_data: pd.DataFrame, target_data: List[str], viewpoints: List[int]
    ) -> None:
        """
        データ全体をレーンごとにスキャンする

        Parameters
        ----------
        whole_data
            データ全体
        target_data
            対象データ名のリスト
        viewpoints
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
        self.viewpoints = viewpoints
        return

    def __iter__(self):
        for name in self.target_data:
            data = self.whole_data[self.whole_data["Data"] == name]

            image_analysis_method = (
                "" if len(data) == 0 else data.iloc[0, :].loc["ImageAnalysisMethod"]
            )

            yield Views(
                name=name,
                image_analysis_method=image_analysis_method,
                data=data,
                viewpoints=self.viewpoints,
            )
