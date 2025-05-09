from typing import Optional, Generator, Any, List, Callable

import pandas as pd


Filter = Callable[[pd.DataFrame], pd.Series]


class LaneDataScanner:
    """
    レーンのデータを視野ごとにスキャンする
    """

    def __init__(
        self,
        name: str,
        image_analysis_method: str,
        data: pd.DataFrame,
        viewpoints: List[int],
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
        """

        self.name = name
        self.image_analysis_method = image_analysis_method
        self.data = data
        self.viewpoints = viewpoints
        return

    def each_viewpoint(
        self, filter: Optional[Filter] = None, skip_empty_viewpoints: bool = False
    ) -> Generator[pd.DataFrame, Any, None]:
        """
        視野ごとのデータを抽出するジェネレータ

        Parameters
        ----------
        filter
            フィルタ条件
        skip_empty_viewpoints
            データのない視野をスキップするかどうか
        """
        target_data = self.data if filter is None else self.data[filter(self.data)]

        for view in self.viewpoints:
            d = target_data[target_data.MultiPointIndex == view]
            if len(d) > 0 or not skip_empty_viewpoints:
                yield d


class Scanner:
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

    def each_lane(self, filter: Optional[Filter] = None):
        """
        各レーンのデータを読み込むLaneDataScannerを生成するジェネレータ

        Parameters
        ----------
        filter
            フィルタ条件
        """
        for name in self.target_data:
            data = self.whole_data[self.whole_data["Data"] == name]

            target_data = data if filter is None else data[filter(data)]

            image_analysis_method = (
                "" if len(data) == 0 else data.iloc[0, :].loc["ImageAnalysisMethod"]
            )

            yield LaneDataScanner(
                name=name,
                image_analysis_method=image_analysis_method,
                data=target_data,
                viewpoints=self.viewpoints,
            )
