from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from analysisrun.__typing import NamedTupleLike
from analysisrun.runner import Output
from analysisrun.scanner import Fields


@dataclass
class AnalyzeArgs[
    Params: Optional[BaseModel],
    ImageAnalysisResults: NamedTupleLike[Fields],
]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    image_analysis_results: ImageAnalysisResults
    """
    解析対象となる画像解析結果のスキャナー
    """
    output: Output
    """
    画像を保存するためのOutput実装
    """


@dataclass
class PostprocessArgs[Params: Optional[BaseModel]]:
    params: Params
    """
    解析全体に関わるパラメータ
    """
    analysis_results: pd.DataFrame
    """
    各レーンの解析結果を格納したDataFrame
    """
