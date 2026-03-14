import os
from typing import NamedTuple

from pydantic import BaseModel

import analysisrun as ar


class Params(BaseModel):
    threshold: float = 0.8


class ImageAnalysisResults(NamedTuple):
    activity_spots: ar.Fields = ar.image_analysis_result_spec(
        description="Activity spots",
        cleansing=ar.entity_filter("Activity Spots"),
    )


os.environ["ANALYSISRUN_MODE"] = "showschema"

ar.read_context(Params, ImageAnalysisResults)
