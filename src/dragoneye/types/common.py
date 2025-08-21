from enum import Enum
from typing import Literal, NewType, Optional, Sequence, Tuple

from pydantic import BaseModel

PredictionType = Literal["image", "video"]
PredictionTaskState = NewType("PredictionTaskState", str)

NormalizedBbox = NewType("NormalizedBbox", Tuple[float, float, float, float])


class TaxonType(str, Enum):
    CATEGORY = ("category",)
    TRAIT = ("trait",)


TaxonID = NewType("TaxonID", int)

PredictionTaskUUID = NewType("PredictionTaskUUID", str)


class TaxonPrediction(BaseModel):
    id: TaxonID
    type: TaxonType
    name: str
    displayName: str
    score: Optional[float]
    children: Sequence["TaxonPrediction"]


BASE_API_URL = "https://api.dragoneye.ai"
