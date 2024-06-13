from enum import Enum
from typing import NewType, Optional, Sequence, Tuple

from pydantic import BaseModel

NormalizedBbox = NewType("NormalizedBbox", Tuple[float, float, float, float])


class TaxonType(str, Enum):
    CATEGORY = ("category",)
    TRAIT = ("trait",)


TaxonID = NewType("TaxonID", int)


class TaxonPrediction(BaseModel):
    id: TaxonID
    type: TaxonType
    name: str
    displayName: str
    score: Optional[float]
    children: Sequence["TaxonPrediction"]


BASE_API_URL = "https://api.dragoneye.ai"
