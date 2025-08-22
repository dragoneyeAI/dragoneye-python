from typing import Sequence

from pydantic import BaseModel

from dragoneye.types.common import (
    NormalizedBbox,
    PredictionTaskState,
    PredictionTaskUUID,
    PredictionType,
    TaxonID,
    TaxonPrediction,
)


class PredictionTaskStatusResponse(BaseModel):
    prediction_task_uuid: PredictionTaskUUID
    prediction_type: PredictionType
    status: PredictionTaskState


class ClassificationTraitRootPrediction(BaseModel):
    id: TaxonID
    name: str
    displayName: str
    taxons: Sequence[TaxonPrediction]


class ClassificationObjectPrediction(BaseModel):
    normalizedBbox: NormalizedBbox
    category: TaxonPrediction
    traits: Sequence[ClassificationTraitRootPrediction]


class ClassificationPredictImageResponse(BaseModel):
    predictions: Sequence[ClassificationObjectPrediction]


class ClassificationVideoObjectPrediction(ClassificationObjectPrediction):
    frame_id: str
    frame_index: int
    timestamp_microseconds: int


class ClassificationPredictVideoResponse(BaseModel):
    timestamp_to_predictions: dict[float, Sequence[ClassificationVideoObjectPrediction]]
    frames_per_second: int
