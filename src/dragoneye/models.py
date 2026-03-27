from typing import Dict, List, Optional

from pydantic import BaseModel

from dragoneye.types.common import (
    NormalizedBbox,
    PredictionTaskState,
    PredictionTaskUUID,
    PredictionType,
)


class PredictionTaskStatusResponse(BaseModel):
    prediction_task_uuid: PredictionTaskUUID
    prediction_type: PredictionType
    status: PredictionTaskState


class ClassificationAttributeOption(BaseModel):
    option_id: int
    name: str
    score: float


class ClassificationAttributeResponse(BaseModel):
    attribute_id: int
    name: str
    options: List[ClassificationAttributeOption]


class ClassificationCategory(BaseModel):
    id: int
    name: str
    score: float


class ClassificationCategoryPrediction(BaseModel):
    category: ClassificationCategory
    attributes: List[ClassificationAttributeResponse]


class ClassificationObjectPrediction(BaseModel):
    normalizedBbox: NormalizedBbox
    predictions: List[ClassificationCategoryPrediction]


class ClassificationPredictImageResponse(BaseModel):
    object_predictions: List[ClassificationObjectPrediction]
    prediction_task_uuid: PredictionTaskUUID
    original_file_name: Optional[str]


class ClassificationVideoObjectPrediction(ClassificationObjectPrediction):
    frame_id: str
    timestamp_microseconds: int


class ClassificationPredictVideoResponse(BaseModel):
    timestamp_us_to_predictions: Dict[
        int, List[ClassificationVideoObjectPrediction]
    ]
    frames_per_second: int
    prediction_task_uuid: PredictionTaskUUID
    original_file_name: Optional[str]
