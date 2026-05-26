from typing import List, Optional

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


class TimestampRange(BaseModel):
    """A contiguous span in microseconds, inclusive on both ends.

    For images, ``timestamp_start_us_inclusive == timestamp_end_us_inclusive == 0``.
    """

    timestamp_start_us_inclusive: int
    timestamp_end_us_inclusive: int


class ScoredTimestampRange(BaseModel):
    """A :class:`TimestampRange` carrying the confidence the option held over it.

    ``score`` is the mean of the option's raw per-frame scores over the range.
    """

    timestamp_start_us_inclusive: int
    timestamp_end_us_inclusive: int
    score: float


class BboxObservation(BaseModel):
    """One bounding-box sighting of a tracked object at a single timestamp.

    For images, ``timestamp_microseconds`` is ``0``.
    """

    timestamp_microseconds: int
    normalized_bbox: NormalizedBbox
    bbox_score: float


class AttributePrediction(BaseModel):
    """A chosen attribute option together with the time runs over which it won.

    The same ``attribute_id`` can appear multiple times across an object's life
    with different options; each entry carries the scored timestamp ranges over
    which its option held.
    """

    attribute_id: int
    attribute_name: str
    option_id: int
    option_name: str
    timestamp_ranges: List[ScoredTimestampRange]


class CategoryPrediction(BaseModel):
    category_id: int
    name: str
    score: float
    attributes: List[AttributePrediction]


class DetectedObject(BaseModel):
    """One tracked entity: its lifespan, every bbox observation, and categories.

    The server returns one parquet row per ``DetectedObject``; the nesting
    already exists in the schema, so deserialization is a straight structural
    map.
    """

    object_id: int
    timestamp_ranges: List[TimestampRange]
    bbox_observations: List[BboxObservation]
    categories: List[CategoryPrediction]


class ClassificationPredictImageResponse(BaseModel):
    objects: List[DetectedObject]
    prediction_task_uuid: PredictionTaskUUID
    original_file_name: Optional[str]


class ClassificationPredictVideoResponse(BaseModel):
    objects: List[DetectedObject]
    frames_per_second: int
    prediction_task_uuid: PredictionTaskUUID
    original_file_name: Optional[str]
