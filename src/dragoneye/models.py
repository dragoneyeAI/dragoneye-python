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

    Used by video responses to describe when an object was visible.
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


class _BboxObservationBase(BaseModel):
    """Fields shared by every bounding-box sighting, image or video."""

    normalized_bbox: NormalizedBbox
    bbox_score: float


class _AttributePredictionBase(BaseModel):
    """The identity of a chosen attribute option, shared by image and video."""

    attribute_id: int
    attribute_name: str
    option_id: int
    option_name: str


class _CategoryPredictionBase(BaseModel):
    """Fields shared by image and video category predictions."""

    category_id: int
    name: str
    score: float


# ---- Image (timestamp-free) ----


class ImageBboxObservation(_BboxObservationBase):
    """The bounding box of a detected object in an image."""


class ImageAttributePrediction(_AttributePredictionBase):
    """A chosen attribute option for an object in an image, with its score."""

    score: float


class ImageCategoryPrediction(_CategoryPredictionBase):
    attributes: List[ImageAttributePrediction]


class ImageDetectedObject(BaseModel):
    """One detected object in an image: its bounding box and categories.

    Unlike :class:`VideoDetectedObject`, an image object has no time dimension:
    a single :class:`ImageBboxObservation` and a single ``score`` per attribute.
    """

    object_id: int
    bbox_observation: ImageBboxObservation
    categories: List[ImageCategoryPrediction]


# ---- Video (time-aware) ----


class VideoBboxObservation(_BboxObservationBase):
    """One bounding-box sighting of a tracked object at a single timestamp."""

    timestamp_microseconds: int


class VideoAttributePrediction(_AttributePredictionBase):
    """A chosen attribute option together with the time runs over which it won.

    The same ``attribute_id`` can appear multiple times across an object's life
    with different options; each entry carries the scored timestamp ranges over
    which its option held.
    """

    timestamp_ranges: List[ScoredTimestampRange]


class VideoCategoryPrediction(_CategoryPredictionBase):
    attributes: List[VideoAttributePrediction]


class VideoDetectedObject(BaseModel):
    """One tracked entity: its lifespan, every bbox observation, and categories.

    The server returns one parquet row per ``VideoDetectedObject``; the nesting
    already exists in the schema, so deserialization is a straight structural
    map.
    """

    object_id: int
    timestamp_ranges: List[TimestampRange]
    bbox_observations: List[VideoBboxObservation]
    categories: List[VideoCategoryPrediction]


class ClassificationPredictImageResponse(BaseModel):
    objects: List[ImageDetectedObject]
    prediction_task_uuid: PredictionTaskUUID
    original_file_name: Optional[str]


class ClassificationPredictVideoResponse(BaseModel):
    objects: List[VideoDetectedObject]
    frames_per_second: int
    prediction_task_uuid: PredictionTaskUUID
    original_file_name: Optional[str]
