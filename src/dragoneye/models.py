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


class BboxObservation(BaseModel):
    """A bounding box and the confidence of the detection that produced it.

    Both fields are always present: a ``BboxObservation`` only ever exists where
    the model actually placed a box. Absence — a video gap frame where the
    tracked object was present but not detected — is represented by a ``None``
    ``observation`` on :class:`VideoBboxObservation`, never by a
    ``BboxObservation`` with null fields.
    """

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


class ImageAttributePrediction(_AttributePredictionBase):
    """A chosen attribute option for an object in an image, with its score."""

    score: float


class ImageCategoryPrediction(_CategoryPredictionBase):
    attributes: List[ImageAttributePrediction]


class ImageDetectedObject(BaseModel):
    """One detected object in an image: its bounding box and categories.

    Unlike :class:`VideoDetectedObject`, an image object has no time dimension:
    a single :class:`BboxObservation` and a single ``score`` per attribute.
    """

    object_id: int
    bbox_observation: BboxObservation
    categories: List[ImageCategoryPrediction]


# ---- Video (time-aware) ----


class VideoBboxObservation(BaseModel):
    """One sighting of a tracked object at a single timestamp.

    Observations span every processed frame within the track's lifespan. A
    detected frame carries a real :class:`BboxObservation`; a "predicted-but-
    undetected" gap frame — where the object was present but not detected —
    carries ``observation=None``. ``timestamp_microseconds`` is always present,
    even on a gap frame.
    """

    timestamp_microseconds: int
    observation: Optional[BboxObservation]


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
