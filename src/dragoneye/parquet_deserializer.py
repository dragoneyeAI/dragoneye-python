"""Convert a zstd-compressed parquet prediction blob into typed SDK models.

This implements the **object-forward** schema: the server returns one parquet
row per tracked object. Everything about that object — its presence ranges,
every bbox observation over time, its categories, and each category's attribute
time-runs — is nested inside that single row. Images are encoded the same way as
video, just with all timestamps equal to ``0``.

The wire format is identical for both media types, but the SDK exposes two
different shapes. Video rows map straight to the time-aware
:class:`VideoDetectedObject`. Image rows are *collapsed* into the timestamp-free
:class:`ImageDetectedObject`: the single bbox observation loses its timestamp,
and each attribute's lone scored timestamp range becomes a bare ``score``.

    object_id:            Int64
    timestamp_ranges:     List(Struct{
                              timestamp_start_us_inclusive: Int64,
                              timestamp_end_us_inclusive:   Int64,
                          })
    bbox_observations:    List(Struct{
                              timestamp_microseconds: Int64,
                              normalized_bbox:        Array(Float32, 4),
                              bbox_score:             Float32,
                          })
    categories:           List(Struct{
                              category_id: Int64,
                              name:        String,
                              score:       Float32,
                              attributes:  List(Struct{
                                  attribute_id:     Int64,
                                  attribute_name:   String,
                                  option_id:        Int64,
                                  option_name:      String,
                                  timestamp_ranges: List(Struct{
                                      timestamp_start_us_inclusive: Int64,
                                      timestamp_end_us_inclusive:   Int64,
                                      score:                        Float32,
                                  }),
                              }),
                          })

There is no client-side grouping or flattening — the nesting already exists in
the parquet schema, so deserialization is a straight structural map from
parquet row to typed object.
"""

import io
from typing import Any, Dict, List

import polars as pl

from .models import (
    ImageAttributePrediction,
    ImageBboxObservation,
    ImageCategoryPrediction,
    ImageDetectedObject,
    ScoredTimestampRange,
    TimestampRange,
    VideoAttributePrediction,
    VideoBboxObservation,
    VideoCategoryPrediction,
    VideoDetectedObject,
)
from .types.common import NormalizedBbox


def _to_timestamp_range(value: Dict[str, Any]) -> TimestampRange:
    return TimestampRange(
        timestamp_start_us_inclusive=value["timestamp_start_us_inclusive"],
        timestamp_end_us_inclusive=value["timestamp_end_us_inclusive"],
    )


def _to_scored_timestamp_range(value: Dict[str, Any]) -> ScoredTimestampRange:
    return ScoredTimestampRange(
        timestamp_start_us_inclusive=value["timestamp_start_us_inclusive"],
        timestamp_end_us_inclusive=value["timestamp_end_us_inclusive"],
        score=value["score"],
    )


# ---- Video: a straight structural map from the parquet nesting ----


def _to_video_bbox_observation(value: Dict[str, Any]) -> VideoBboxObservation:
    return VideoBboxObservation(
        timestamp_microseconds=value["timestamp_microseconds"],
        normalized_bbox=NormalizedBbox(tuple(value["normalized_bbox"])),
        bbox_score=value["bbox_score"],
    )


def _to_video_attribute_prediction(value: Dict[str, Any]) -> VideoAttributePrediction:
    return VideoAttributePrediction(
        attribute_id=value["attribute_id"],
        attribute_name=value["attribute_name"],
        option_id=value["option_id"],
        option_name=value["option_name"],
        timestamp_ranges=[
            _to_scored_timestamp_range(tr)
            for tr in (value["timestamp_ranges"] or [])
        ],
    )


def _to_video_category_prediction(value: Dict[str, Any]) -> VideoCategoryPrediction:
    return VideoCategoryPrediction(
        category_id=value["category_id"],
        name=value["name"],
        score=value["score"],
        attributes=[
            _to_video_attribute_prediction(attr)
            for attr in (value["attributes"] or [])
        ],
    )


def _to_video_detected_object(row: Dict[str, Any]) -> VideoDetectedObject:
    return VideoDetectedObject(
        object_id=row["object_id"],
        timestamp_ranges=[
            _to_timestamp_range(tr) for tr in (row["timestamp_ranges"] or [])
        ],
        bbox_observations=[
            _to_video_bbox_observation(obs)
            for obs in (row["bbox_observations"] or [])
        ],
        categories=[
            _to_video_category_prediction(cat)
            for cat in (row["categories"] or [])
        ],
    )


# ---- Image: collapse the time dimension out of the same parquet rows ----


def _to_image_bbox_observation(value: Dict[str, Any]) -> ImageBboxObservation:
    return ImageBboxObservation(
        normalized_bbox=NormalizedBbox(tuple(value["normalized_bbox"])),
        bbox_score=value["bbox_score"],
    )


def _to_image_attribute_prediction(value: Dict[str, Any]) -> ImageAttributePrediction:
    # An image attribute has exactly one timestamp range.
    return ImageAttributePrediction(
        attribute_id=value["attribute_id"],
        attribute_name=value["attribute_name"],
        option_id=value["option_id"],
        option_name=value["option_name"],
        score=value["timestamp_ranges"][0]["score"],
    )


def _to_image_category_prediction(value: Dict[str, Any]) -> ImageCategoryPrediction:
    return ImageCategoryPrediction(
        category_id=value["category_id"],
        name=value["name"],
        score=value["score"],
        attributes=[
            _to_image_attribute_prediction(attr)
            for attr in (value["attributes"] or [])
        ],
    )


def _to_image_detected_object(row: Dict[str, Any]) -> ImageDetectedObject:
    # An image object always has exactly one bbox observation (at timestamp 0).
    return ImageDetectedObject(
        object_id=row["object_id"],
        bbox_observation=_to_image_bbox_observation(row["bbox_observations"][0]),
        categories=[
            _to_image_category_prediction(cat)
            for cat in (row["categories"] or [])
        ],
    )


def deserialize_object_forward_video_predictions(
    parquet_bytes: bytes,
) -> List[VideoDetectedObject]:
    """Map an object-forward parquet blob to a list of ``VideoDetectedObject``.

    One parquet row yields one ``VideoDetectedObject``, preserving every bbox
    observation and scored timestamp range as it appears on the wire.
    """
    df = pl.read_parquet(io.BytesIO(parquet_bytes))
    return [_to_video_detected_object(row) for row in df.iter_rows(named=True)]


def deserialize_object_forward_image_predictions(
    parquet_bytes: bytes,
) -> List[ImageDetectedObject]:
    """Map an object-forward parquet blob to a list of ``ImageDetectedObject``.

    The wire format is the same as video, but the time dimension is collapsed:
    each object keeps only its single bbox observation (without timestamp) and
    each attribute keeps only its ``score``.
    """
    df = pl.read_parquet(io.BytesIO(parquet_bytes))
    return [_to_image_detected_object(row) for row in df.iter_rows(named=True)]
