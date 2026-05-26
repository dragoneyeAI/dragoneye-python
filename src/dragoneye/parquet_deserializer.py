"""Convert a zstd-compressed parquet prediction blob into typed SDK models.

This implements the **object-forward** schema: the server returns one parquet
row per tracked object (a ``DetectedObject``). Everything about that object —
its presence ranges, every bbox observation over time, its categories, and each
category's attribute time-runs — is nested inside that single row. Images are
encoded the same way as video, just with all timestamps equal to ``0``.

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
    AttributePrediction,
    BboxObservation,
    CategoryPrediction,
    DetectedObject,
    ScoredTimestampRange,
    TimestampRange,
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


def _to_bbox_observation(value: Dict[str, Any]) -> BboxObservation:
    return BboxObservation(
        timestamp_microseconds=value["timestamp_microseconds"],
        normalized_bbox=NormalizedBbox(tuple(value["normalized_bbox"])),
        bbox_score=value["bbox_score"],
    )


def _to_attribute_prediction(value: Dict[str, Any]) -> AttributePrediction:
    return AttributePrediction(
        attribute_id=value["attribute_id"],
        attribute_name=value["attribute_name"],
        option_id=value["option_id"],
        option_name=value["option_name"],
        timestamp_ranges=[
            _to_scored_timestamp_range(tr)
            for tr in (value["timestamp_ranges"] or [])
        ],
    )


def _to_category_prediction(value: Dict[str, Any]) -> CategoryPrediction:
    return CategoryPrediction(
        category_id=value["category_id"],
        name=value["name"],
        score=value["score"],
        attributes=[
            _to_attribute_prediction(attr) for attr in (value["attributes"] or [])
        ],
    )


def _to_detected_object(row: Dict[str, Any]) -> DetectedObject:
    return DetectedObject(
        object_id=row["object_id"],
        timestamp_ranges=[
            _to_timestamp_range(tr) for tr in (row["timestamp_ranges"] or [])
        ],
        bbox_observations=[
            _to_bbox_observation(obs) for obs in (row["bbox_observations"] or [])
        ],
        categories=[
            _to_category_prediction(cat) for cat in (row["categories"] or [])
        ],
    )


def deserialize_object_forward_predictions(
    parquet_bytes: bytes,
) -> List[DetectedObject]:
    """Map an object-forward parquet blob to a list of ``DetectedObject``.

    One parquet row yields one ``DetectedObject``. Used for both image and
    video responses — images simply carry a single bbox observation at
    ``timestamp_microseconds == 0`` with a zero-width timestamp range.
    """
    df = pl.read_parquet(io.BytesIO(parquet_bytes))
    return [_to_detected_object(row) for row in df.iter_rows(named=True)]
