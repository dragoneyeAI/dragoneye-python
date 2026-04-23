"""Convert a zstd-compressed parquet prediction blob into typed SDK models.

Matches the server-side schema produced by the dragoneye pipeline:

    image_id:             String
    normalized_bbox:      Array(Float64, 4)
    bbox_score:           Float64
    predictions:          List(Struct{
                              category_id: Int64,
                              name:        String,
                              score:       Float64,
                              attributes:  List(Struct{
                                  attribute_id: Int64,
                                  name:         String,
                                  options:      List(Struct{
                                      option_id: Int64,
                                      name:      String,
                                      score:     Float64,
                                  }),
                              }),
                          })

For video parquet blobs the extra column ``timestamp_microseconds: Int64`` is
present.
"""

import io
from typing import Any, Dict, List

import polars as pl

from .models import (
    ClassificationAttributeOption,
    ClassificationAttributeResponse,
    ClassificationCategory,
    ClassificationCategoryPrediction,
    ClassificationObjectPrediction,
    ClassificationVideoObjectPrediction,
)
from .types.common import NormalizedBbox


def _predictions_to_models(
    raw_predictions: List[Dict[str, Any]],
) -> List[ClassificationCategoryPrediction]:
    return [
        ClassificationCategoryPrediction(
            category=ClassificationCategory(
                id=pred["category_id"],
                name=pred["name"],
                score=pred["score"],
            ),
            attributes=[
                ClassificationAttributeResponse(
                    attribute_id=attr["attribute_id"],
                    name=attr["name"],
                    options=[
                        ClassificationAttributeOption(
                            option_id=opt["option_id"],
                            name=opt["name"],
                            score=opt["score"],
                        )
                        for opt in attr["options"]
                    ],
                )
                for attr in pred["attributes"]
            ],
        )
        for pred in raw_predictions
    ]


def _read_dataframe(parquet_bytes: bytes, columns: List[str]) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(parquet_bytes), columns=columns)


def deserialize_image_predictions(
    parquet_bytes: bytes,
) -> List[ClassificationObjectPrediction]:
    df = _read_dataframe(
        parquet_bytes, columns=["normalized_bbox", "predictions"]
    ).filter(pl.col("normalized_bbox").is_not_null())
    return [
        ClassificationObjectPrediction(
            normalizedBbox=NormalizedBbox(tuple(row["normalized_bbox"])),
            predictions=_predictions_to_models(row["predictions"] or []),
        )
        for row in df.select("normalized_bbox", "predictions").iter_rows(named=True)
    ]


def deserialize_video_predictions(
    parquet_bytes: bytes,
) -> Dict[int, List[ClassificationVideoObjectPrediction]]:
    df = _read_dataframe(
        parquet_bytes,
        columns=[
            "image_id",
            "normalized_bbox",
            "predictions",
            "timestamp_microseconds",
        ],
    )

    # Frames with detections contribute rows; frames without still appear with a
    # null bbox so the caller can distinguish "no detections" from "frame not
    # processed".
    real_df = df.filter(pl.col("normalized_bbox").is_not_null())

    result: Dict[int, List[ClassificationVideoObjectPrediction]] = {
        int(ts): [
            ClassificationVideoObjectPrediction(
                normalizedBbox=NormalizedBbox(tuple(row["normalized_bbox"])),
                predictions=_predictions_to_models(row["predictions"] or []),
                frame_id=row["image_id"],
                timestamp_microseconds=int(ts),
            )
            for row in group.select(
                "normalized_bbox", "predictions", "image_id"
            ).iter_rows(named=True)
        ]
        for (ts,), group in real_df.group_by(
            "timestamp_microseconds",
        )
    }

    # Include no-detection frames with empty prediction lists.
    if len(df):
        for row in df.select("timestamp_microseconds").unique().iter_rows(named=True):
            ts_raw = row["timestamp_microseconds"]
            if ts_raw is None:
                continue
            ts = int(ts_raw)
            result.setdefault(ts, [])

    return result
