from typing import Literal, NewType, Tuple

PredictionType = Literal["image", "video"]
PredictionTaskState = NewType("PredictionTaskState", str)

NormalizedBbox = NewType("NormalizedBbox", Tuple[float, float, float, float])

PredictionTaskUUID = NewType("PredictionTaskUUID", str)

TimestampUs = NewType("TimestampUs", int)

BASE_API_URL = "https://api.dragoneye.ai"
