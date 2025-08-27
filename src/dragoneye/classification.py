import time
from typing import TYPE_CHECKING, Any, Optional, Sequence

import requests
from pydantic import BaseModel

from .types.common import (
    BASE_API_URL,
    NormalizedBbox,
    PredictionTaskState,
    PredictionTaskUUID,
    PredictionType,
    TaxonID,
    TaxonPrediction,
)
from .types.exception import (
    PredictionTaskBeginError,
    PredictionTaskError,
    PredictionTaskResultsUnavailableError,
    PredictionUploadError,
)
from .types.media import Media

if TYPE_CHECKING:
    from .client import Dragoneye


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


class _PresignedPostRequest(BaseModel):
    url: str
    fields: dict[str, Any]


class _MediaUploadUrl(BaseModel):
    blob_path: str
    presigned_post_request: _PresignedPostRequest


class _PredictionTaskBeginResponse(BaseModel):
    prediction_task_uuid: PredictionTaskUUID
    prediction_type: PredictionType
    signed_urls: list[_MediaUploadUrl]


class PredictionTaskStatusResponse(BaseModel):
    prediction_task_uuid: PredictionTaskUUID
    prediction_type: PredictionType
    status: PredictionTaskState


def _is_task_successful(status: PredictionTaskState) -> bool:
    return status == "predicted"


def _is_task_failed(status: PredictionTaskState) -> bool:
    return status.startswith("failed")


def _is_task_complete(status: PredictionTaskState) -> bool:
    """
    Returns True if the prediction task is complete, either successfully or unsuccessfully.

    Avoid enum to allow additional states to be backwards compatible.
    """
    return _is_task_successful(status) or _is_task_failed(status)


class Classification:
    def __init__(self, client: "Dragoneye"):
        self._client = client

    def predict(
        self, media: Media, model_name: str
    ) -> ClassificationPredictImageResponse | ClassificationPredictVideoResponse:
        return self._predict_unified(media, model_name, frames_per_second=None)

    def status(
        self, prediction_task_uuid: PredictionTaskUUID
    ) -> PredictionTaskStatusResponse:
        """
        Given a prediction task UUID, return
        """
        url = f"{BASE_API_URL}/prediction-task/status?predictionTaskUuid={prediction_task_uuid}"

        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return PredictionTaskStatusResponse.model_validate(response.json())

    def get_results(
        self, prediction_task_uuid: PredictionTaskUUID, prediction_type: PredictionType
    ) -> ClassificationPredictImageResponse | ClassificationPredictVideoResponse:
        url = f"{BASE_API_URL}/prediction-task/results?predictionTaskUuid={prediction_task_uuid}"

        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as error:
            raise PredictionTaskResultsUnavailableError(
                f"Error getting prediction task results: {error}"
            )

        match prediction_type:
            case "image":
                return ClassificationPredictImageResponse.model_validate(
                    response.json()
                )
            case "video":
                return ClassificationPredictVideoResponse.model_validate(
                    response.json()
                )
            case _:  # pyright: ignore[reportUnnecessaryComparison]
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

    ##### Internal API methods #####
    def _predict_unified(
        self,
        media: Media,
        model_name: str,
        frames_per_second: Optional[int],
    ) -> ClassificationPredictImageResponse | ClassificationPredictVideoResponse:
        prediction_task_begin_response = self._begin_prediction_task(
            mime_type=media.mime_type,
            frames_per_second=frames_per_second,
        )

        self._upload_media_to_prediction_task(
            media, prediction_task_begin_response.signed_urls[0]
        )

        predict_url = f"{BASE_API_URL}/predict"
        predict_data = {
            "model_name": model_name,
            "prediction_task_uuid": prediction_task_begin_response.prediction_task_uuid,
        }
        try:
            response = requests.post(predict_url, data=predict_data)
            response.raise_for_status()
        except requests.RequestException as error:
            raise PredictionTaskError("Error initiating prediction:", error)

        status = self._wait_for_prediction_task_completion(
            prediction_task_uuid=prediction_task_begin_response.prediction_task_uuid
        )

        if _is_task_failed(status.status):
            raise PredictionTaskError(f"Prediction task failed: {status.status}")

        return self.get_results(
            prediction_task_uuid=status.prediction_task_uuid,
            prediction_type=status.prediction_type,
        )

    def _wait_for_prediction_task_completion(
        self, prediction_task_uuid: PredictionTaskUUID, polling_interval: float = 1.0
    ) -> PredictionTaskStatusResponse:
        while True:
            status = self.status(prediction_task_uuid)
            if _is_task_complete(status.status):
                return status
            time.sleep(polling_interval)

    def _upload_media_to_prediction_task(
        self, media: Media, signed_url: _MediaUploadUrl
    ) -> None:
        form_data = {}
        for key, value in signed_url.presigned_post_request.fields.items():
            form_data[key] = (None, value)

        form_data["file"] = ("file", media.bytes_io())

        try:
            response = requests.post(
                signed_url.presigned_post_request.url,
                data=form_data,
                headers={"Content-Type": "multipart/form-data"},
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise PredictionUploadError(
                "Error uploading media to prediction task:", error
            )

    def _begin_prediction_task(
        self,
        mime_type: str,
        frames_per_second: Optional[int],
    ) -> _PredictionTaskBeginResponse:
        url = f"{BASE_API_URL}/prediction-task/begin"

        form_data = {}
        form_data["mimetype"] = mime_type
        if frames_per_second is not None:
            form_data["frames_per_second"] = str(frames_per_second)

        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "multipart/form-data",
        }

        try:
            response = requests.post(url, data=form_data, headers=headers)
            response.raise_for_status()
        except requests.RequestException as error:
            raise PredictionTaskBeginError("Error beginning prediction task:", error)

        return _PredictionTaskBeginResponse.model_validate(response.json())
