import asyncio
from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import aiohttp
from aiohttp import ClientError
from pydantic import BaseModel

from .models import (
    ClassificationPredictImageResponse,
    ClassificationPredictVideoResponse,
    PredictionTaskStatusResponse,
)
from .types.common import (
    BASE_API_URL,
    PredictionTaskState,
    PredictionTaskUUID,
    PredictionType,
)
from .types.exception import (
    IncorrectMediaTypeError,
    PredictionTaskBeginError,
    PredictionTaskError,
    PredictionTaskResultsUnavailableError,
    PredictionUploadError,
)
from .types.media import Media

if TYPE_CHECKING:
    from .client import Dragoneye


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

    async def predict_image(
        self, media: Media, model_name: str
    ) -> ClassificationPredictImageResponse:
        media_type = media.mime_type.split("/")[0]
        match media_type:
            case "image":
                return await self._predict_unified(
                    media=media,
                    model_name=model_name,
                    prediction_type="image",
                    frames_per_second=None,
                )
            case "video":
                raise IncorrectMediaTypeError(
                    "Incorrect media type for predict_image, use predict_video instead"
                )
            case _:
                raise IncorrectMediaTypeError(
                    f"Unsupported media type for predict_image: {media.mime_type}"
                )

    async def predict_video(
        self, media: Media, model_name: str, frames_per_second: int
    ) -> ClassificationPredictVideoResponse:
        media_type = media.mime_type.split("/")[0]
        match media_type:
            case "video":
                return await self._predict_unified(
                    media=media,
                    model_name=model_name,
                    prediction_type="video",
                    frames_per_second=frames_per_second,
                )
            case "image":
                raise IncorrectMediaTypeError(
                    "Incorrect media type for predict_video, use predict_image instead"
                )
            case _:
                raise IncorrectMediaTypeError(
                    f"Unsupported media type for predict_video: {media.mime_type}"
                )

    async def status(
        self, prediction_task_uuid: PredictionTaskUUID
    ) -> PredictionTaskStatusResponse:
        """
        Given a prediction task UUID, return
        """
        url = f"{BASE_API_URL}/prediction-task/status?predictionTaskUuid={prediction_task_uuid}"
        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                payload = await resp.json()

        return PredictionTaskStatusResponse.model_validate(payload)

    @overload
    async def get_results(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        prediction_type: Literal["image"],
    ) -> ClassificationPredictImageResponse: ...

    @overload
    async def get_results(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        prediction_type: Literal["video"],
    ) -> ClassificationPredictVideoResponse: ...

    @overload
    async def get_results(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        prediction_type: PredictionType,
    ) -> ClassificationPredictImageResponse | ClassificationPredictVideoResponse: ...

    async def get_results(
        self, prediction_task_uuid: PredictionTaskUUID, prediction_type: PredictionType
    ) -> ClassificationPredictImageResponse | ClassificationPredictVideoResponse:
        url = f"{BASE_API_URL}/prediction-task/results?predictionTaskUuid={prediction_task_uuid}"
        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
        except ClientError as error:
            raise PredictionTaskResultsUnavailableError(
                f"Error getting prediction task results: {error}"
            )

        # Add the prediction task uuid to the response before returning
        payload["prediction_task_uuid"] = prediction_task_uuid

        match prediction_type:
            case "image":
                return ClassificationPredictImageResponse.model_validate(payload)
            case "video":
                return ClassificationPredictVideoResponse.model_validate(payload)
            case _:  # pyright: ignore [reportUnnecessaryComparison]
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

    ##### Internal API methods #####
    @overload
    async def _predict_unified(
        self,
        media: Media,
        model_name: str,
        prediction_type: Literal["image"],
        frames_per_second: Literal[None],
    ) -> ClassificationPredictImageResponse: ...

    @overload
    async def _predict_unified(
        self,
        media: Media,
        model_name: str,
        prediction_type: Literal["video"],
        frames_per_second: Optional[int],
    ) -> ClassificationPredictVideoResponse: ...

    async def _predict_unified(
        self,
        media: Media,
        model_name: str,
        prediction_type: PredictionType,
        frames_per_second: Optional[int],
    ) -> ClassificationPredictImageResponse | ClassificationPredictVideoResponse:
        prediction_task_begin_response = await self._begin_prediction_task(
            mime_type=media.mime_type,
            frames_per_second=frames_per_second,
        )

        if prediction_task_begin_response.prediction_type != prediction_type:
            raise PredictionTaskBeginError(
                f"Prediction type mismatch: {prediction_task_begin_response.prediction_type} != {prediction_type}"
            )

        await self._upload_media_to_prediction_task(
            media, prediction_task_begin_response.signed_urls[0]
        )

        predict_url = f"{BASE_API_URL}/predict"
        predict_data = {
            "model_name": model_name,
            "prediction_task_uuid": prediction_task_begin_response.prediction_task_uuid,
        }
        predict_headers = {
            "Authorization": f"Bearer {self._client.api_key}",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    predict_url, data=predict_data, headers=predict_headers
                ) as resp:
                    resp.raise_for_status()
        except ClientError as error:
            raise PredictionTaskError("Error initiating prediction:", error)

        status = await self._wait_for_prediction_task_completion(
            prediction_task_uuid=prediction_task_begin_response.prediction_task_uuid
        )

        if _is_task_failed(status.status):
            raise PredictionTaskError(f"Prediction task failed: {status.status}")

        return await self.get_results(
            prediction_task_uuid=status.prediction_task_uuid,
            prediction_type=prediction_type,
        )

    async def _wait_for_prediction_task_completion(
        self, prediction_task_uuid: PredictionTaskUUID, polling_interval: float = 1.0
    ) -> PredictionTaskStatusResponse:
        while True:
            status = await self.status(prediction_task_uuid)
            if _is_task_complete(status.status):
                return status
            await asyncio.sleep(polling_interval)

    async def _upload_media_to_prediction_task(
        self, media: Media, signed_url: _MediaUploadUrl
    ) -> None:
        # Build multipart form: include all presigned fields + the file
        form = aiohttp.FormData()
        for k, v in signed_url.presigned_post_request.fields.items():
            form.add_field(k, str(v))

        file_obj = media.bytes_io()
        try:
            file_obj.seek(0)
        except Exception:
            pass  # if it's already at start or non-seekable

        form.add_field(
            "file",
            file_obj,
            filename="file",
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    signed_url.presigned_post_request.url,
                    data=form,
                ) as resp:
                    resp.raise_for_status()
        except ClientError as error:
            raise PredictionUploadError(
                "Error uploading media to prediction task:", error
            )

    async def _begin_prediction_task(
        self,
        mime_type: str,
        frames_per_second: Optional[int],
    ) -> _PredictionTaskBeginResponse:
        url = f"{BASE_API_URL}/prediction-task/begin"

        form_data = aiohttp.FormData()
        form_data.add_field("mimetype", mime_type)
        if frames_per_second is not None:
            form_data.add_field("frames_per_second", str(frames_per_second))

        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form_data, headers=headers) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
        except ClientError as error:
            raise PredictionTaskBeginError("Error beginning prediction task:", error)

        return _PredictionTaskBeginResponse.model_validate(payload)
