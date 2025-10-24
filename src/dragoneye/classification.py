import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, overload

import aiohttp
import backoff
from aiohttp import ClientError
from pydantic import BaseModel

from .constants import FAILED_STATUS_PREFIX, PREDICTED_STATUS
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
    PredictionTaskBeginError,
    PredictionTaskError,
    PredictionTaskResultsUnavailableError,
    PredictionTimeoutException,
    PredictionUploadError,
)
from .types.media import Image, Media, Video

if TYPE_CHECKING:
    from .client import Dragoneye


class _PresignedPostRequest(BaseModel):
    url: str
    fields: Dict[str, Any]


class _MediaUploadUrl(BaseModel):
    blob_path: str
    presigned_post_request: _PresignedPostRequest


class _PredictionTaskBeginResponse(BaseModel):
    prediction_task_uuid: PredictionTaskUUID
    prediction_type: PredictionType
    signed_urls: List[_MediaUploadUrl]


def _is_task_successful(status: PredictionTaskState) -> bool:
    return status == PREDICTED_STATUS


def _is_task_failed(status: PredictionTaskState) -> bool:
    return status.startswith(FAILED_STATUS_PREFIX)


def _is_task_complete(status: PredictionTaskState) -> bool:
    """
    Returns True if the prediction task is complete, either successfully or unsuccessfully.

    Avoid enum to allow additional states to be backwards compatible.
    """
    return _is_task_successful(status) or _is_task_failed(status)


class Classification:
    def __init__(self, client: "Dragoneye"):
        self._client = client

        # Create a reusable backoff decorator for 429 rate limit errors
        def _should_retry_429(exception: Exception) -> bool:
            """Check if exception is a 429 rate limit error"""
            return (
                isinstance(exception, aiohttp.ClientResponseError)
                and exception.status == 429
            )

        # Store the backoff decorator as an instance method
        self._backoff_on_429 = backoff.on_exception(
            wait_gen=backoff.expo,
            exception=aiohttp.ClientResponseError,
            max_tries=client.max_retries,
            max_time=client.max_backoff_time,
            on_backoff=lambda e: logging.info(
                f"Rate limit exceeded - backing off: {e}"
            ),
            on_giveup=lambda e: logging.info(f"Rate limit exceeded - giving up: {e}"),
            giveup=lambda e: not _should_retry_429(e),
            jitter=client.backoff_jitter,
        )

    async def predict_image(
        self,
        media: Image,
        model_name: str,
        timeout_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> ClassificationPredictImageResponse:
        return await self._predict_unified(
            media=media,
            model_name=model_name,
            frames_per_second=None,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

    async def predict_video(
        self,
        media: Video,
        model_name: str,
        frames_per_second: int = 1,
        timeout_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> ClassificationPredictVideoResponse:
        return await self._predict_unified(
            media=media,
            model_name=model_name,
            frames_per_second=frames_per_second,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

    async def status(
        self, prediction_task_uuid: PredictionTaskUUID
    ) -> PredictionTaskStatusResponse:
        """
        Given a prediction task UUID, return
        """
        url = f"{BASE_API_URL}/prediction-task/status?predictionTaskUuid={prediction_task_uuid}"
        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        @self._backoff_on_429
        async def _make_request():
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
            return payload

        payload = await _make_request()
        return PredictionTaskStatusResponse.model_validate(payload)

    async def get_image_results(
        self,
        prediction_task_uuid: PredictionTaskUUID,
    ) -> ClassificationPredictImageResponse:
        return await self._get_results_unified(
            prediction_task_uuid=prediction_task_uuid,
            prediction_type="image",
        )

    async def get_video_results(
        self,
        prediction_task_uuid: PredictionTaskUUID,
    ) -> ClassificationPredictVideoResponse:
        return await self._get_results_unified(
            prediction_task_uuid=prediction_task_uuid,
            prediction_type="video",
        )

    @overload
    async def _get_results_unified(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        prediction_type: Literal["image"],
    ) -> ClassificationPredictImageResponse: ...

    @overload
    async def _get_results_unified(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        prediction_type: Literal["video"],
    ) -> ClassificationPredictVideoResponse: ...

    @overload
    async def _get_results_unified(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        prediction_type: PredictionType,
    ) -> Union[
        ClassificationPredictImageResponse, ClassificationPredictVideoResponse
    ]: ...

    async def _get_results_unified(
        self, prediction_task_uuid: PredictionTaskUUID, prediction_type: PredictionType
    ) -> Union[ClassificationPredictImageResponse, ClassificationPredictVideoResponse]:
        url = f"{BASE_API_URL}/prediction-task/results?predictionTaskUuid={prediction_task_uuid}"
        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        @self._backoff_on_429
        async def _make_request():
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
            return payload

        try:
            payload = await _make_request()
        except ClientError as error:
            raise PredictionTaskResultsUnavailableError(
                f"Error getting prediction task results: {error}"
            )

        # Add the prediction task uuid to the response before returning
        payload["prediction_task_uuid"] = prediction_task_uuid

        if prediction_type == "image":
            return ClassificationPredictImageResponse.model_validate(payload)
        elif prediction_type == "video":
            return ClassificationPredictVideoResponse.model_validate(payload)
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

    ##### Internal API methods #####
    @overload
    async def _predict_unified(
        self,
        media: Image,
        model_name: str,
        frames_per_second: Optional[int],
        timeout_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> ClassificationPredictImageResponse: ...

    @overload
    async def _predict_unified(
        self,
        media: Video,
        model_name: str,
        frames_per_second: Optional[int],
        timeout_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> ClassificationPredictVideoResponse: ...

    async def _predict_unified(
        self,
        media: Union[Image, Video],
        model_name: str,
        frames_per_second: Optional[int],
        timeout_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[ClassificationPredictImageResponse, ClassificationPredictVideoResponse]:
        prediction_task_begin_response = await self._begin_prediction_task(
            mime_type=media.mime_type,
            frames_per_second=frames_per_second,
            file_name=media.name,
        )

        await self._upload_media_to_prediction_task(
            media, prediction_task_begin_response.signed_urls[0]
        )

        predict_url = f"{BASE_API_URL}/predict"
        predict_data = {
            "model_name": model_name,
            "prediction_task_uuid": prediction_task_begin_response.prediction_task_uuid,
            **kwargs,
        }
        predict_headers = {
            "Authorization": f"Bearer {self._client.api_key}",
        }

        @self._backoff_on_429
        async def _make_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    predict_url, data=predict_data, headers=predict_headers
                ) as resp:
                    resp.raise_for_status()

        try:
            await _make_request()
        except ClientError as error:
            raise PredictionTaskError("Error initiating prediction:", error)

        status = await self._wait_for_prediction_task_completion(
            prediction_task_uuid=prediction_task_begin_response.prediction_task_uuid,
            timeout_seconds=timeout_seconds,
        )

        if _is_task_failed(status.status):
            raise PredictionTaskError(f"Prediction task failed: {status.status}")

        return await self._get_results_unified(
            prediction_task_uuid=status.prediction_task_uuid,
            prediction_type=prediction_task_begin_response.prediction_type,
        )

    async def _wait_for_prediction_task_completion(
        self,
        prediction_task_uuid: PredictionTaskUUID,
        polling_interval: float = 1.0,
        timeout_seconds: Optional[int] = None,
    ) -> PredictionTaskStatusResponse:
        start_time = time.monotonic()
        while True:
            # Check if we've exceeded the timeout
            if timeout_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout_seconds:
                    raise PredictionTimeoutException(
                        f"Prediction task {prediction_task_uuid} did not complete within {timeout_seconds} seconds."
                    )

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
        file_name: Optional[str],
    ) -> _PredictionTaskBeginResponse:
        url = f"{BASE_API_URL}/prediction-task/begin"

        form_data = aiohttp.FormData()
        form_data.add_field("mimetype", mime_type)
        if file_name is not None:
            form_data.add_field("file_name", file_name)

        if frames_per_second is not None:
            form_data.add_field("frames_per_second", str(frames_per_second))

        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
        }

        @self._backoff_on_429
        async def _make_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form_data, headers=headers) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
            return payload

        try:
            payload = await _make_request()
        except ClientError as error:
            raise PredictionTaskBeginError("Error beginning prediction task:", error)

        return _PredictionTaskBeginResponse.model_validate(payload)
