import io
from typing import TYPE_CHECKING, BinaryIO, Mapping, Optional, Sequence

import requests
from pydantic import BaseModel

from .types.common import BASE_API_URL, NormalizedBbox, ObjectKey, TaxonID
from .types.image import Image

if TYPE_CHECKING:
    from .client import Dragoneye


class ScreenshotObjectPrediction(BaseModel):
    object_key: ObjectKey
    bbox: NormalizedBbox
    taxon_id: TaxonID
    score: float | None


class ScreenshotParseImageResponse(BaseModel):
    objects: Sequence[ScreenshotObjectPrediction]
    sibling_relationships: Mapping[ObjectKey, Sequence[ObjectKey]]
    parent_relationships: Mapping[ObjectKey, Optional[ObjectKey]]
    child_relationships: Mapping[ObjectKey, Sequence[ObjectKey]]


class Screenshot:
    def __init__(self, client: Dragoneye):
        self._client = client

    def parse(self, image: Image, model_name: str) -> ScreenshotParseImageResponse:
        url = f"{BASE_API_URL}/screenshots/parse"

        files = {}
        data = {}

        if image.file_or_bytes is not None:
            if isinstance(image.file_or_bytes, bytes):
                files["image_file"] = io.BytesIO(image.file_or_bytes)
            elif isinstance(
                image.file_or_bytes, BinaryIO
            ):  # pyright: ignore [reportUnnecessaryIsInstance]
                files["image_file"] = image.file_or_bytes
            else:
                raise ValueError("Invalid image type: Must be bytes or BinaryIO")
        elif image.url is not None:
            data["image_url"] = image.url
        else:
            raise ValueError(
                "Missing image: Either image file or image url must be specified"
            )

        data["model_name"] = model_name

        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        try:
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(
                f"Error during Dragoneye Screenshot parse request: {e}"
            ) from e

        return ScreenshotParseImageResponse.model_validate(response.json())
