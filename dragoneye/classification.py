import io
from typing import TYPE_CHECKING, BinaryIO, Sequence

import requests
from pydantic import BaseModel

from .types.common import BASE_API_URL, NormalizedBbox, TaxonID, TaxonPrediction
from .types.image import Image

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


class Classification:
    def __init__(self, client: "Dragoneye"):
        self._client = client

    def predict(
        self, image: Image, model_name: str
    ) -> ClassificationPredictImageResponse:
        url = f"{BASE_API_URL}/predict"

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
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
        except requests.RequestException as error:
            raise Exception(
                "Error during Dragoneye Classification prediction request:", error
            )

        return ClassificationPredictImageResponse.model_validate(response.json())
