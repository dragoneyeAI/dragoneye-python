from io import BytesIO
from typing import TYPE_CHECKING, Any, Sequence

import requests
from pydantic import BaseModel

from .types.common import BASE_API_URL, NormalizedBbox, TaxonID, TaxonPrediction
from .types.image import Image, assert_consistent_data_type

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


class ClassificationProductPrediction(BaseModel):
    category: TaxonPrediction
    traits: Sequence[ClassificationTraitRootPrediction]


class ClassificationPredictProductResponse(BaseModel):
    predictions: Sequence[ClassificationProductPrediction]


class Classification:
    def __init__(self, client: "Dragoneye"):
        self._client = client

    def predict_image(
        self, image: Image, model_name: str
    ) -> ClassificationPredictImageResponse:
        url = f"{BASE_API_URL}/predict"

        files: dict[str, BytesIO] = {}
        data: dict[str, str] = {}

        if image.file_or_bytes is not None:
            files["image_file"] = image.bytes_io_x()
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

    def predict_product(
        self, images: Sequence[Image], model_name: str
    ) -> ClassificationPredictProductResponse:
        url = f"{BASE_API_URL}/predict-product"

        files: list[tuple[str, BytesIO]] = []
        data: dict[str, Any] = {}

        assert_consistent_data_type(images)

        for image in images:
            if image.file_or_bytes is not None:
                files.append(("image_file", image.bytes_io_x()))
            elif image.url is not None:
                data.setdefault("image_urls", []).append(image.url)

        data["model_name"] = model_name

        headers = {"Authorization": f"Bearer {self._client.api_key}"}

        try:
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
        except requests.RequestException as error:
            raise Exception(
                "Error during Dragoneye Classification prediction request:", error
            )

        return ClassificationPredictProductResponse.model_validate(response.json())
