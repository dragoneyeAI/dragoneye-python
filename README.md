# dragoneye-python

[![PyPI version](https://img.shields.io/pypi/v/dragoneye-python.svg)](https://pypi.org/project/dragoneye-python/)
[![Python Versions](https://img.shields.io/pypi/pyversions/dragoneye-python.svg)](https://pypi.org/project/dragoneye-python/)
[![License: MIT](https://img.shields.io/pypi/l/dragoneye-python.svg)](https://github.com/dragoneyeAI/dragoneye-python/blob/main/LICENSE)

The official Python SDK for [Dragoneye](https://dragoneye.ai) — build and call custom computer vision models from Python.

Describe what you want to detect in plain English on the [Dragoneye Playground](https://playground.dragoneye.ai/), and the AI Model Builder assembles a vision model with your categories and attributes. This SDK lets you run that model on images and videos and get back structured predictions with bounding boxes, category scores, and attribute scores.

- 📘 **Full documentation:** https://docs.dragoneye.ai/integrating/python-sdk
- 🎮 **Playground:** https://playground.dragoneye.ai/
- 🐍 **PyPI:** https://pypi.org/project/dragoneye-python/

---

## Using the Python SDK

If you're integrating with our APIs using Python, the Dragoneye SDK streamlines the process with minimal setup. Here's how you can get started and explore the types and endpoints in detail.

## Installation

Install the package using pip.

```bash
pip install dragoneye-python
```

## Quick Start

> **Tip — Prerequisites**: Don't have an API key yet? See [Creating an Access Token](https://docs.dragoneye.ai/account-management/creating-access-token).

To call the classifier, follow these steps:

```python
import asyncio
from dragoneye import Dragoneye, Image, Video

async def main():
    # The api_key can also be set via the DRAGONEYE_API_KEY environment variable.
    client = Dragoneye(api_key="<YOUR_ACCESS_TOKEN>")

    # Example: predict from an image
    image = Image.from_path("photo.jpg")
    image_result = await client.classification.predict_image(
        media=image,
        model_name="recognize_anything/your_model_name",  # change to your desired model
    )

    # Example: predict from a video
    # NOTE! When loading a file, you can optionally pass a file name or identifier
    # that you use to identify your own files.
    video = Video.from_path(
        path="example.mp4",
        name="any-file-name",
    )
    video_result = await client.classification.predict_video(
        media=video,
        model_name="recognize_anything/your_model_name",
    )

    # Accessing image results
    for obj in image_result.object_predictions:
        bbox = obj.normalizedBbox
        for pred in obj.predictions:
            print(f"Category: {pred.category.name} ({pred.category.score:.2f})")
            for attr in pred.attributes:
                top_option = max(attr.options, key=lambda o: o.score)
                print(f"  {attr.name}: {top_option.name} ({top_option.score:.2f})")

asyncio.run(main())
```

> **Note — Model names**: Model names follow the format `recognize_anything/model_name`. Use the name you specified when creating the model.

### Example Video Response

Below is an example of what a `ClassificationPredictVideoResponse` looks like for a Building Detection model. The response maps each sampled frame's timestamp (in microseconds) to the objects detected in that frame:

```python
ClassificationPredictVideoResponse(
    prediction_task_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    original_file_name="any-file-name",
    frames_per_second=1,
    timestamp_us_to_predictions={
        0: [
            ClassificationVideoObjectPrediction(
                frame_id="frame_0",
                timestamp_microseconds=0,
                normalizedBbox=(0.12, 0.25, 0.55, 0.78),
                predictions=[
                    ClassificationCategoryPrediction(
                        category=ClassificationCategory(
                            id=2084323334,
                            name="House (detached)",
                            score=0.92,
                        ),
                        attributes=[
                            ClassificationAttributeResponse(
                                attribute_id=1371766615,
                                name="Building Exterior Color",
                                options=[
                                    ClassificationAttributeOption(option_id=3498033303, name="White / Off-white", score=0.85),
                                    ClassificationAttributeOption(option_id=496739380, name="Light gray", score=0.10),
                                    # ... remaining options omitted for brevity
                                ],
                            ),
                            ClassificationAttributeResponse(
                                attribute_id=448392115,
                                name="Building Exterior Material",
                                options=[
                                    ClassificationAttributeOption(option_id=3887467550, name="Wood (incl. timber siding)", score=0.78),
                                    ClassificationAttributeOption(option_id=562768697, name="Brick", score=0.12),
                                    # ...
                                ],
                            ),
                            ClassificationAttributeResponse(
                                attribute_id=4240554102,
                                name="Building Size (Stories)",
                                options=[
                                    ClassificationAttributeOption(option_id=3067238669, name="2 stories", score=0.91),
                                    ClassificationAttributeOption(option_id=2398426374, name="1 story", score=0.06),
                                    # ...
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            ClassificationVideoObjectPrediction(
                frame_id="frame_0",
                timestamp_microseconds=0,
                normalizedBbox=(0.60, 0.30, 0.88, 0.75),
                predictions=[
                    ClassificationCategoryPrediction(
                        category=ClassificationCategory(
                            id=3212613421,
                            name="Garage (detached)",
                            score=0.87,
                        ),
                        attributes=[
                            # ... attributes omitted for brevity
                        ],
                    ),
                ],
            ),
        ],
        1000000: [
            # Objects detected at t=1s (1,000,000 microseconds)
            # ...
        ],
    },
)
```

Each timestamp key (e.g., `0`, `1000000`) corresponds to a sampled frame. Within each frame, every detected object has its own bounding box, category prediction with a confidence score, and attribute predictions with scored options.

---

## Client

**`Dragoneye`**

The main client used to interact with the API.

```python
client = Dragoneye(
    api_key="<YOUR_ACCESS_TOKEN>",
    max_retries=10,
    max_backoff_time=120,
)
```

**Arguments:**

- `api_key` (Optional[str]): Your API key. If omitted, the SDK reads from the `DRAGONEYE_API_KEY` environment variable.
- `max_retries` (int): Maximum retry attempts on rate-limit (429) responses. Default: `10`.
- `max_backoff_time` (int): Maximum backoff time in seconds for exponential backoff. Default: `120`.

---

## Media Classes

`Image` and `Video` are used to wrap media before passing it to a prediction endpoint. Each class restricts the MIME type to its respective media type (`image/*` or `video/*`).

### Constructors

**`from_path`**

```python
media = Image.from_path(
    path="photo.jpg",
    name="my-photo",          # optional identifier
    mime_type=None,            # auto-detected from extension by default
    guess_from_extension=True, # set False to require explicit mime_type
    read_into_memory=False,    # set True to load bytes into memory immediately
)
```

**`from_bytes`**

```python
media = Image.from_bytes(
    data=raw_bytes,
    mime_type="image/jpeg",
    name="my-photo",  # optional
)
```

**`from_stream`**

```python
media = Video.from_stream(
    stream=open("clip.mp4", "rb"),
    mime_type="video/mp4",
    name="my-clip",  # optional
)
```

---

## Types and Endpoints

### Types

The response types form a nested hierarchy. Here's how they fit together for image predictions:

```
ClassificationPredictImageResponse
└── object_predictions: [ClassificationObjectPrediction]
    ├── normalizedBbox: (x1, y1, x2, y2)
    └── predictions: [ClassificationCategoryPrediction]
        ├── category: ClassificationCategory (id, name, score)
        └── attributes: [ClassificationAttributeResponse]
            └── options: [ClassificationAttributeOption] (option_id, name, score)
```

For video predictions, `ClassificationPredictVideoResponse` maps timestamps to lists of `ClassificationVideoObjectPrediction` (which extends `ClassificationObjectPrediction` with `frame_id` and `timestamp_microseconds`).

---

**`ClassificationCategory`**
Represents a predicted category.

Attributes:

- `id` (int): Unique identifier for the category.
- `name` (str): The name of the category.
- `score` (float): Confidence score for the prediction.

**`ClassificationAttributeOption`**
Represents a single option within an attribute prediction.

Attributes:

- `option_id` (int): Unique identifier for the option.
- `name` (str): The name of the option.
- `score` (float): Confidence score for this option.

**`ClassificationAttributeResponse`**
Represents a predicted attribute with its possible options.

Attributes:

- `attribute_id` (int): Unique identifier for the attribute.
- `name` (str): The name of the attribute.
- `options` (List[ClassificationAttributeOption]): The predicted options for this attribute.

**`ClassificationCategoryPrediction`**
Represents a category prediction along with its associated attribute predictions.

Attributes:

- `category` (ClassificationCategory): The predicted category.
- `attributes` (List[ClassificationAttributeResponse]): Attribute predictions for this category.

**`ClassificationObjectPrediction`**
Represents the prediction of a detected object in an image.

Attributes:

- `normalizedBbox` (NormalizedBbox): A bounding box for the detected object (coordinates are normalized).
- `predictions` (List[ClassificationCategoryPrediction]): Category and attribute predictions for this object.

**`ClassificationVideoObjectPrediction`**
Extends `ClassificationObjectPrediction` with video-specific fields.

Attributes:

- `normalizedBbox` (NormalizedBbox): A bounding box for the detected object.
- `predictions` (List[ClassificationCategoryPrediction]): Category and attribute predictions.
- `frame_id` (str): Identifier for the frame.
- `timestamp_microseconds` (int): Timestamp of the frame in microseconds.

**`ClassificationPredictImageResponse`**
The response object returned after predicting an image.

Attributes:

- `object_predictions` (List[ClassificationObjectPrediction]): Detected objects and their predictions.
- `prediction_task_uuid` (str): The unique identifier for the prediction task.
- `original_file_name` (Optional[str]): The file name of the original media, if provided.

**`ClassificationPredictVideoResponse`**
The response object returned after predicting a video.

Attributes:

- `timestamp_us_to_predictions` (Dict[int, List[ClassificationVideoObjectPrediction]]): A mapping from timestamp (in microseconds) to object predictions for that frame.
- `frames_per_second` (int): The number of frames per second that were sampled.
- `prediction_task_uuid` (str): The unique identifier for the prediction task.
- `original_file_name` (Optional[str]): The file name of the original media, if provided.

**`PredictionTaskStatusResponse`**
Represents the status of a prediction task.

Attributes:

- `prediction_task_uuid` (str): The unique identifier for the task.
- `prediction_type` (str): Either `"image"` or `"video"`.
- `status` (str): The current task status (`predicted`, `failed`, etc.).

**`NormalizedBbox`**
Type alias for normalized bounding boxes, represented as a tuple of four float values.

---

### Endpoints

#### `client.classification.predict_image`

```python
await client.classification.predict_image(
    media: Image,
    model_name: str,
    timeout_seconds: Optional[int] = None,
) -> ClassificationPredictImageResponse
```

Performs a classification prediction on a single image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `media` | `Image` | *required* | An `Image` object (from `from_path`, `from_bytes`, or `from_stream`). |
| `model_name` | `str` | *required* | The name of the model to use for prediction. |
| `timeout_seconds` | `Optional[int]` | `None` | Maximum wait time in seconds. Raises `PredictionTimeoutException` on timeout. `None` polls indefinitely. |

**Returns:** `ClassificationPredictImageResponse` — detected objects and their predictions.

---

#### `client.classification.predict_video`

```python
await client.classification.predict_video(
    media: Video,
    model_name: str,
    frames_per_second: int = 1,
    timeout_seconds: Optional[int] = None,
) -> ClassificationPredictVideoResponse
```

Performs a classification prediction on a video.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `media` | `Video` | *required* | A `Video` object (from `from_path`, `from_bytes`, or `from_stream`). |
| `model_name` | `str` | *required* | The name of the model to use for prediction. |
| `frames_per_second` | `int` | `1` | How many frames per second to sample from the video. |
| `timeout_seconds` | `Optional[int]` | `None` | Maximum wait time in seconds. Raises `PredictionTimeoutException` on timeout. `None` polls indefinitely. |

**Returns:** `ClassificationPredictVideoResponse` — frame-level prediction results.

---

#### `client.classification.status`

```python
await client.classification.status(
    prediction_task_uuid: str,
) -> PredictionTaskStatusResponse
```

Checks the status of a prediction task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_task_uuid` | `str` | *required* | The UUID of the prediction task. |

**Returns:** `PredictionTaskStatusResponse` — the task's current status.

---

#### `client.classification.get_image_results`

```python
await client.classification.get_image_results(
    prediction_task_uuid: str,
) -> ClassificationPredictImageResponse
```

Retrieves the results of a completed image prediction task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_task_uuid` | `str` | *required* | The UUID of the prediction task. |

**Returns:** `ClassificationPredictImageResponse`

---

#### `client.classification.get_video_results`

```python
await client.classification.get_video_results(
    prediction_task_uuid: str,
) -> ClassificationPredictVideoResponse
```

Retrieves the results of a completed video prediction task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_task_uuid` | `str` | *required* | The UUID of the prediction task. |

**Returns:** `ClassificationPredictVideoResponse`

---

## Error Handling

The SDK defines the following exception types:

| Exception | When it's raised |
|-----------|-----------------|
| `PredictionTimeoutException` | The prediction did not complete within the specified `timeout_seconds`. |
| `PredictionTaskError` | The prediction task failed on the server. |
| `PredictionUploadError` | The media file could not be uploaded. |
| `PredictionTaskBeginError` | The prediction task could not be started. |
| `PredictionTaskResultsUnavailableError` | Results were requested for a task that has not completed. |

```python
from dragoneye import Dragoneye, Image
from dragoneye.types.exception import (
    PredictionTimeoutException,
    PredictionTaskError,
    PredictionUploadError,
)

try:
    result = await client.classification.predict_image(
        media=image,
        model_name="recognize_anything/your_model_name",
        timeout_seconds=60,
    )
except PredictionTimeoutException:
    print("Prediction timed out — try increasing timeout_seconds")
except PredictionUploadError:
    print("Failed to upload media — check file path and format")
except PredictionTaskError:
    print("Prediction task failed on the server")
```

---

## Notes

- All public methods are **asynchronous**. Use `asyncio.run` or an async loop to call them.
- For images, use `predict_image` with an `Image` object. For videos, use `predict_video` with a `Video` object. Passing the wrong media type will raise a `ValueError`.
- Predictions are executed as tasks: the SDK automatically handles task creation, media upload, polling, and result retrieval.
- The SDK automatically retries on rate-limit (429) responses using exponential backoff. You can configure this behavior via the `max_retries` and `max_backoff_time` parameters on the `Dragoneye` client.
