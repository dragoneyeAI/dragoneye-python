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
    for obj in image_result.objects:
        # Each image object has a single bounding box and per-attribute scores.
        bbox = obj.bbox_observation.normalized_bbox
        top_category = max(obj.categories, key=lambda c: c.score)
        print(f"Category: {top_category.name} ({top_category.score:.2f})")
        for attr in top_category.attributes:
            print(f"  {attr.attribute_name}: {attr.option_name} ({attr.score:.2f})")

asyncio.run(main())
```

> **Note — Model names**: Model names follow the format `recognize_anything/model_name`. Use the name you specified when creating the model.

### Example Image Response

Below is an example of what a `ClassificationPredictImageResponse` looks like for a Building Detection model. The response is a flat list of `objects`, where each `ImageDetectedObject` is a single detected entity with one bounding box and per-attribute scores:

```python
ClassificationPredictImageResponse(
    prediction_task_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    original_file_name="my-photo",
    objects=[
        ImageDetectedObject(
            object_id=1,
            bbox_observation=BboxObservation(normalized_bbox=(0.12, 0.25, 0.55, 0.78), bbox_score=0.97),
            categories=[
                ImageCategoryPrediction(
                    category_id=2084323334,
                    name="House (detached)",
                    score=0.92,
                    attributes=[
                        ImageAttributePrediction(
                            attribute_id=1371766615,
                            attribute_name="Building Exterior Color",
                            option_id=3498033303,
                            option_name="White / Off-white",
                            score=0.85,
                        ),
                        ImageAttributePrediction(
                            attribute_id=448392115,
                            attribute_name="Building Exterior Material",
                            option_id=3887467550,
                            option_name="Wood (incl. timber siding)",
                            score=0.78,
                        ),
                        # ... more attributes omitted for brevity
                    ],
                ),
            ],
        ),
        ImageDetectedObject(
            object_id=2,
            bbox_observation=BboxObservation(normalized_bbox=(0.60, 0.30, 0.88, 0.75), bbox_score=0.90),
            categories=[
                ImageCategoryPrediction(
                    category_id=3212613421,
                    name="Garage (detached)",
                    score=0.87,
                    attributes=[
                        # ... attributes omitted for brevity
                    ],
                ),
            ],
        ),
    ],
)
```

Each `ImageDetectedObject` carries an `object_id`, a single `bbox_observation`, and its `categories`. Every attribute is a chosen option with a single `score` — images have no time dimension.

### Example Video Response

Below is an example of what a `ClassificationPredictVideoResponse` looks like for the same model. The response is a flat list of `objects`, where each `VideoDetectedObject` is a single entity tracked across the whole video:

```python
ClassificationPredictVideoResponse(
    prediction_task_uuid="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    original_file_name="any-file-name",
    frames_per_second=1,
    objects=[
        VideoDetectedObject(
            object_id=1,
            timestamp_ranges=[TimestampRange(timestamp_start_us_inclusive=0, timestamp_end_us_inclusive=2000000)],
            # One bbox sighting per sampled frame the object appears in.
            bbox_observations=[
                VideoBboxObservation(timestamp_microseconds=0, observation=BboxObservation(normalized_bbox=(0.12, 0.25, 0.55, 0.78), bbox_score=0.97)),
                VideoBboxObservation(timestamp_microseconds=1000000, observation=BboxObservation(normalized_bbox=(0.13, 0.26, 0.56, 0.79), bbox_score=0.96)),
                VideoBboxObservation(timestamp_microseconds=2000000, observation=BboxObservation(normalized_bbox=(0.14, 0.27, 0.57, 0.80), bbox_score=0.95)),
            ],
            categories=[
                VideoCategoryPrediction(
                    category_id=2084323334,
                    name="House (detached)",
                    score=0.92,
                    # Each attribute is a chosen option with the scored timestamp ranges over which it held.
                    attributes=[
                        VideoAttributePrediction(
                            attribute_id=1371766615,
                            attribute_name="Building Exterior Color",
                            option_id=3498033303,
                            option_name="White / Off-white",
                            timestamp_ranges=[
                                ScoredTimestampRange(timestamp_start_us_inclusive=0, timestamp_end_us_inclusive=2000000, score=0.85),
                            ],
                        ),
                        VideoAttributePrediction(
                            attribute_id=448392115,
                            attribute_name="Building Exterior Material",
                            option_id=3887467550,
                            option_name="Wood (incl. timber siding)",
                            timestamp_ranges=[
                                ScoredTimestampRange(timestamp_start_us_inclusive=0, timestamp_end_us_inclusive=2000000, score=0.78),
                            ],
                        ),
                        # ... more attributes omitted for brevity
                    ],
                ),
            ],
        ),
        VideoDetectedObject(
            object_id=2,
            timestamp_ranges=[TimestampRange(timestamp_start_us_inclusive=0, timestamp_end_us_inclusive=1000000)],
            bbox_observations=[
                VideoBboxObservation(timestamp_microseconds=0, observation=BboxObservation(normalized_bbox=(0.60, 0.30, 0.88, 0.75), bbox_score=0.90)),
                VideoBboxObservation(timestamp_microseconds=1000000, observation=BboxObservation(normalized_bbox=(0.61, 0.31, 0.89, 0.76), bbox_score=0.89)),
            ],
            categories=[
                VideoCategoryPrediction(
                    category_id=3212613421,
                    name="Garage (detached)",
                    score=0.87,
                    attributes=[
                        # ... attributes omitted for brevity
                    ],
                ),
            ],
        ),
    ],
)
```

Each `VideoDetectedObject` carries a stable `object_id` and the `timestamp_ranges` over which it was visible. Its `bbox_observations` record where the object was at each sampled timestamp (in microseconds). Attributes are organized per chosen option: each `VideoAttributePrediction` carries the scored `timestamp_ranges` over which its option held, so the same `attribute_id` may appear more than once with different options if the chosen option changes during the object's lifetime. The same model returns the simpler, timestamp-free `ImageDetectedObject` shape (shown above) when run on an image.

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

The response types form a nested hierarchy. Image and video responses have different object shapes: images are timestamp-free, while videos carry the time dimension.

Image responses use the simpler, timestamp-free shape:

```
ClassificationPredictImageResponse
└── objects: [ImageDetectedObject]
    ├── object_id: int
    ├── bbox_observation: BboxObservation
    │   ├── normalized_bbox: (x1, y1, x2, y2)
    │   └── bbox_score: float
    └── categories: [ImageCategoryPrediction]
        ├── category_id, name, score
        └── attributes: [ImageAttributePrediction]
            ├── attribute_id, attribute_name
            ├── option_id, option_name
            └── score: float
```

Video responses carry the time dimension:

```
ClassificationPredictVideoResponse
└── objects: [VideoDetectedObject]
    ├── object_id: int
    ├── timestamp_ranges: [TimestampRange] (timestamp_start_us_inclusive, timestamp_end_us_inclusive)
    ├── bbox_observations: [VideoBboxObservation]
    │   ├── timestamp_microseconds: int
    │   └── observation: BboxObservation | None  # None on gap frames
    │       ├── normalized_bbox: (x1, y1, x2, y2)
    │       └── bbox_score: float
    └── categories: [VideoCategoryPrediction]
        ├── category_id, name, score
        └── attributes: [VideoAttributePrediction]
            ├── attribute_id, attribute_name
            ├── option_id, option_name
            └── timestamp_ranges: [ScoredTimestampRange] (timestamp_start_us_inclusive, timestamp_end_us_inclusive, score)
```

An image object has exactly one `BboxObservation` and one `score` per attribute. A video object accumulates one `VideoBboxObservation` per processed frame within its lifespan, and each attribute carries the scored timestamp ranges over which its option held. Frames where the object was present but not detected ("predicted-but-undetected" gap frames) are still included, with `observation` set to `None` — skip these when drawing.

---

**`TimestampRange`**
A contiguous span in microseconds, inclusive on both ends. Used by video responses to describe when an object was visible.

Properties:

- `timestamp_start_us_inclusive` (int): Start of the span in microseconds.
- `timestamp_end_us_inclusive` (int): End of the span in microseconds.

**`ScoredTimestampRange`**
A `TimestampRange` that also carries the confidence the chosen option held over it. `score` is the mean of the option's raw per-frame scores over the range.

Properties:

- `timestamp_start_us_inclusive` (int): Start of the span in microseconds.
- `timestamp_end_us_inclusive` (int): End of the span in microseconds.
- `score` (float): Confidence score for the option over this span.

#### Shared types

**`BboxObservation`**
A bounding box and the confidence of the detection that produced it. Shared by image and video responses. Both fields are always present — a `BboxObservation` only ever exists where a box was actually placed (absence is represented by a `None` `observation` on `VideoBboxObservation`).

Properties:

- `normalized_bbox` (NormalizedBbox): The bounding box (normalized coordinates).
- `bbox_score` (float): Confidence score for the bounding box.

#### Image types

**`ImageAttributePrediction`**
A chosen attribute option for an object in an image, with its confidence score.

Properties:

- `attribute_id` (int): Unique identifier for the attribute.
- `attribute_name` (str): The name of the attribute.
- `option_id` (int): Unique identifier for the chosen option.
- `option_name` (str): The name of the chosen option.
- `score` (float): Confidence score for the chosen option.

**`ImageCategoryPrediction`**
A predicted category and its attribute predictions for an object in an image.

Properties:

- `category_id` (int): Unique identifier for the category.
- `name` (str): The name of the category.
- `score` (float): Confidence score for the category.
- `attributes` (List[ImageAttributePrediction]): Attribute predictions for this category.

**`ImageDetectedObject`**
A single detected object in an image: its bounding box and its categories.

Properties:

- `object_id` (int): Identifier for the detected object.
- `bbox_observation` (BboxObservation): The object's bounding box.
- `categories` (List[ImageCategoryPrediction]): Category and attribute predictions for this object.

#### Video types

**`VideoBboxObservation`**
A single sighting of a tracked object at one timestamp. Observations span every processed frame within the track's lifespan; a detected frame carries a real `BboxObservation`, while a "predicted-but-undetected" gap frame carries `observation=None`.

Properties:

- `timestamp_microseconds` (int): Timestamp of the sighting in microseconds.
- `observation` (Optional[BboxObservation]): The bounding box and its score at this timestamp, or `None` on a gap frame where the object was present but not detected.

**`VideoAttributePrediction`**
A chosen attribute option together with the scored timestamp ranges over which it held. The same `attribute_id` may appear more than once across an object's life with different options.

Properties:

- `attribute_id` (int): Unique identifier for the attribute.
- `attribute_name` (str): The name of the attribute.
- `option_id` (int): Unique identifier for the chosen option.
- `option_name` (str): The name of the chosen option.
- `timestamp_ranges` (List[ScoredTimestampRange]): The scored spans over which this option was chosen.

**`VideoCategoryPrediction`**
A predicted category and its attribute predictions.

Properties:

- `category_id` (int): Unique identifier for the category.
- `name` (str): The name of the category.
- `score` (float): Confidence score for the category.
- `attributes` (List[VideoAttributePrediction]): Attribute predictions for this category.

**`VideoDetectedObject`**
A single entity tracked across the video: its lifespan, every bounding-box observation, and its categories.

Properties:

- `object_id` (int): Stable identifier for the tracked object.
- `timestamp_ranges` (List[TimestampRange]): The spans over which the object was visible.
- `bbox_observations` (List[VideoBboxObservation]): Bounding-box sightings over time, one per processed frame in the object's lifespan; gap frames carry a null bbox.
- `categories` (List[VideoCategoryPrediction]): Category and attribute predictions for this object.

#### Response types

**`ClassificationPredictImageResponse`**
The response object returned after predicting an image.

Properties:

- `objects` (List[ImageDetectedObject]): Detected objects and their predictions.
- `prediction_task_uuid` (str): The unique identifier for the prediction task.
- `original_file_name` (Optional[str]): The file name of the original media, if provided.

**`ClassificationPredictVideoResponse`**
The response object returned after predicting a video.

Properties:

- `objects` (List[VideoDetectedObject]): Tracked objects and their predictions across the video.
- `frames_per_second` (int): The number of frames per second that were sampled.
- `prediction_task_uuid` (str): The unique identifier for the prediction task.
- `original_file_name` (Optional[str]): The file name of the original media, if provided.

**`PredictionTaskStatusResponse`**
Represents the status of a prediction task.

Properties:

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
