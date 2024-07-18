from typing import BinaryIO, NamedTuple, Optional, Sequence, Union


class Image(NamedTuple):
    file_or_bytes: Optional[Union[bytes, BinaryIO]] = None
    url: Optional[str] = None


def assert_consistent_data_type(images: Sequence[Image]) -> None:
    assert all(image.file_or_bytes is not None for image in images) ^ all(
        image.url is not None for image in images
    )
