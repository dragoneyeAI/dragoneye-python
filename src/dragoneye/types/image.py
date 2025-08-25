from io import BytesIO
from typing import BinaryIO, NamedTuple, Optional, Sequence, Union


class Image(NamedTuple):
    file_or_bytes: Optional[Union[bytes, BinaryIO]] = None
    url: Optional[str] = None

    def bytes_io_x(self) -> BytesIO:
        assert self.file_or_bytes is not None, (
            "file_or_bytes is required to fetch BytesIO object"
        )
        match self.file_or_bytes:
            case bytes():
                return BytesIO(self.file_or_bytes)
            case BytesIO():
                return self.file_or_bytes
            case BinaryIO():
                return BytesIO(self.file_or_bytes.read())
            case _:  # pyright: ignore [reportUnnecessaryComparison]
                raise ValueError("Invalid image type: Must be bytes or BinaryIO")


def assert_consistent_data_type(images: Sequence[Image]) -> None:
    assert all(image.file_or_bytes is not None for image in images) ^ all(
        image.url is not None for image in images
    )
