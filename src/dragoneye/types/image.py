from typing import BinaryIO, NamedTuple, Optional, Union


class Image(NamedTuple):
    file_or_bytes: Optional[Union[bytes, BinaryIO]] = None
    url: Optional[str] = None
