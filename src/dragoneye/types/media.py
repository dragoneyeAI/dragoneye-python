from io import BufferedReader, BytesIO
from typing import BinaryIO, NamedTuple, Union


class Media(NamedTuple):
    file_or_bytes: Union[bytes, BinaryIO, BufferedReader]
    mime_type: str

    def bytes_io(self) -> BytesIO:
        match self.file_or_bytes:
            case bytes():
                return BytesIO(self.file_or_bytes)
            case BytesIO():
                return self.file_or_bytes
            case BufferedReader():
                return BytesIO(self.file_or_bytes.read())
            case BinaryIO():
                return BytesIO(self.file_or_bytes.read())
            case _:  # pyright: ignore [reportUnnecessaryComparison]
                raise ValueError("Invalid image type: Must be bytes or BinaryIO")
