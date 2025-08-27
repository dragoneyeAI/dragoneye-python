from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import BinaryIO, ClassVar, Self, Union


@dataclass(frozen=True)
class Media:
    """Generic binary media + mime_type with conservative, non-destructive access."""

    file_or_bytes: Union[bytes, BytesIO, BinaryIO, BufferedReader]
    mime_type: str

    # Subclasses set this to enforce a family of mimetypes, e.g. ("image/",)
    ACCEPT_PREFIXES: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # Enforce subtype-specific mimetype families when specified.
        if self.ACCEPT_PREFIXES and not any(
            self.mime_type.startswith(p) for p in self.ACCEPT_PREFIXES
        ):
            raise ValueError(
                f"{self.__class__.__name__} requires mime_type starting with "
                f"{' or '.join(self.ACCEPT_PREFIXES)}; got {self.mime_type!r}"
            )

    # ---------- Convenience constructors ----------

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> Self:
        return cls(file_or_bytes=data, mime_type=mime_type)

    @classmethod
    def from_stream(cls, stream: BinaryIO, *, mime_type: str) -> Self:
        """
        Accepts any readable binary stream (e.g., open('file', 'rb')).
        Keeps the stream as-is; reading is deferred to bytes_io().
        """
        return cls(file_or_bytes=stream, mime_type=mime_type)

    @classmethod
    def from_path(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        mime_type: str | None = None,
        guess_from_extension: bool = True,
        read_into_memory: bool = False,
    ) -> Self:
        """
        Create a Media (or subclass) from a filesystem path.

        - `path`: Path to the file on disk.
        - `mime_type`: Explicit mime type. If omitted and `guess_from_extension=True`,
           we'll try to guess from the file extension.
        - `read_into_memory=True`: load file bytes into memory (closes file immediately).
          Otherwise, keep an open file stream.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        mt = mime_type or (
            mimetypes.guess_type(path.name)[0] if guess_from_extension else None
        )
        if mt is None:
            raise ValueError(
                f"mime_type is required for {path} (no extension-based guess available)."
            )

        if read_into_memory:
            data = path.read_bytes()
            return cls(file_or_bytes=data, mime_type=mt)
        else:
            f = path.open("rb")
            return cls(file_or_bytes=f, mime_type=mt)

    # ---------- Utilities ----------

    def bytes_io(self) -> BytesIO:
        """
        Returns a fresh BytesIO with the full content.
        - If we wrap a BytesIO, we non-destructively rewind and copy.
        - If we wrap a stream, we read it (without assuming seekability).
        - If we hold bytes, we just wrap them.
        """
        src = self.file_or_bytes

        if isinstance(src, bytes):
            return BytesIO(src)

        if isinstance(src, BytesIO):
            # Non-destructively copy contents
            pos = src.tell()
            try:
                src.seek(0)
            except Exception:
                pass
            data = src.read()
            try:
                src.seek(pos)
            except Exception:
                pass
            return BytesIO(data)

        # For any readable object with .read()
        if hasattr(src, "read"):
            pos = _tell_safe(src)
            data = src.read()
            _seek_safe(src, pos)
            return BytesIO(data)

        raise TypeError(
            "Invalid media source: expected bytes, BytesIO, or a readable binary stream."
        )

    def size_bytes(self) -> int | None:
        """
        Best-effort size inference without consuming the stream.
        Returns None if size can't be determined cheaply.
        """
        src = self.file_or_bytes
        if isinstance(src, bytes):
            return len(src)
        if isinstance(src, BytesIO):
            pos = src.tell()
            try:
                src.seek(0, os.SEEK_END)
                end = src.tell()
            finally:
                _seek_safe(src, pos)
            return end
        if hasattr(src, "fileno"):
            try:
                return os.fstat(src.fileno()).st_size  # type: ignore[arg-type]
            except Exception:
                return None
        # Path-based size if it looks like a buffered reader with .name
        if hasattr(src, "name"):
            try:
                return Path(src.name).stat().st_size  # type: ignore[arg-type]
            except Exception:
                return None
        return None


@dataclass(frozen=True)
class Image(Media):
    """Media restricted to image/* mimetypes."""

    ACCEPT_PREFIXES: ClassVar[tuple[str, ...]] = ("image/",)


@dataclass(frozen=True)
class Video(Media):
    """Media restricted to video/* mimetypes."""

    ACCEPT_PREFIXES: ClassVar[tuple[str, ...]] = ("video/",)


# ---------- Helpers ----------


def _tell_safe(stream: BinaryIO) -> int | None:
    try:
        if hasattr(stream, "tell"):
            return stream.tell()
    except Exception:
        pass
    return None


def _seek_safe(stream: BinaryIO, pos: int | None) -> None:
    if pos is None:
        return
    try:
        if hasattr(stream, "seek"):
            stream.seek(pos)
    except Exception:
        pass
