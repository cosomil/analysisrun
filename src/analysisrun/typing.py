from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Iterator,
    Protocol,
    SupportsIndex,
    Tuple,
    overload,
    runtime_checkable,
)


@runtime_checkable
class NamedTupleLike[E](Protocol):
    _fields: Tuple[str, ...]
    _field_defaults: dict[str, Any]

    def __init__(self, *args): ...
    def __iter__(self) -> Iterator[E]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> E: ...
    @overload
    def __getitem__(self, index: slice, /) -> tuple[E, ...]: ...
    def _asdict(self) -> dict[str, Any]: ...


VirtualFileLike = str | Path | BinaryIO
