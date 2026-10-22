"""Typed helpers for assertions shared by archive integration tests."""

import tarfile


def read_tar_member(archive: tarfile.TarFile, member: str | tarfile.TarInfo) -> bytes:
    stream = archive.extractfile(member)
    assert stream is not None, f"expected a regular TAR file: {member}"
    return stream.read()
