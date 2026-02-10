import tarfile
from io import BytesIO

from pydantic import BaseModel

from analysisrun.interactive import VirtualFile
from analysisrun.tar import FileIO, create_tar_from_dict, read_tar_as_dict


def test_read_tar_as_dict():
    class Parameter(BaseModel):
        covariance: float
        max_iterations: int

    class Input(BaseModel):
        count: int
        threshold: float
        data: VirtualFile
        parameters: Parameter

    fileData = b"name,age\nAlice,30\nBob,25\nCharlie,35\n"
    buf = create_tar_from_dict(
        {
            "count": 10,
            "threshold": 0.5,
            "data": BytesIO(fileData),
            "parameters/covariance": 0.1,
            "parameters/max_iterations": 100,
        }
    )

    got = read_tar_as_dict(buf)

    assert got["count"] == "10"
    assert got["threshold"] == "0.5"
    data = got["data"]
    assert isinstance(data, BytesIO)
    assert data.read() == fileData
    data.seek(0)
    assert got["parameters"]["covariance"] == "0.1"
    assert got["parameters"]["max_iterations"] == "100"

    # pydanticで期待した通りに変換できる。
    v = Input(**got)
    assert v.count == 10
    assert v.threshold == 0.5
    assert v.data.read() == fileData
    assert v.parameters.covariance == 0.1
    assert v.parameters.max_iterations == 100


def test_create_tar_from_dict_default_is_gzip_and_readable():
    payload = {
        "count": 10,
        "data": BytesIO(b"abc"),
    }

    buf = create_tar_from_dict(payload)

    # gzipマジックナンバー (0x1f, 0x8b)
    head = buf.getvalue()[:2]
    assert head == b"\x1f\x8b"

    got = read_tar_as_dict(BytesIO(buf.getvalue()))
    assert got["count"] == "10"
    data = got["data"]
    assert isinstance(data, BytesIO)
    assert data.read() == b"abc"


def test_create_tar_from_dict_gzip_false_is_plain_tar_and_readable():
    payload = {
        "count": 10,
        "data": BytesIO(b"abc"),
    }

    buf = create_tar_from_dict(payload, gzip=False)

    # 非圧縮tarなのでgzipマジックではない
    head = buf.getvalue()[:2]
    assert head != b"\x1f\x8b"

    got = read_tar_as_dict(BytesIO(buf.getvalue()))
    assert got["count"] == "10"
    data = got["data"]
    assert isinstance(data, BytesIO)
    assert data.read() == b"abc"


def _make_plain_tar_with_one_file(
    *,
    name: str,
    content: bytes,
    pax_headers: dict[str, str] | None = None,
) -> BytesIO:
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        if pax_headers is not None:
            info.pax_headers = pax_headers
        tar.addfile(info, BytesIO(content))
    buf.seek(0)
    return buf


def test_read_tar_as_dict_returns_fileio_and_preserves_pax_headers_when_is_file_set():
    buf = _make_plain_tar_with_one_file(
        name="data.bin",
        content=b"\x00\x01\x02",
        pax_headers={
            "is_file": "true",
            "x-meta": "hello",
        },
    )

    got = read_tar_as_dict(buf)
    data = got["data.bin"]

    assert isinstance(data, FileIO)
    assert data.read() == b"\x00\x01\x02"
    assert data.headers["is_file"] == "true"
    assert data.headers["x-meta"] == "hello"


def test_read_tar_as_dict_without_is_file_decodes_as_string():
    buf = _make_plain_tar_with_one_file(
        name="note.txt",
        content=b"  hello\n",
        pax_headers={},
    )

    got = read_tar_as_dict(buf)
    assert got["note.txt"] == "hello"


def test_create_tar_from_dict_roundtrips_fileio_with_arbitrary_pax_headers():
    src = FileIO({"is_file": "true", "x-foo": "bar"}, b"abc")
    buf = create_tar_from_dict({"data": src})

    got = read_tar_as_dict(BytesIO(buf.getvalue()))
    data = got["data"]

    assert isinstance(data, FileIO)
    assert data.read() == b"abc"
    assert data.headers["is_file"] == "true"
    assert data.headers["x-foo"] == "bar"


def test_read_tar_as_dict_raises_on_conflicting_paths_file_then_nested():
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        # First: foo is a file
        foo = tarfile.TarInfo(name="foo")
        foo.size = 1
        tar.addfile(foo, BytesIO(b"x"))

        # Then: foo/bar requires foo to be a dict
        foobar = tarfile.TarInfo(name="foo/bar")
        foobar.size = 1
        tar.addfile(foobar, BytesIO(b"y"))
    buf.seek(0)

    try:
        read_tar_as_dict(buf)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "requires foo to be a dictionary" in str(e)
