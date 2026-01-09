from io import BytesIO

from pydantic import BaseModel

from analysisrun.tar import create_tar_from_dict, read_tar_as_dict
from analysisrun.interactive import VirtualFile


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
