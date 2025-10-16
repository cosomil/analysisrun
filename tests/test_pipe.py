from io import BytesIO

from pydantic import BaseModel

from analysisrun.__pipe import create_tar_from_dict, read_tar_as_dict
from analysisrun.interactive import VirtualFile


def test_read_tar_as_dict():
    class Input(BaseModel):
        count: int
        threshold: float
        data: VirtualFile

    fileData = b"name,age\nAlice,30\nBob,25\nCharlie,35\n"
    buf = create_tar_from_dict(
        {
            "count": 10,
            "threshold": 0.5,
            "data.csv": BytesIO(fileData),
        }
    )

    got = read_tar_as_dict(buf)

    assert got["count"] == "10"
    assert got["threshold"] == "0.5"
    data = got["data"]
    assert isinstance(data, BytesIO)
    assert data.read() == fileData
    data.seek(0)

    # pydanticで期待した通りに変換できる。
    v = Input(**got)
    assert v.count == 10
    assert v.threshold == 0.5
    assert v.data.read() == fileData
