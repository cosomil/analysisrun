from io import BytesIO
from pathlib import Path

from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile


def assert_sample_csv(v):
    assert v == {
        "Alice": "30",
        "Bob": "25",
        "Charlie": "35",
    }


class Test_read_dict:
    def test_from_PathLike_VirtualFile(self):
        _in = VirtualFile(Path(__file__).parent / "testdata" / "sample.csv")
        out = read_dict(_in, "name", "age")

        assert_sample_csv(out)

    def test_from_FileLike_VirtualFile(self):
        with open(Path(__file__).parent / "testdata" / "sample.csv", "rb") as f:
            content = f.read()
        _in = BytesIO(content)

        out = read_dict(_in, "name", "age")

        assert_sample_csv(out)
