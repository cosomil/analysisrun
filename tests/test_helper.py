from io import BytesIO
from pathlib import Path

from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile


def assert_sample_csv(v):
    assert v == {
        "0000": "SampleA",
        "0001": "SampleB",
    }


class Test_read_dict:
    def test_from_PathLike_VirtualFile(self):
        _in = VirtualFile(Path(__file__).parent / "testdata" / "samples.csv")
        out = read_dict(_in, "data", "sample")

        assert_sample_csv(out)

    def test_from_FileLike_VirtualFile(self):
        with open(Path(__file__).parent / "testdata" / "samples.csv", "rb") as f:
            content = f.read()
        _in = BytesIO(content)

        out = read_dict(_in, "data", "sample")

        assert_sample_csv(out)
