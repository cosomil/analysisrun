from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import analysisrun as ar
from analysisrun import helper
from analysisrun.helper import read_dict
from analysisrun.interactive import VirtualFile


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 8, 22, 6, 30, 15, 123456, tzinfo=UTC)
        return value.astimezone(tz)


def test_get_utc_timestamp(monkeypatch):
    monkeypatch.setattr(helper, "datetime", FrozenDatetime)

    assert ar.get_utc_timestamp() == "2026-08-22T06:30:15.123456+00:00"


def test_get_jst_timestamp(monkeypatch):
    monkeypatch.setattr(helper, "datetime", FrozenDatetime)

    assert ar.get_jst_timestamp() == "2026-08-22T15:30:15.123456+09:00"


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
