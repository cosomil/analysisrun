from io import BytesIO
from os import getcwd
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from analysisrun.interactive import VirtualFile


class Input(BaseModel):
    target: VirtualFile = Field(description="target file")


def assert_readable_as_csv(v):
    df = pd.read_csv(v)
    assert list(df["sample"]) == ["SampleA", "SampleB"]


class Test_VirtualFile:
    def test_from_Path(self):
        _in: dict[str, Any] = {
            "target": Path(__file__).parent / "testdata" / "samples.csv"
        }
        out = Input(**_in)

        # ディレクトリを取得できる
        assert out.target.parent == (Path(__file__).parent / "testdata")
        # ファイルを読み込むことができる
        assert_readable_as_csv(out.target.unwrap())

    def test_from_str(self):
        _in: dict[str, Any] = {
            "target": str(Path(__file__).parent / "testdata" / "samples.csv")
        }
        out = Input(**_in)

        # ディレクトリを取得できる
        assert out.target.parent == (Path(__file__).parent / "testdata")
        # ファイルを読み込むことができる
        assert_readable_as_csv(out.target.unwrap())

    def test_from_quoted_str(self):
        _in: dict[str, Any] = {
            "target": f"'{Path(__file__).parent / 'testdata' / 'samples.csv'}'"
        }
        out = Input(**_in)

        # ディレクトリを取得できる
        assert out.target.parent == (Path(__file__).parent / "testdata")
        # ファイルを読み込むことができる
        assert_readable_as_csv(out.target.unwrap())

    def test_from_BytesIO(self):
        with open(Path(__file__).parent / "testdata" / "samples.csv", "rb") as f:
            content = f.read()

        _in: dict[str, Any] = {"target": BytesIO(content)}
        out = Input(**_in)

        # ディレクトリを取得できる（作業ディレクトリが入る）
        assert out.target.parent == Path(getcwd())
        # ファイルを読み込むことができる
        assert_readable_as_csv(out.target.unwrap())
