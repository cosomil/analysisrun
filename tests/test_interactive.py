from io import BytesIO
from os import getcwd
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from analysisrun import interactive
from analysisrun.interactive import VirtualFile, scan_model_input


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

    def test_from_double_quoted_str(self):
        path = Path(__file__).parent / "testdata" / "samples.csv"
        _in: dict[str, Any] = {"target": f'"{path}"'}
        out = Input(**_in)

        # ディレクトリを取得できる
        assert out.target.parent == (Path(__file__).parent / "testdata")
        # ファイルを読み込むことができる
        assert_readable_as_csv(out.target)

    def test_from_BytesIO(self):
        with open(Path(__file__).parent / "testdata" / "samples.csv", "rb") as f:
            content = f.read()

        _in: dict[str, Any] = {"target": BytesIO(content)}
        out = Input(**_in)

        # ディレクトリを取得できる（作業ディレクトリが入る）
        assert out.target.parent == Path(getcwd())
        # ファイルを読み込むことができる
        assert_readable_as_csv(out.target.unwrap())


class _FakeQuestion:
    def __init__(self, answers: list[str], validate):
        self._answers = answers
        self._validate = validate

    def ask(self):
        while self._answers:
            answer = self._answers.pop(0)
            if self._validate is None or self._validate(answer) is True:
                return answer
        raise AssertionError("有効な入力が不足しています")


def _patch_questionary_text(monkeypatch: pytest.MonkeyPatch, answers: list[str]):
    def fake_text(message: str, default: str = "", validate=None, **kwargs):
        return _FakeQuestion(answers, validate)

    monkeypatch.setattr(interactive.questionary, "text", fake_text)


def _patch_questionary_path(monkeypatch: pytest.MonkeyPatch, answers: list[str]):
    def fake_path(message: str, default: str = "", validate=None, **kwargs):
        return _FakeQuestion(answers, validate)

    monkeypatch.setattr(interactive.questionary, "path", fake_path)


def main():
    class Nested(BaseModel):
        first_name: str = Field(description="Your first name", min_length=1)
        last_name: str = Field(description="Your last name", min_length=1)

    class NT(NamedTuple):
        a: int = 123
        b: int = 456

    class Input(BaseModel):
        you: Nested = Field(description="Your information")
        target: VirtualFile = Field(
            description="target file", default=VirtualFile(Path("AGENTS.md"))
        )
        tup: tuple[int, int] = Field(description="A pair of integers", default=(88, 22))
        nt: NT = Field(description="A named tuple")

    d = scan_model_input(Input)
    print(d)


if __name__ == "__main__":
    main()
