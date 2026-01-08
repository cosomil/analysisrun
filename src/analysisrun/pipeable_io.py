import sys
import traceback
from contextlib import contextmanager
from enum import Enum
from typing import IO, Iterator, NoReturn, Optional

from pydantic import BaseModel, Field

from analysisrun.__env import get_interactivity
from analysisrun.helper import cowsay
from analysisrun.interactive import VirtualFile
from analysisrun.tar import create_tar_from_dict


@contextmanager
def redirect_stdout_to_stderr() -> Iterator[None]:
    """
    標準出力を標準エラー出力にリダイレクトするコンテキストマネージャ。
    
    解析処理中に print() などで標準出力に出力されると、
    tarフォーマットの出力が破損するため、標準出力を標準エラー出力に
    リダイレクトすることで、構造化されたデータを安全に出力できるようにする。
    
    Examples
    --------
    >>> with redirect_stdout_to_stderr():
    ...     print("This goes to stderr")
    ...     # Now safe to write structured data to stdout
    
    Notes
    -----
    このコンテキストマネージャはスレッドセーフではない。
    sys.stdoutの変更はプロセス全体に影響するため、マルチスレッド環境では注意が必要。
    """
    original_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        yield
    finally:
        sys.stdout = original_stdout


class ExitCodes(Enum):
    """
    解析の異常終了時の終了コード
    """

    PROCESSING_ERROR = 1
    INVALID_USAGE = 2


class ErrorResult(BaseModel):
    """
    標準出力を通じて返されるエラー情報
    """

    error: str = Field(description="エラーメッセージ")


def _print(msg: str | bytes, out: IO[bytes]) -> None:
    if isinstance(msg, str):
        out.write((msg + "\n").encode(encoding="utf-8"))
    else:
        out.write(msg)


def exit_with_error(
    code: ExitCodes,
    message: str,
    stdout: IO[bytes],
    stderr: IO[bytes],
    exception: Optional[Exception] = None,
) -> NoReturn:
    """
    解析の異常終了を行う。
    標準エラー出力にエラー情報を出力するほか、スタックトレースやユーザーフレンドリーな
    エラーメッセージを標準エラー出力に出力する。

    Parameters
    ----------
    code
        終了コード
    message
        エラーメッセージ
    stdout
        標準出力ストリーム
    stderr
        標準エラー出力ストリーム
    exception
        発生した例外オブジェクト
    """

    interactivity = get_interactivity()

    # Jupyter notebook環境ではsys.exit()の使用を避けるため、例外をraiseして処理を中断させることとする。
    # その際に例外情報が出力されるので、ここでは例外情報の出力を行わないこととする。
    if exception is not None and interactivity != "notebook":
        details = traceback.format_exception(exception)
        _print("".join(details), stderr)
        stderr.flush()

    if interactivity is None:
        tar_data = create_tar_from_dict(ErrorResult(error=message).model_dump())
        _print(tar_data.getvalue(), stdout)
    else:
        _print(cowsay(message), stdout)
    stdout.flush()

    if interactivity == "notebook":
        raise exception or RuntimeError(message)
    else:
        sys.exit(code.value)


class AnalysisInputModel[
    Params: BaseModel,
    ImageAnalysisResultsInput: BaseModel,
](BaseModel):
    """
    分散実行時に使用される解析(analyze)の入力データモデル

    `read_tar_as_dict`で読み込まれた入力をバリデーションする際に使用される。
    """

    data_name: str = Field(description="解析対象のデータ名")
    sample_name: str = Field(description="解析対象のサンプル名")
    params: Params = Field(description="解析全体に関わるパラメータ")
    image_analysis_results: ImageAnalysisResultsInput = Field(
        description="画像解析結果CSVデータ"
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }


class PostprocessInputModel[
    Params: BaseModel,
](BaseModel):
    """
    分散実行時に使用される後処理(postprocess)の入力データモデル

    `read_tar_as_dict`で読み込まれた入力をバリデーションする際に使用される。
    """

    analysis_results: VirtualFile = Field(description="解析結果データセット")
    params: Params = Field(description="解析全体に関わるパラメータ")

    model_config = {
        "arbitrary_types_allowed": True,
    }
