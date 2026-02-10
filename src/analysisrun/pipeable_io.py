import io
import sys
import tarfile
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
def redirect_stdout_to_stderr(stderr: IO[bytes]) -> Iterator[None]:
    """
    標準出力を標準エラー出力にリダイレクトするコンテキストマネージャ。

    解析処理中に print() などで標準出力に出力されると、
    tarフォーマットの出力が破損するため、標準出力を標準エラー出力に
    リダイレクトすることで、構造化されたデータを安全に出力できるようにする。

    Parameters
    ----------
    stderr
        標準エラー出力ストリーム（バイナリモード）

    Examples
    --------
    >>> with redirect_stdout_to_stderr(sys.stderr.buffer):
    ...     print("This goes to stderr")
    ...     # Now safe to write structured data to stdout

    Notes
    -----
    このコンテキストマネージャはスレッドセーフではない。
    sys.stdoutの変更はプロセス全体に影響するため、マルチスレッド環境では
    複数のスレッドが同時にsys.stdoutを変更しようとすると競合状態が発生する可能性がある。
    このコンテキストマネージャはシングルスレッド環境での使用を想定している。
    """
    original_stdout = sys.stdout
    stderr_text_wrapper = None
    try:
        # stderr is IO[bytes], but sys.stdout needs a text stream
        # Wrap it with TextIOWrapper to make it compatible
        # Set closefd=False to prevent closing the underlying buffer
        stderr_text_wrapper = io.TextIOWrapper(
            stderr, encoding="utf-8", line_buffering=True, write_through=True
        )
        # Prevent the wrapper from closing the underlying buffer
        stderr_text_wrapper._CHUNK_SIZE = 1  # Force immediate writes # type: ignore
        sys.stdout = stderr_text_wrapper
        yield
    finally:
        # Flush before restoring to ensure all output is written
        if stderr_text_wrapper and hasattr(stderr_text_wrapper, "flush"):
            try:
                stderr_text_wrapper.flush()
            except (ValueError, OSError):
                pass  # Ignore errors if already closed
        # Detach the wrapper without closing the underlying buffer
        if stderr_text_wrapper:
            try:
                stderr_text_wrapper.detach()
            except (ValueError, OSError):
                pass  # Ignore errors if already detached
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


def exit_with_error_streaming(
    code: ExitCodes,
    message: str,
    tar: tarfile.TarFile,
    stderr: IO[bytes],
    exception: Optional[Exception] = None,
) -> NoReturn:
    """
    ストリーミングTAR用のエラー終了処理。
    ErrorResult仕様に従い、TARに"error"エントリを追加する。

    sys.exit()を呼ぶが、with文の__exit__が必ず実行されるため
    TARの正常なクローズが保証される。

    Parameters
    ----------
    code
        終了コード
    message
        エラーメッセージ
    tar
        出力先のTARストリーム
    stderr
        標準エラー出力ストリーム
    exception
        発生した例外オブジェクト
    """
    # スタックトレースを標準エラー出力に出力
    if exception is not None:
        details = traceback.format_exception(exception)
        _print("".join(details), stderr)
        stderr.flush()

    # ErrorResult仕様に従ってTARエントリを作成
    # create_tar_from_dict(ErrorResult(error=message).model_dump()) と同じ構造
    error_content = message.encode("utf-8")
    tar_info = tarfile.TarInfo(name="error")
    tar_info.size = len(error_content)
    tar.addfile(tar_info, io.BytesIO(error_content))

    # sys.exit()を呼ぶ
    # SystemExitが発生し、with文の__exit__が呼ばれてTARがクローズされる
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


class AnalyzeSeqInputModel[
    Params: BaseModel,
    ImageAnalysisResultsInput: BaseModel,
](BaseModel):
    """
    分散実行時に複数ターゲットの解析をシーケンシャルに実行する際の入力データモデル

    `read_tar_as_dict`で読み込まれた入力をバリデーションする際に使用される。

    Examples
    --------
    >>> # 入力tar構造:
    >>> {
    ...     "targets": {"0001": "SampleA", "0002": "SampleB"},
    ...     "params": {...},
    ...     "image_analysis_results": {
    ...         "activity_spots": BytesIO(...),
    ...         "surrounding_spots": BytesIO(...),
    ...     }
    ... }
    """

    targets: dict[str, str] = Field(
        description="解析対象データ情報。keyはdata_name、valueはsample_name"
    )
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

    analysis_results: dict[str, VirtualFile] = Field(description="解析結果データセット")
    params: Params = Field(description="解析全体に関わるパラメータ")

    model_config = {
        "arbitrary_types_allowed": True,
    }


def list_from_dict[V](d: dict[str, V]) -> list[V]:
    return [v for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
