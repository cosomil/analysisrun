import sys
from io import BytesIO

import pytest

from analysisrun.pipeable_io import redirect_stdout_to_stderr


def test_redirect_stdout_to_stderr():
    """
    redirect_stdout_to_stderr関数が正しくstderrパラメータを使用することを確認する。
    """

    # stderrとしてBytesIOを用意
    stderr_buf = BytesIO()

    # print文の出力先を確認
    with redirect_stdout_to_stderr(stderr_buf):
        print("Test message")
        sys.stdout.flush()

    # stderrに出力されていることを確認
    stderr_buf.seek(0)
    output = stderr_buf.read().decode("utf-8")
    assert "Test message" in output


@pytest.mark.parametrize("interactivity", ["notebook", "terminal"])
def test_human_input_errors_use_stderr_traceback_then_cowsay(
    monkeypatch, interactivity
):
    from analysisrun import pipeable_io

    monkeypatch.setattr(pipeable_io, "get_interactivity", lambda: interactivity)
    stdout, stderr = BytesIO(), BytesIO()
    error = ValueError("input failure")
    with pytest.raises(
        ValueError if interactivity == "notebook" else SystemExit
    ) as caught:
        pipeable_io.exit_with_error(
            pipeable_io.ExitCodes.INVALID_USAGE,
            "friendly message",
            stdout,
            stderr,
            error,
        )
    if interactivity == "notebook":
        assert caught.value is error
    else:
        assert caught.value.code == 2
    assert stdout.getvalue() == b""
    diagnostics = stderr.getvalue().decode()
    assert diagnostics.index("ValueError: input failure") < diagnostics.index(
        "friendly message"
    )
