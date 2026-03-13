from io import BytesIO
import sys

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
