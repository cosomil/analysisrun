import os
import sys
from pathlib import Path


def is_notebook_env() -> bool:
    """
    Jupyter Notebook環境かどうかを判定する。
    PSEUDO_NBENV環境変数が設定されている場合は、擬似的にノートブック環境とみなす。
    """

    if os.getenv("PSEUDO_NBENV"):
        return True

    try:
        env_name = get_ipython().__class__.__name__  # type: ignore
    except NameError:
        return False

    if env_name == "TerminalInteractiveShell":
        # IPython shell
        return False
    # Jupyter Notebook (env_name == 'ZMQInteractiveShell')
    return True


def get_interactivity():
    """
    対話的な実行環境である場合、その環境を識別する"notebook"または"terminal"を返す。
    """

    forced = os.getenv("ANALYSISRUN_INTERACTIVITY")
    if forced in ("notebook", "terminal"):
        return forced

    if is_notebook_env():
        return "notebook"
    elif sys.stdin.isatty():
        return "terminal"
    else:
        return None


def get_entrypoint() -> Path | None:
    """
    スクリプトのエントリポイントのパスを取得する。
    """

    main = sys.modules.get("__main__")
    if main and hasattr(main, "__file__") and main.__file__:
        return Path(main.__file__).resolve()
    return None
