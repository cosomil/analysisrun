"""QC サンプルと同じ公開経路で、画像出力直後の障害を注入する。"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
example = importlib.import_module("example.qc-feedback.__main__")
original = example.apply_qc_evaluation


def fail_after_output(args):  # type: ignore
    result = original(args)
    # 入力で指定したレーン自身が画像を出してから失敗するため、
    # Worker 間の実行順や sleep に依存せず部分出力を検証できる。
    if args.sample_name == "FAIL_AFTER_OUTPUT":
        raise LookupError("QC report failure after output")
    return result


example.apply_qc_evaluation = fail_after_output  # pyright: ignore[reportAttributeAccessIssue]
example.main()
