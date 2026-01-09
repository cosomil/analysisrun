import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "testdata" / "results"


def _list_files(dir_path: Path) -> list[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file()], key=lambda p: p.name)


def _assert_outputs_match_golden(output_dir: Path) -> None:
    expected = _list_files(GOLDEN_DIR)
    actual = _list_files(output_dir)

    expected_names = {p.name for p in expected}
    actual_names = {p.name for p in actual}

    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        msg = ["出力ファイルの集合が一致しません。"]
        if missing:
            msg.append(f"  missing: {missing}")
        if extra:
            msg.append(f"  extra: {extra}")
        raise AssertionError("\n".join(msg))

    for exp in expected:
        out = output_dir / exp.name
        exp_bytes = exp.read_bytes()
        out_bytes = out.read_bytes()
        if exp_bytes != out_bytes:
            raise AssertionError(f"ファイル内容が一致しません: {exp.name}")


def _run_one(*, mode: str) -> float:
    if mode not in {"sequential", "parallel"}:
        raise ValueError(f"unknown mode: {mode}")

    with tempfile.TemporaryDirectory(prefix=f"analysisrun-bench-{mode}-") as td:
        out_dir = Path(td)

        env = os.environ.copy()
        env["ANALYSISRUN_OUTPUT_DIR"] = str(out_dir)

        # 直列/並列の切り替え（環境変数）
        env["ANALYSISRUN_INTERACTIVITY"] = (
            "notebook" if mode == "sequential" else "terminal"
        )

        cmd = ["uv", "run", "./tests/testimpl"]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        t1 = time.perf_counter()

        if proc.returncode != 0:
            raise RuntimeError(f"実行に失敗しました: {cmd} (exit={proc.returncode})")

        elapsed = t1 - t0

        # 時間を出力した後に一致検証
        print(f"[{mode}] elapsed_sec={elapsed:.6f}")
        _assert_outputs_match_golden(out_dir)
        print(f"[{mode}] outputs_ok=true")

        return elapsed


def main() -> int:
    if not GOLDEN_DIR.exists():
        print(f"golden dir not found: {GOLDEN_DIR}", file=sys.stderr)
        return 2

    seq = _run_one(mode="sequential")
    par = _run_one(mode="parallel")

    # 参考情報（要件外の最小出力）
    if par > 0:
        print(f"speedup(sequential/parallel)={seq / par:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
