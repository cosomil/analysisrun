"""QC サンプルの streaming クライアント。リポジトリのルートで実行する。"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path


def make_request(*, fail: bool = False) -> bytes:
    # Value=0 は QC サンプルの reference > 0 の検証で解析エラーになる。
    value = 0 if fail else 10
    entries = {
        "params": (b'{"minimum_score":0.8}', False),
        "sample_names": (b"data,sample\n0000,QC\n", True),
        "image_analysis_results/measurements": (
            (
                "MultiPointIndex,Entity,Value,Filename\n"
                f"1,Activity Spots,{value},Measurement_000_0000.nd2\n"
            ).encode(),
            True,
        ),
    }
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, (content, is_file) in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            if is_file:
                info.pax_headers = {"is_file": "true"}
            archive.addfile(info, BytesIO(content))
    return buffer.getvalue()


def read_response(payload: bytes, status: int) -> dict[str, bytes]:
    # gzip の CRC/footer を含めて検証する。tarfile 単独では厳密な終端検証にならない。
    raw = gzip.decompress(payload)
    entries: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if not member.isfile() or member.name in entries:
                raise ValueError(f"invalid response entry: {member.name}")
            content = archive.extractfile(member)
            if content is None:
                raise ValueError(f"unreadable response entry: {member.name}")
            entries[member.name] = content.read()
        end = archive.offset
    if len(raw) % 512 or len(raw) < end + 1024 or any(raw[end:]):
        raise ValueError("incomplete or corrupt TAR ending")
    if status != 0 or "error" in entries:
        message = entries.get("error", b"no error entry").decode("utf-8")
        raise RuntimeError(f"analysis failed (exit={status}): {message}")
    if "result_csv" not in entries:
        raise ValueError("missing result_csv")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail", action="store_true", help="解析エラーを再現する")
    args = parser.parse_args()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ANALYSISRUN_") and key != "PSEUDO_NBENV"
    }
    env.update(ANALYSISRUN_MAX_WORKERS="2", MPLBACKEND="Agg")
    completed = subprocess.run(
        [sys.executable, "-m", "example.qc-feedback"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        input=make_request(fail=args.fail),
        capture_output=True,
        check=False,
        timeout=60,
    )
    sys.stderr.buffer.write(completed.stderr)
    try:
        entries = read_response(completed.stdout, completed.returncode)
    except (OSError, EOFError, tarfile.TarError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    for name, content in entries.items():
        print(name, f"({len(content)} bytes)")
        if name.endswith("/result_json"):
            print(json.loads(content))


if __name__ == "__main__":
    main()
