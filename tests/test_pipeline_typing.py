from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["pipeline_valid.py", "pipeline_invalid.py"])
def test_public_pipeline_types(tmp_path: Path, name: str):
    root = Path(__file__).resolve().parents[1]
    source = root / "tests" / "typing" / name
    config = tmp_path / "pyrightconfig.json"
    config.write_text(
        json.dumps(
            {
                "include": [str(source)],
                "extraPaths": [str(root / "src"), str(root)],
                "pythonVersion": "3.13",
                "typeCheckingMode": "standard",
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "basedpyright",
            "--project",
            str(config),
            "--outputjson",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=60,
        check=False,
    )
    report = json.loads(completed.stdout)
    diagnostics = report["generalDiagnostics"]
    expected = {
        (number, match[1])
        for number, line in enumerate(source.read_text().splitlines(), 1)
        if (match := re.search(r"# error: (\w+)", line))
    }
    actual = {
        (item["range"]["start"]["line"] + 1, item.get("rule")) for item in diagnostics
    }
    assert all(Path(item["file"]) == source for item in diagnostics), diagnostics
    assert actual == expected, diagnostics
    assert len(diagnostics) == len(expected), diagnostics
    assert completed.returncode == bool(expected), completed.stderr
