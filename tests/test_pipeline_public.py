import json
import os
import subprocess
import sys
import tarfile
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from analysisrun import (
    ExecutionContext,
    InitialPipeline,
    PipelineBuilder,
    pipeable,
    read_context,
)
from tests.test_pipeable import (
    IMAGE_ANALYSIS_RESULT_CSV,
    SAMPLES_CSV,
    ImageResults,
    ManualInput,
    Params,
    _build_streaming_input_tar,
    _force_interactivity,
)


def context(monkeypatch, tmp_path):
    _force_interactivity(monkeypatch, "notebook")
    return read_context(
        Params,
        ImageResults,
        manual_input=ManualInput(
            params=Params(),
            sample_names=SAMPLES_CSV,
            image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        ),
        output_dir=tmp_path,
    )


def test_builders_are_lazy_immutable_and_reusable(monkeypatch, tmp_path):
    ctx = context(monkeypatch, tmp_path)
    assert isinstance(ctx, ExecutionContext)
    initial = ctx.pipeline()
    assert isinstance(initial, InitialPipeline)
    assert not isinstance(initial, PipelineBuilder)
    calls = []

    def prepare(args):
        calls.append("prepare")
        return args.image_analysis_results

    def mapping(args):
        assert args.value is args.synthesized is None
        return args.params.threshold

    branch = initial.preprocess(prepare).map(mapping)
    other = initial.synthesize(lambda a: 3)
    assert calls == []
    with pytest.raises(FrozenInstanceError):
        branch._stages = ()
    for builder in [initial.preprocess(prepare), initial.map(mapping), other]:
        with pytest.raises(AttributeError):
            builder.preprocess(prepare)
    assert calls == []
    for _ in range(2):
        assert branch.finalize(
            lambda a: pd.DataFrame({"v": list(a.values.values())})
        ).v.tolist() == [1, 1]
    assert calls == ["prepare", "prepare"]
    assert other.finalize(
        lambda a: pd.DataFrame({"v": [a.synthesized]})
    ).v.tolist() == [3]
    assert initial.finalize(
        lambda a: pd.DataFrame({"v": list(a.values.values())})
    ).v.tolist() == [None, None]
    with pytest.raises(TypeError, match="finalize must return"):
        initial.finalize(lambda a: 1)


def test_public_pipeline_consecutive_stages(monkeypatch, tmp_path):
    ctx = context(monkeypatch, tmp_path)
    result = (
        ctx.pipeline()
        .map(lambda a: 2)
        .map(lambda a: str(a.value))
        .synthesize(lambda a: 3)
        .synthesize(lambda a: a.synthesized + 1)
        .map(lambda a: int(a.value) + a.synthesized)
        .finalize(lambda a: pd.DataFrame({"v": list(a.values.values())}))
    )
    assert result.v.tolist() == [6, 6]
    result = (
        ctx.pipeline()
        .synthesize(lambda a: 3)
        .map(lambda a: a.synthesized if a.value is None else 0)
        .finalize(lambda a: pd.DataFrame({"v": list(a.values.values())}))
    )
    assert result.v.tolist() == [3, 3]


def test_worker_read_context_skips_all_parent_input_loading(monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MODE", "pipeline-worker")
    monkeypatch.setattr(
        pipeable, "_get_image_analysis_specs", lambda *a: pytest.fail("read specs")
    )
    ctx = read_context(Params, ImageResults, stdin=BytesIO(b"not a TAR"))
    assert ctx.mode == "pipeline-worker"
    assert ctx._pipeline_inputs is None
    assert isinstance(ctx.pipeline(), InitialPipeline)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("preprocess", [False, True])
def test_public_entrypoint_end_to_end(tmp_path, streaming, preprocess):
    env = {**os.environ, "TEST_OUTPUT": str(tmp_path), "ANALYSISRUN_MAX_WORKERS": "2"}
    env["TEST_PREPROCESS"] = str(int(preprocess))
    env.pop("ANALYSISRUN_MODE", None)
    env.pop("PSEUDO_NBENV", None)
    if streaming:
        env.pop("ANALYSISRUN_INTERACTIVITY", None)
    else:
        env["ANALYSISRUN_INTERACTIVITY"] = "terminal"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("public_pipeline_entry.py"))],
        input=_build_streaming_input_tar(
            [("0000", "A"), ("0001", "B")], Params(threshold=2)
        ).getvalue()
        if streaming
        else b"",
        capture_output=True,
        check=False,
        env=env,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    expected = [{"data": "0000", "score": 6}, {"data": "0001", "score": 6}]
    if streaming:
        with tarfile.open(fileobj=BytesIO(completed.stdout), mode="r:gz") as archive:
            entries = {
                member.name: archive.extractfile(member).read()
                for member in archive
                if member.isfile()
            }
        assert any(name.endswith("0000.png") for name in entries)
        assert any(name.endswith("0001.png") for name in entries)
        assert not (tmp_path / "result.json").exists()
        # Result CSVs use the existing streaming serialization contract.
        assert "result_csv" in entries
        result = pd.read_csv(BytesIO(entries["result_csv"]), dtype={"data": str})
        assert result.to_dict("records") == expected
    else:
        assert json.loads((tmp_path / "result.json").read_text()) == expected
        assert (tmp_path / "0000.png").exists()
        assert (tmp_path / "0001.png").exists()
