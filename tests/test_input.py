from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from pydantic import BaseModel

from analysisrun import (
    ManualInput,
    dropna,
    entity_filter,
    image_analysis_result_spec,
    read_context,
)
from analysisrun.input import create_image_analysis_results_input_model
from analysisrun.tar import create_tar_from_dict, read_tar_as_dict

DATA_DIR = Path(__file__).parent / "testdata"
IMAGE_ANALYSIS_RESULT_CSV = DATA_DIR / "image_analysis_result.csv"
SAMPLES_CSV = DATA_DIR / "samples.csv"


class Params(BaseModel):
    threshold: int = 1


class ImageResults(NamedTuple):
    activity_spots: pd.DataFrame = image_analysis_result_spec(
        description="Activity spots",
        cleansing=entity_filter("Activity Spots"),
    )


def _dump_csv(df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _force_interactivity(monkeypatch, value: str | None) -> None:
    """read_context と exit_with_error の両方で同じ interactivity を返すように固定する。"""

    from analysisrun import env, pipeable_io

    monkeypatch.setattr(env, "get_interactivity", lambda: value)
    monkeypatch.setattr(pipeable_io, "get_interactivity", lambda: value)


def _build_streaming_input_tar(
    rows: list[tuple[str, str]],
    params: Params | None = None,
) -> BytesIO:
    sample_buf = _dump_csv(pd.DataFrame(rows, columns=["data", "sample"]))
    image_buf = _dump_csv(pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV))
    return create_tar_from_dict(
        {
            "params": (params or Params()).model_dump_json(),
            "sample_names": sample_buf,
            "image_analysis_results/activity_spots": image_buf,
        }
    )


def test_create_image_analysis_results_input_model_requires_spec():
    class InvalidImageResults(NamedTuple):
        activity_spots: pd.DataFrame

    with pytest.raises(TypeError):
        create_image_analysis_results_input_model(InvalidImageResults)


def test_dropna_removes_rows_with_missing_values_in_specified_columns():
    data = pd.DataFrame(
        {
            "foo_Sum": [1.0, None, 3.0, 4.0],
            "bar_Sum": [1.0, 2.0, None, 4.0],
            "unrelated": [None, "value", "value", None],
        }
    )

    result = dropna(["foo_Sum", "bar_Sum"])(data)

    assert list(result._data.index) == [0, 3]


def test_dropna_accepts_single_column_name():
    data = pd.DataFrame(
        {
            "foo_Sum": [1.0, None, 3.0],
            "bar_Sum": [None, 2.0, 3.0],
        }
    )

    result = dropna("foo_Sum")(data)

    assert list(result._data.index) == [0, 2]


def test_dropna_can_follow_another_cleansing_function():
    data = pd.DataFrame(
        {
            "Entity": ["Activity Spots", "Nuclei", "Activity Spots"],
            "foo_Sum": [1.0, None, None],
        }
    )

    filtered = entity_filter("Activity Spots")(data)
    result = dropna(["foo_Sum"])(filtered)

    assert list(result._data.index) == [0]


def test_dropna_requires_at_least_one_column():
    with pytest.raises(ValueError, match="column must contain at least one column"):
        dropna([])


def test_image_analysis_result_spec_accepts_cleansing_list():
    spec = image_analysis_result_spec(
        description="Activity spots",
        cleansing=[entity_filter("Activity Spots"), dropna(["Value"])],
    )

    data = pd.DataFrame(
        {
            "Entity": ["Activity Spots", "Activity Spots", "Nuclei"],
            "Value": [1.0, None, 3.0],
        }
    )
    result = data
    for cleansing in spec.cleansing:
        result = cleansing(result)

    assert list(result._data.index) == [0]


def test_read_context_showschema_outputs_streaming_input_schema(monkeypatch):
    monkeypatch.setenv("ANALYSISRUN_MODE", "showschema")

    stdout_buf = BytesIO()

    with pytest.raises(SystemExit) as excinfo:
        read_context(Params, ImageResults, stdout=stdout_buf)

    assert excinfo.value.code == 0

    schema = json.loads(stdout_buf.getvalue().decode("utf-8"))
    assert schema["schema_version"] == "1"
    assert schema["transport"] == {
        "type": "tar",
        "compression": ["tar", "tar.gz"],
        "path_separator": "/",
    }

    tar_entries = {entry["path"]: entry for entry in schema["tar_entries"]}
    assert tar_entries["params"]["required"] is True
    assert tar_entries["params"]["content_type"] == "application/json"
    assert tar_entries["params"]["pax_headers"] == {}
    assert tar_entries["params"]["json_schema"]["type"] == "object"
    assert tar_entries["params"]["json_schema"]["properties"]["threshold"]["type"] == (
        "integer"
    )

    assert tar_entries["sample_names"] == {
        "path": "sample_names",
        "required": True,
        "content_type": "text/csv",
        "description": "サンプル名CSVファイル（サンプル名とレーン番号の対応表）",
        "pax_headers": {"is_file": "true"},
    }
    assert tar_entries["image_analysis_results/activity_spots"] == {
        "path": "image_analysis_results/activity_spots",
        "required": True,
        "content_type": "text/csv",
        "description": "Activity spots",
        "pax_headers": {"is_file": "true"},
    }


def test_read_context_noninteractive_invalid_stdin_tar_outputs_error_tar(monkeypatch):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()
    with pytest.raises(SystemExit) as excinfo:
        read_context(
            Params, ImageResults, stdin=BytesIO(b"not a tar"), stdout=stdout_buf
        )

    assert excinfo.value.code == 2
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert (
        tar_result["error"]
        == "入力データの読み込みに失敗しました。入力形式を確認してください。"
    )


@pytest.mark.parametrize("method", ["analyze", "postprocess"])
def test_read_context_unsupported_distributed_mode_outputs_error_tar(
    monkeypatch, method: str
):
    monkeypatch.setenv("ANALYSISRUN_MODE", method)
    monkeypatch.delenv("PSEUDO_NBENV", raising=False)
    _force_interactivity(monkeypatch, None)

    stdout_buf = BytesIO()
    with pytest.raises(SystemExit) as excinfo:
        read_context(Params, ImageResults, stdin=BytesIO(b""), stdout=stdout_buf)

    assert excinfo.value.code == 2
    tar_result = read_tar_as_dict(BytesIO(stdout_buf.getvalue()))
    assert tar_result["error"] == f"未対応のANALYSISRUN_MODEです: {method}"


def test_read_context_notebook_requires_manual_input(monkeypatch):
    """notebook 環境では manual_input 指定が必須。"""

    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    _force_interactivity(monkeypatch, "notebook")

    with pytest.raises(RuntimeError) as excinfo:
        read_context(Params, ImageResults, manual_input=None)

    assert "Jupyter notebook環境ではmanual_inputの指定が必須" in str(excinfo.value)


def _manual_input():
    return ManualInput(
        params=Params(threshold=3),
        image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        sample_names=SAMPLES_CSV,
    )


@pytest.mark.parametrize(
    "interactivity, mode",
    [("notebook", "notebook"), ("terminal", "interactive"), (None, "streaming")],
)
def test_context_preserves_input_protocol_and_prepares_once(
    monkeypatch, tmp_path, interactivity, mode
):
    from analysisrun import context as context_module

    _force_interactivity(monkeypatch, interactivity)
    monkeypatch.setattr(context_module, "get_entrypoint", lambda: Path(__file__))
    ctx = read_context(
        Params,
        ImageResults,
        manual_input=_manual_input(),
        output_dir=tmp_path,
        stdin=_build_streaming_input_tar(
            [("0001", "B"), ("0000", "A")], Params(threshold=9)
        ),
    )
    assert ctx.mode == mode
    assert ctx.process_role == "parent"
    assert ctx.params.threshold == (9 if mode == "streaming" else 3)
    assert ctx.targets == (
        {"0001": "B", "0000": "A"}
        if mode == "streaming"
        else {"0000": "SampleA", "0001": "SampleB"}
    )
    frame = ctx.image_analysis_results.activity_spots
    assert frame.Entity.unique().tolist() == ["Activity Spots"]
    assert frame.Data.unique().tolist() == ["0000", "0001"]
    assert frame.ImageAnalysisMethod.unique().tolist() == ["ImgAnalysisV1"]
    assert not hasattr(ctx, "state")


def test_terminal_manual_input_avoids_prompt_and_infers_output_directory(monkeypatch):
    from analysisrun import context as context_module

    _force_interactivity(monkeypatch, "terminal")
    monkeypatch.setattr(context_module, "get_entrypoint", lambda: Path(__file__))
    monkeypatch.setattr(
        context_module, "scan_model_input", lambda *a: pytest.fail("prompted")
    )
    ctx = read_context(Params, ImageResults, manual_input=_manual_input())
    assert ctx.output_dir == DATA_DIR


def test_terminal_without_manual_input_prompts(monkeypatch, tmp_path):
    from analysisrun import context as context_module

    _force_interactivity(monkeypatch, "terminal")
    monkeypatch.setattr(context_module, "get_entrypoint", lambda: Path(__file__))
    calls = []

    def prompt(model):
        calls.append(model)
        return model(
            params=Params(threshold=4),
            sample_names=SAMPLES_CSV,
            image_analysis_results={"activity_spots": IMAGE_ANALYSIS_RESULT_CSV},
        )

    monkeypatch.setattr(context_module, "scan_model_input", prompt)
    ctx = read_context(Params, ImageResults, output_dir=tmp_path)
    assert ctx.mode == "interactive"
    assert ctx.params.threshold == 4
    assert len(calls) == 1


@pytest.mark.parametrize("invalid", ["samples", "image", "params", "cleansing"])
def test_invalid_input_stays_in_streaming_error_protocol(
    monkeypatch, tmp_path, invalid
):
    _force_interactivity(monkeypatch, None)
    samples = _dump_csv(pd.DataFrame({"data": ["0000"], "sample": ["A"]}))
    image = _dump_csv(pd.read_csv(IMAGE_ANALYSIS_RESULT_CSV))
    params = Params().model_dump_json()
    if invalid == "samples":
        samples = _dump_csv(pd.DataFrame({"data": [], "sample": []}))
    elif invalid == "image":
        image = BytesIO(b"invalid,csv\n1,2")
    elif invalid == "params":
        params = '{"threshold": "bad"}'

    class InvalidCleansing(NamedTuple):
        activity_spots: pd.DataFrame = image_analysis_result_spec(
            "invalid", lambda data: data
        )

    request = create_tar_from_dict(
        {
            "params": params,
            "sample_names": samples,
            "image_analysis_results/activity_spots": image,
        }
    )
    response = BytesIO()
    with pytest.raises(SystemExit) as caught:
        read_context(
            Params,
            InvalidCleansing if invalid == "cleansing" else ImageResults,
            stdin=request,
            stdout=response,
            output_dir=tmp_path,
        )
    assert caught.value.code == 2
    result = read_tar_as_dict(BytesIO(response.getvalue()))
    assert "error" in result


@pytest.mark.parametrize(
    "mode", ["analyzeseq", "pipeline-worker", "sequential", "parallel-entrypoint"]
)
def test_removed_modes_are_rejected(monkeypatch, mode):
    monkeypatch.setenv("ANALYSISRUN_MODE", mode)
    _force_interactivity(monkeypatch, None)
    response = BytesIO()
    with pytest.raises(SystemExit) as caught:
        read_context(Params, ImageResults, stdin=BytesIO(), stdout=response)
    assert caught.value.code == 2
    assert mode in read_tar_as_dict(BytesIO(response.getvalue()))["error"]
