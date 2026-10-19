"""本番エンジンから独立した #81 の成立性検証。"""

import pytest

from tests import pipeline_probe as probe


def request_for(definition, index):
    return {
        "manifest": probe.canonical_manifest(
            (*definition.stages, probe.Stage("finalize", probe.to_result))
        ),
        "stage_index": index,
        # 子は前段の計算結果を受け取り、元入力や extra も再生成しない。
        "state": {"input": 4, "value": 8, "synthesized": 11, "extra": 10},
    }


@pytest.mark.parametrize("with_preprocess", [True, False])
def test_child_rebuilds_manifest_and_executes_only_second_map(with_preprocess):
    definition = probe.build(with_preprocess)
    request = request_for(definition, 3 if with_preprocess else 2)
    code, response, stderr = probe.run_child(request, with_preprocess=with_preprocess)
    assert code == 0, stderr
    assert response["manifest"] == request["manifest"]
    assert response["stage_index"] == request["stage_index"]
    assert response["calls"] == ["apply_qc"]
    assert response["state"] == request["state"] | {"value": 29}


@pytest.mark.parametrize(
    "mutation", ["add", "delete", "reorder", "kind", "callable", "finalize"]
)
def test_child_rejects_changed_definition_before_execution(mutation):
    code, response, stderr = probe.run_child(
        request_for(probe.build(), 3), mutation=mutation
    )
    assert code == 1
    assert response == {"error": "pipeline manifest mismatch", "calls": []}
    assert "pipeline manifest mismatch" in stderr


def test_child_rejects_preprocess_presence_mismatch():
    code, response, stderr = probe.run_child(
        request_for(probe.build(), 3), with_preprocess=False
    )
    assert code == 1
    assert response["calls"] == []
    assert "pipeline manifest mismatch" in stderr


@pytest.mark.parametrize("index", [-1, 5, 999, True, 1.0])
def test_child_rejects_invalid_index(index):
    code, response, stderr = probe.run_child(request_for(probe.build(), index))
    assert code == 1
    assert response == {"error": "stage index out of range", "calls": []}
    assert "stage index out of range" in stderr


@pytest.mark.parametrize("index", [0, 2, 4])
def test_child_rejects_non_map_stage(index):
    code, response, stderr = probe.run_child(request_for(probe.build(), index))
    assert code == 1
    assert response == {"error": "requested stage is not map", "calls": []}
    assert "requested stage is not map" in stderr


def test_building_is_immutable_and_never_calls_stages():
    def forbidden(state):
        pytest.fail("construction executed a callable")

    initial = probe.Definition()
    prepared = initial.append("preprocess", forbidden)
    mapped = prepared.append("map", forbidden)
    synthesized = mapped.append("synthesize", forbidden)
    assert initial.stages == ()
    assert len(prepared.stages) == 1
    assert len(mapped.stages) == 2
    assert len(synthesized.stages) == 3
    for definition in (prepared, mapped, synthesized):
        with pytest.raises(ValueError, match="first and unique"):
            definition.append("preprocess", forbidden)


@pytest.mark.parametrize("with_preprocess", [True, False])
def test_sequential_execution_needs_no_manifest(monkeypatch, with_preprocess):
    def forbidden(*args):
        pytest.fail("sequential execution generated a manifest")

    monkeypatch.setattr(probe, "canonical_manifest", forbidden)
    result = probe.build(with_preprocess).finalize(
        probe.to_result,
        {"input": 3, "value": None, "synthesized": None, "extra": 0},
    )
    assert result["calls"] == (
        (["prepare"] if with_preprocess else [])
        + ["extract_features", "evaluate_qc", "apply_qc", "to_result"]
    )
    assert result["state"]["result"] == (29 if with_preprocess else 15)


def test_zero_stages_can_finalize_initial_none_values():
    state = {"value": None, "synthesized": None}
    result = probe.Definition().finalize(lambda args: args, state)
    assert result["state"] == state


def test_synthesize_can_start_and_same_kinds_can_repeat():
    def synthesize(state):
        return state | {"synthesized": (state["synthesized"] or 0) + 1}

    def map_value(state):
        return state | {"value": (state["value"] or 0) + state["synthesized"]}

    definition = (
        probe.Definition()
        .append("synthesize", synthesize)
        .append("synthesize", synthesize)
        .append("map", map_value)
        .append("map", map_value)
    )
    result = definition.finalize(probe.to_result, {"value": None, "synthesized": None})
    assert result["state"] == {"value": 4, "synthesized": 2, "result": 4}
