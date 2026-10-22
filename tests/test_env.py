from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from analysisrun import env


@pytest.mark.parametrize(
    "notebook, tty, expected",
    [
        (True, True, "notebook"),
        (True, False, "notebook"),
        (False, True, "interactive"),
        (False, False, "streaming"),
    ],
)
def test_mode_from_notebook_and_tty(
    monkeypatch: pytest.MonkeyPatch, notebook: bool, tty: bool, expected: str
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.delenv("ANALYSISRUN_INTERACTIVITY", raising=False)
    monkeypatch.setattr(env, "is_notebook_env", lambda: notebook)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    assert env.get_execution_mode() == expected


@pytest.mark.parametrize(
    "forced, expected", [("notebook", "notebook"), ("terminal", "interactive")]
)
def test_interactivity_override(
    monkeypatch: pytest.MonkeyPatch, forced: str, expected: str
):
    monkeypatch.delenv("ANALYSISRUN_MODE", raising=False)
    monkeypatch.setenv("ANALYSISRUN_INTERACTIVITY", forced)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert env.get_execution_mode() == expected


def test_showschema_is_an_independent_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANALYSISRUN_MODE", "showschema")
    assert env.get_execution_mode() == "showschema"


def test_worker_role_does_not_classify_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANALYSISRUN_PROCESS_ROLE", "parallel-worker")
    monkeypatch.setattr(
        env, "get_interactivity", lambda: pytest.fail("classified worker")
    )
    assert env.get_process_role() == "parallel-worker"
