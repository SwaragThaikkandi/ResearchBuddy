"""
Tests for services.py — Docker lifecycle + preference persistence.

All Docker subprocess calls are mocked; no real docker daemon required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from researchbuddy.core import services as svc


# ── Preferences ──────────────────────────────────────────────────────────────

def test_load_prefs_returns_empty_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "researchbuddy.config.DATA_DIR", tmp_path / "missing",
    )
    assert svc.load_prefs() == {}


def test_save_and_load_prefs_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("researchbuddy.config.DATA_DIR", tmp_path)
    svc.save_prefs({"neo4j_auto_launch": "yes", "grobid_auto_launch": "never"})
    assert svc.load_prefs() == {
        "neo4j_auto_launch": "yes",
        "grobid_auto_launch": "never",
    }


def test_load_prefs_handles_corrupt_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("researchbuddy.config.DATA_DIR", tmp_path)
    p = tmp_path / "service_prefs.json"
    p.write_text("{ this is not json }", encoding="utf-8")
    assert svc.load_prefs() == {}


# ── Docker availability ──────────────────────────────────────────────────────

def test_docker_unavailable_when_binary_missing():
    with patch("researchbuddy.core.services.shutil.which", return_value=None):
        assert svc.docker_available() is False


def test_docker_available_when_info_succeeds():
    with patch("researchbuddy.core.services.shutil.which", return_value="/usr/bin/docker"):
        with patch("researchbuddy.core.services.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="24.0.5\n", stderr="")
            assert svc.docker_available() is True


def test_docker_unavailable_when_daemon_unreachable():
    with patch("researchbuddy.core.services.shutil.which", return_value="/usr/bin/docker"):
        with patch("researchbuddy.core.services.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Cannot connect")
            assert svc.docker_available() is False


# ── Container state ──────────────────────────────────────────────────────────

def test_container_state_running():
    with patch("researchbuddy.core.services.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="running\n")
        assert svc._container_state("foo") == "running"


def test_container_state_stopped():
    with patch("researchbuddy.core.services.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="exited\n")
        assert svc._container_state("foo") == "stopped"


def test_container_state_missing():
    with patch("researchbuddy.core.services.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="No such container")
        assert svc._container_state("foo") == "missing"


# ── ensure_running orchestration ─────────────────────────────────────────────

def test_ensure_running_short_circuits_when_already_alive():
    with patch("researchbuddy.core.services._service_alive", return_value=True):
        with patch("researchbuddy.core.services.docker_available") as mock_docker:
            res = svc.ensure_running(svc.GROBID_SPEC)
            assert res.already_running is True
            assert res.started is False
            mock_docker.assert_not_called()


def test_ensure_running_reports_when_docker_missing():
    with patch("researchbuddy.core.services._service_alive", return_value=False):
        with patch("researchbuddy.core.services.docker_available", return_value=False):
            res = svc.ensure_running(svc.GROBID_SPEC)
            assert res.already_running is False
            assert res.started is False
            assert res.error and "Docker" in res.error


def test_ensure_running_starts_existing_stopped_container():
    with patch("researchbuddy.core.services._service_alive", side_effect=[False, True]):
        with patch("researchbuddy.core.services.docker_available", return_value=True):
            with patch("researchbuddy.core.services._container_state", return_value="stopped"):
                with patch("researchbuddy.core.services._docker_start", return_value=True) as mock_start:
                    res = svc.ensure_running(svc.GROBID_SPEC)
                    mock_start.assert_called_once()
                    assert res.started is True


def test_ensure_running_runs_new_container_when_missing():
    with patch("researchbuddy.core.services._service_alive", side_effect=[False, True]):
        with patch("researchbuddy.core.services.docker_available", return_value=True):
            with patch("researchbuddy.core.services._container_state", return_value="missing"):
                with patch("researchbuddy.core.services._docker_run", return_value=True) as mock_run:
                    res = svc.ensure_running(svc.GROBID_SPEC)
                    mock_run.assert_called_once()
                    assert res.started is True


def test_ensure_running_reports_unhealthy_after_timeout():
    """If the container starts but never becomes healthy, surface that."""
    with patch("researchbuddy.core.services._service_alive", return_value=False):
        with patch("researchbuddy.core.services.docker_available", return_value=True):
            with patch("researchbuddy.core.services._container_state", return_value="running"):
                # Patch the wait loop to immediately give up
                with patch("researchbuddy.core.services._wait_until_alive", return_value=False):
                    res = svc.ensure_running(svc.GROBID_SPEC)
                    assert res.started is False
                    assert "did not become healthy" in (res.error or "")


# ── Health-match callbacks ───────────────────────────────────────────────────

def test_grobid_alive_match():
    r = MagicMock(status_code=200, text="true")
    assert svc._is_grobid_alive(r) is True
    r = MagicMock(status_code=200, text="false")
    assert svc._is_grobid_alive(r) is False
    r = MagicMock(status_code=503, text="")
    assert svc._is_grobid_alive(r) is False


def test_neo4j_alive_match():
    r = MagicMock(status_code=200)
    r.json.return_value = {"neo4j_version": "5.13.0"}
    assert svc._is_neo4j_alive(r) is True
    # Even on 200 with no JSON, falls back to status code
    r2 = MagicMock(status_code=200)
    r2.json.side_effect = ValueError("no json")
    assert svc._is_neo4j_alive(r2) is True
    # 503 → not alive
    r3 = MagicMock(status_code=503)
    r3.json.side_effect = ValueError("no json")
    assert svc._is_neo4j_alive(r3) is False


def test_service_alive_returns_false_on_connection_error():
    with patch("researchbuddy.core.services.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("refused")
        assert svc._service_alive(svc.GROBID_SPEC) is False
