"""Tests for the headless sentinel runner + Task Scheduler wrapper."""

from __future__ import annotations

import sys

import pytest

from researchbuddy import sentinel_cli as sc


# ── schtasks command construction ──────────────────────────────────────────────

def test_build_task_command():
    cmd = sc.build_task_command("03:30", autotune_rounds=20)
    assert cmd[0] == "schtasks" and "/Create" in cmd
    assert cmd[cmd.index("/TN") + 1] == sc.TASK_NAME
    assert cmd[cmd.index("/SC") + 1] == "DAILY"
    assert cmd[cmd.index("/ST") + 1] == "03:30"
    tr = cmd[cmd.index("/TR") + 1]
    assert sys.executable in tr
    assert "-m researchbuddy.sentinel_cli" in tr
    assert "--autotune 20" in tr
    assert "/F" in cmd                      # idempotent re-install


def test_build_task_command_no_autotune():
    tr = sc.build_task_command("03:00", 0)
    assert "--autotune" not in tr[tr.index("/TR") + 1]


def test_build_uninstall_command():
    cmd = sc.build_uninstall_command()
    assert "/Delete" in cmd and sc.TASK_NAME in cmd


# ── main() flows ──────────────────────────────────────────────────────────────

def test_install_task_invokes_schtasks(monkeypatch):
    calls = {}

    class _Res:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(sc.sys, "platform", "win32")
    def _fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return _Res()

    monkeypatch.setattr(sc.subprocess, "run", _fake_run)
    rc = sc.main(["--install-task", "--time", "04:00", "--autotune", "5"])
    assert rc == 0
    assert calls["cmd"][0] == "schtasks"
    assert "04:00" in calls["cmd"]


def test_headless_run(graph_with_papers, monkeypatch):
    import researchbuddy.core.state_manager as sm
    from researchbuddy.core import sentinel as sn
    from researchbuddy.core import autotune as at

    monkeypatch.setattr(sm, "load", lambda: graph_with_papers)
    saved = {}
    monkeypatch.setattr(sm, "save", lambda g: saved.setdefault("g", g))
    monkeypatch.setattr(sn, "run_scan",
                        lambda g, progress=None: {"new": 2, "per_watch": [],
                                                  "digest": None})
    monkeypatch.setattr(at, "apply_saved_tuning", lambda g: [])
    # Isolate the scout: without this the headless run reads the developer's
    # real ~/.researchbuddy/scout_graph.pkl and hits the network.
    from researchbuddy.core import scout as sg
    monkeypatch.setattr(sg, "load_state", lambda path=None: {"enabled": False})
    monkeypatch.setattr(
        at, "run_session",
        lambda g, rounds, progress=None: {"ready": True, "baseline": 0.5,
                                          "best": 0.52, "kept": {"alpha": 0.7},
                                          "improved": True, "experiments": []})

    rc = sc.main(["--autotune", "3"])
    assert rc == 0
    assert saved["g"] is graph_with_papers   # graph saved at the end


def test_headless_run_no_graph(monkeypatch):
    import researchbuddy.core.state_manager as sm
    monkeypatch.setattr(sm, "load", lambda: None)
    assert sc.main([]) == 2
