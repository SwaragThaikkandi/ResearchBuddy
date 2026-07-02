"""Tests for runtime CORE API key handling (core_fetcher.set_api_key)."""

from __future__ import annotations

from researchbuddy.core import core_fetcher as cf


def test_set_and_clear_api_key(monkeypatch):
    monkeypatch.setattr(cf, "_CORE_API_KEY", "")
    cf._HEADERS.pop("Authorization", None)

    cf.set_api_key("my-secret-key")
    assert cf.has_api_key()
    assert cf._HEADERS["Authorization"] == "Bearer my-secret-key"
    assert cf._REQUEST_DELAY == 0.15          # fast lane with a key

    cf.set_api_key("")
    assert not cf.has_api_key()
    assert "Authorization" not in cf._HEADERS
    assert cf._REQUEST_DELAY == 1.1           # polite anonymous rate


def test_key_persists_via_service_prefs(tmp_path, monkeypatch):
    """The CLI stores the key in service prefs; a fresh session reapplies it."""
    from researchbuddy.core import services as svc
    monkeypatch.setattr(svc, "_prefs_path", lambda: tmp_path / "prefs.json")

    prefs = svc.load_prefs()
    prefs["core_api_key"] = "saved-key"
    svc.save_prefs(prefs)
    assert svc.load_prefs()["core_api_key"] == "saved-key"
