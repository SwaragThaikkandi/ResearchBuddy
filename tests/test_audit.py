"""Tests for the PRISMA audit trail (core/audit.py)."""

from __future__ import annotations

from researchbuddy.core import audit


def test_log_and_read_roundtrip(tmp_path):
    log = tmp_path / "prisma.jsonl"
    audit.log_event("search", log_path=log, query="drift diffusion",
                    keywords=["choice"], n_results=12)
    audit.log_event("screen", log_path=log, paper_id="p1", title="T",
                    doi="10.1/x", rating=8, decision="included")

    events = audit.read_events(log_path=log)
    assert len(events) == 2
    assert events[0]["event"] == "search"
    assert events[0]["n_results"] == 12
    assert events[1]["decision"] == "included"
    assert all("ts" in e for e in events)

    only_screen = audit.read_events("screen", log_path=log)
    assert len(only_screen) == 1


def test_read_missing_log_is_empty(tmp_path):
    assert audit.read_events(log_path=tmp_path / "nope.jsonl") == []


def test_read_skips_corrupt_lines(tmp_path):
    log = tmp_path / "prisma.jsonl"
    audit.log_event("search", log_path=log, n_results=1)
    with open(log, "a", encoding="utf-8") as f:
        f.write("{not json}\n")
    audit.log_event("search", log_path=log, n_results=2)
    assert len(audit.read_events(log_path=log)) == 2


def test_screen_decision_thresholds():
    assert audit.screen_decision(None) == "skipped"
    assert audit.screen_decision(9) == "included"
    assert audit.screen_decision(3) == "excluded"


def test_prisma_counts_aggregation(tmp_path):
    log = tmp_path / "prisma.jsonl"
    audit.log_event("search", log_path=log, n_results=10,
                    sources={"openalex": 6, "arxiv": 4})
    audit.log_event("search", log_path=log, n_results=5,
                    sources={"openalex": 5})
    audit.log_event("snowball", log_path=log, new_unique=7, fetched=40)
    audit.log_event("watch_check", log_path=log, query="q", n_new=2)
    audit.log_event("screen", log_path=log, paper_id="a", rating=8,
                    decision="included")
    audit.log_event("screen", log_path=log, paper_id="b", rating=2,
                    decision="excluded")
    audit.log_event("fulltext", log_path=log, paper_id="a",
                    provider="unpaywall")

    c = audit.prisma_counts(log_path=log)
    assert c["n_searches"] == 2
    assert c["identified_search"] == 15
    assert c["identified_snowball"] == 7
    assert c["identified_watch"] == 2
    assert c["identified_total"] == 24
    assert c["identified_by_source"] == {"openalex": 11, "arxiv": 4}
    assert c["screened"] == 2
    assert c["included"] == 1
    assert c["excluded"] == 1
    assert c["fulltext_retrieved"] == 1


def test_prisma_counts_latest_decision_wins(tmp_path):
    log = tmp_path / "prisma.jsonl"
    audit.log_event("screen", log_path=log, paper_id="a", rating=2,
                    decision="excluded")
    audit.log_event("screen", log_path=log, paper_id="a", rating=9,
                    decision="included")
    c = audit.prisma_counts(log_path=log)
    assert c["screened"] == 1
    assert c["included"] == 1
    assert c["excluded"] == 0
