"""Tests for the legal open-access harvester (core/oa_harvester.py)."""

from __future__ import annotations

import pytest

from researchbuddy.core import oa_harvester as oh
from researchbuddy.core.graph_model import PaperMeta


def _meta(**kw):
    base = dict(paper_id="p001", title="A Paper", abstract="An abstract.")
    base.update(kw)
    return PaperMeta(**base)


# ── Resolver order & parsing ───────────────────────────────────────────────────

def test_arxiv_resolver_wins_first():
    meta = _meta(arxiv_id="2101.12345", doi="10.1/x")
    loc = oh.resolve_oa(meta)
    assert loc is not None
    assert loc.provider == "arxiv"
    assert "arxiv.org/pdf/2101.12345" in loc.pdf_url


def test_unpaywall_parsing(monkeypatch):
    monkeypatch.setattr(oh, "UNPAYWALL_EMAIL", "test@example.com")

    def fake_get_json(url, params=None):
        if "unpaywall" in url:
            return {
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": "https://repo.example.org/oa.pdf",
                    "license": "cc-by",
                    "version": "publishedVersion",
                    "host_type": "repository",
                    "url_for_landing_page": "https://repo.example.org/land",
                },
            }
        return None

    monkeypatch.setattr(oh, "_get_json", fake_get_json)
    loc = oh.resolve_oa(_meta(doi="10.1234/abcd"))
    assert loc.provider == "unpaywall"
    assert loc.license == "cc-by"
    assert loc.pdf_url.endswith("oa.pdf")


def test_no_oa_returns_none(monkeypatch):
    monkeypatch.setattr(oh, "UNPAYWALL_EMAIL", "test@example.com")
    monkeypatch.setattr(oh, "_get_json", lambda url, params=None: None)
    assert oh.resolve_oa(_meta(doi="10.1234/closed")) is None


def test_unpaywall_closed_falls_through_to_openalex(monkeypatch):
    monkeypatch.setattr(oh, "UNPAYWALL_EMAIL", "test@example.com")

    def fake_get_json(url, params=None):
        if "unpaywall" in url:
            return {"is_oa": False}
        if "openalex" in url:
            return {
                "best_oa_location": {
                    "is_oa": True,
                    "pdf_url": "https://oa.example.org/x.pdf",
                    "license": "cc-by-nc",
                    "version": "acceptedVersion",
                    "source": {"type": "repository"},
                },
            }
        return None

    monkeypatch.setattr(oh, "_get_json", fake_get_json)
    loc = oh.resolve_oa(_meta(doi="10.1234/abcd"))
    assert loc.provider == "openalex"
    assert loc.license == "cc-by-nc"


def test_europepmc_resolver(monkeypatch):
    monkeypatch.setattr(oh, "UNPAYWALL_EMAIL", "")  # skip unpaywall

    def fake_get_json(url, params=None):
        if "europepmc" in url:
            return {"resultList": {"result": [
                {"pmcid": "PMC1234567", "license": "cc by"},
            ]}}
        return None  # openalex returns nothing

    monkeypatch.setattr(oh, "_get_json", fake_get_json)
    loc = oh.resolve_oa(_meta(doi="10.1234/bio"))
    assert loc.provider == "europepmc"
    assert "PMC1234567" in loc.pdf_url


# ── Download safety ────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status=200, chunks=(b"%PDF-1.5 fake body",)):
        self.status_code = status
        self._chunks = chunks

    def iter_content(self, chunk_size=65536):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_rejects_html_masquerading_as_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(oh.requests, "get",
                        lambda *a, **k: _FakeResponse(chunks=(b"<html>paywall</html>",)))
    loc = oh.OALocation(pdf_url="https://x/y.pdf", provider="unpaywall")
    assert oh.download_pdf(loc, tmp_path / "out.pdf") is False


def test_download_accepts_real_pdf_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(oh.requests, "get",
                        lambda *a, **k: _FakeResponse())
    loc = oh.OALocation(pdf_url="https://x/y.pdf", provider="arxiv")
    dest = tmp_path / "out.pdf"
    assert oh.download_pdf(loc, dest) is True
    assert dest.read_bytes().startswith(b"%PDF")


def test_download_handles_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(oh.requests, "get",
                        lambda *a, **k: _FakeResponse(status=404))
    loc = oh.OALocation(pdf_url="https://x/y.pdf", provider="openalex")
    assert oh.download_pdf(loc, tmp_path / "out.pdf") is False


# ── Harvest orchestration ──────────────────────────────────────────────────────

def test_harvestable_papers_filters_and_orders(graph_with_papers):
    g = graph_with_papers
    papers = g.all_papers()
    papers[0].doi = "10.1/a"
    papers[1].doi = "10.1/b"
    papers[1].user_rating = 9.0
    papers[2].filepath = "C:/already/have.pdf"
    papers[2].doi = "10.1/c"
    # papers[3], papers[4] have no doi/arxiv -> not harvestable

    todo = oh.harvestable_papers(g)
    ids = [m.paper_id for m in todo]
    assert papers[2].paper_id not in ids          # already has a file
    assert ids[0] == papers[1].paper_id           # rated first
    assert papers[0].paper_id in ids
    assert len(todo) == 2


def test_harvest_no_oa_path(graph_with_papers, monkeypatch, tmp_path):
    g = graph_with_papers
    m = g.all_papers()[0]
    m.doi = "10.1/a"

    monkeypatch.setattr(oh, "resolve_oa", lambda meta: None)
    monkeypatch.setattr(oh, "OA_LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(oh, "REQUEST_DELAY", 0)

    report = oh.harvest(g, papers=[m])
    assert report.checked == 1
    assert report.no_oa == 1
    assert report.downloaded == 0
    assert report.ingested == 0


def test_harvest_success_path(graph_with_papers, monkeypatch, tmp_path):
    g = graph_with_papers
    m = g.all_papers()[0]
    m.doi = "10.1/a"
    m.filepath = ""

    loc = oh.OALocation(pdf_url="https://x/y.pdf", provider="unpaywall",
                        license="cc-by")
    monkeypatch.setattr(oh, "resolve_oa", lambda meta: loc)
    monkeypatch.setattr(oh, "OA_LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(oh, "REQUEST_DELAY", 0)

    def fake_download(l, dest):
        dest.write_bytes(b"%PDF-1.5 stub")
        return True

    monkeypatch.setattr(oh, "download_pdf", fake_download)

    def fake_ingest(graph, meta, pdf_path):
        meta.filepath = str(pdf_path)
        return {"parser": "grobid", "n_sections": 5, "n_refs": 12}

    import researchbuddy.core.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "ingest_pdf_into_meta", fake_ingest)

    # Audit into a temp log
    from researchbuddy.core import audit
    monkeypatch.setattr(audit, "PRISMA_LOG", tmp_path / "log.jsonl")

    report = oh.harvest(g, papers=[m])
    assert report.ingested == 1
    assert report.by_provider == {"unpaywall": 1}
    assert m.filepath  # meta upgraded

    # Provenance sidecar written next to the PDF
    sidecars = list(tmp_path.glob("*.provenance.json"))
    assert len(sidecars) == 1
    assert "unpaywall" in sidecars[0].read_text(encoding="utf-8")
