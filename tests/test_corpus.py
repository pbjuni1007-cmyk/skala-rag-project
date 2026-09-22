"""Source contracts use tiny local fixtures; no PDFs or models are downloaded."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import urllib.error

import pytest

from rag.corpus import Corpus, digest, sentence_windows
from rag.settings import Settings


@pytest.fixture
def corpus_fixture(tmp_path, monkeypatch):
    settings = Settings({"RAG_DATA_DIR": str(tmp_path / "data"),
                         "RAG_INDEX_DIR": str(tmp_path / "index"),
                         "RAG_WEB_SNAPSHOT_DIR": str(tmp_path / "web")})
    config = {"papers": [
        {"id": "kivi", "technology": "KIVI", "version": "2402.02750v2", "title": "KIVI", "authors": "KIVI authors", "date": "2024", "url": "https://arxiv.org/pdf/2402.02750v2"},
        {"id": "infinigen", "technology": "InfiniGen", "version": "2406.19707v1", "title": "InfiniGen", "authors": "InfiniGen authors", "date": "2024", "url": "https://arxiv.org/pdf/2406.19707v1"}],
        "web": [{"id": "official", "title": "Official cache guide", "authors": "Hugging Face",
                 "url": "https://huggingface.co/docs/transformers/kv_cache"}]}
    pages = {paper["version"]: ["A valid page with source evidence."] for paper in config["papers"]}
    for paper in config["papers"]:
        path = tmp_path / "data/papers" / (paper["version"] + ".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-pdf-" + paper["version"].encode())

    def fake_reader(path):
        return SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda extraction_mode, text=text: text)
                                     for text in pages[Path(path).stem]])

    def forbidden(*args, **kwargs):
        pytest.fail("Unit tests must not download sources or models")

    monkeypatch.setattr("rag.corpus.PdfReader", fake_reader)
    monkeypatch.setattr("rag.corpus.download", forbidden)
    text = "KV cache source text for the offline snapshot."
    snapshot = {**config["web"][0], "text": text, "text_sha256": digest(text.encode()),
                "raw_sha256": digest(text.encode()), "accessed_at": "2026-01-01T00:00:00+00:00", "status": "ok"}
    snapshot_path = tmp_path / "web/official.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text(json.dumps(snapshot))
    return Corpus(settings, deepcopy(config)), pages, snapshot_path


@pytest.mark.parametrize("count,accepted", [(200, True), (201, False)])
def test_total_pdf_page_limit_is_enforced_across_both_papers(corpus_fixture, count, accepted):
    corpus, pages, _ = corpus_fixture
    pages["2402.02750v2"] *= 100
    pages["2406.19707v1"] *= count - 100
    if accepted:
        assert corpus.prepare_sources() == count
        manifest = json.loads((corpus.root / "source_manifest.json").read_text())
        assert manifest["total_pdf_pages"] == count
    else:
        with pytest.raises(ValueError, match="maximum of 200"):
            corpus.prepare_sources()
        assert not (corpus.root / "source_manifest.json").exists()


@pytest.mark.parametrize("mutation", ["unversioned", "different_url", "duplicate", "missing"])
def test_paper_pool_requires_both_approved_versions_and_exact_urls(corpus_fixture, mutation):
    corpus, _, _ = corpus_fixture
    if mutation == "unversioned":
        corpus.config["papers"][0]["version"] = "2402.02750"
    elif mutation == "different_url":
        corpus.config["papers"][0]["url"] = "https://example.invalid/paper.pdf"
    elif mutation == "duplicate":
        corpus.config["papers"][1] = deepcopy(corpus.config["papers"][0])
    else:
        corpus.config["papers"].pop()
    with pytest.raises(ValueError, match="approved"):
        corpus.prepare_sources()


def test_unregistered_web_url_is_rejected_before_download(corpus_fixture):
    corpus, _, _ = corpus_fixture
    corpus.config["web"][0]["url"] = "https://example.invalid/unregistered"
    with pytest.raises(ValueError, match="Unregistered source URL"):
        corpus.prepare_sources()


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_pdf_page_stops_source_preparation(corpus_fixture, text):
    corpus, pages, _ = corpus_fixture
    pages["2402.02750v2"] = [text]
    with pytest.raises(ValueError, match="Empty PDF page 2402.02750v2:1"):
        corpus.prepare_sources()


@pytest.mark.parametrize("field,value", [("text", "tampered content"), ("url", "https://example.invalid")])
def test_cached_web_snapshot_integrity_is_checked(corpus_fixture, field, value):
    corpus, _, path = corpus_fixture
    snapshot = json.loads(path.read_text())
    snapshot[field] = value
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="snapshot integrity mismatch"):
        corpus.prepare_sources()


def test_cached_sources_preserve_version_hash_access_date_and_page_location(corpus_fixture):
    corpus, _, snapshot_path = corpus_fixture
    cached = json.loads(snapshot_path.read_text())
    assert corpus.prepare_sources() == 2
    assert corpus.pages[0]["page"] == 1 and corpus.pages[0]["source_id"] == "kivi"
    for source_id in ("kivi", "infinigen"):
        source = corpus.sources[source_id]
        assert source["sha256"] == digest(Path(source["local_path"]).read_bytes())
        assert source["authors"] and source["date"] and source["version"] and source["accessed_at"]
    web = corpus.sources["official"]
    assert web["version"] == cached["text_sha256"]
    assert web["accessed_at"] == cached["accessed_at"]
    hit = corpus.web_search("KV cache")[0]
    assert hit["source_id"] == "official" and hit["page"] is None and hit["section"]


def test_refresh_failure_keeps_verified_snapshot_available(corpus_fixture, monkeypatch):
    corpus, _, snapshot_path = corpus_fixture
    cached = json.loads(snapshot_path.read_text())

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("private remote error body")

    monkeypatch.setattr("rag.corpus.download", unavailable)
    corpus.prepare_sources(refresh_web=True)
    assert json.loads(snapshot_path.read_text()) == cached
    source = corpus.sources["official"]
    assert source["version"] == cached["text_sha256"]
    assert source["accessed_at"] == cached["accessed_at"]
    assert source["freshness"] == "cached_after_refresh_failure"
    assert corpus.web_search("KV cache")[0]["source_id"] == "official"
    assert "private remote error body" not in json.dumps(source)


@pytest.mark.parametrize("field,value", [("text", "tampered content"), ("url", "https://example.invalid")])
def test_refresh_failure_never_reuses_a_tampered_snapshot(corpus_fixture, monkeypatch, field, value):
    corpus, _, snapshot_path = corpus_fixture
    cached = json.loads(snapshot_path.read_text())
    cached[field] = value
    snapshot_path.write_text(json.dumps(cached))

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("rag.corpus.download", unavailable)
    corpus.prepare_sources(refresh_web=True)
    assert corpus.sources["official"]["status"] == "unavailable"
    assert corpus.web_search("KV cache") == []


def test_long_comparison_sentence_is_not_cut_at_character_limit():
    comparison = "InfiniGen " + "important context " * 85 + "improves performance compared to prior KV cache management methods."
    windows = list(sentence_windows("An introduction. " + comparison + " A limitation remains."))
    assert any(comparison in text for _, text in windows)
    assert all(not text.endswith("compar") for _, text in windows)


def test_sentence_windows_keep_source_offsets_and_boundary_context():
    sentences = [f"Evidence {n} " + "memory " * 40 + "." for n in range(8)]
    source = " ".join(sentences)
    windows = list(sentence_windows(source, target=650))
    assert all(source[offset:offset + len(text)] == text for offset, text in windows)
    assert all(any(sentence in text for _, text in windows) for sentence in sentences)
    assert all(any(left in text and right in text for _, text in windows)
               for left, right in zip(sentences, sentences[1:]))


def test_web_search_preserves_complete_comparison_clause(corpus_fixture):
    corpus, _, path = corpus_fixture
    comparison = "InfiniGen " + "KV cache context " * 80 + "improves performance compared to prior KV cache management methods."
    snapshot = json.loads(path.read_text())
    snapshot.update(text=comparison, text_sha256=digest(comparison.encode()))
    path.write_text(json.dumps(snapshot))
    corpus.prepare_sources()
    hits = corpus.web_search("InfiniGen performance compared")
    assert hits and comparison in hits[0]["text"]


def test_query_length_guard_runs_before_encoder(corpus_fixture):
    corpus, _, _ = corpus_fixture
    def forbidden(*args, **kwargs):
        pytest.fail("Encoder must not silently truncate an oversized query")
    corpus.embedding = {"prefixes": ["", ""], "max_tokens": 256}
    corpus.encoder = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **kw: list(range(257))), encode=forbidden)
    with pytest.raises(ValueError, match="query would be truncated"):
        corpus.search("too long", "KIVI")


def test_model_change_does_not_reuse_other_models_revision(corpus_fixture, monkeypatch):
    corpus, _, _ = corpus_fixture
    corpus.settings.values["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    corpus.index.mkdir(parents=True)
    (corpus.index / "embedding.json").write_text(json.dumps({"model": "intfloat/multilingual-e5-small", "revision": "old-e5-only-revision"}))
    calls = []
    def snapshot(model, revision, **kwargs):
        calls.append((model, revision))
        return "local-public-model"
    class Missing(Exception):
        pass
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(LocalEntryNotFoundError=Missing))
    encoder = SimpleNamespace(max_seq_length=256, get_embedding_dimension=lambda: 384)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **kw: encoder))
    corpus.load_encoder()
    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")]
    assert corpus.embedding["prefixes"] == ["", ""]
