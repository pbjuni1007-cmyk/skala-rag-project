from copy import deepcopy

import pytest

from agents.researchers.contract import ResearchResult
from agents.researchers.result import (
    cited_artifacts,
    contract_chunk,
    contract_source,
    trace_errors,
)


def _result():
    return ResearchResult.model_validate(
        {
            "contract_version": "agent-contract-v1",
            "run_id": "run-trace-1",
            "request_id": "request-market-1",
            "attempt": 1,
            "context": {
                "technologies": ["KIVI", "InfiniGen"],
                "domain": "기업 문서 검토",
                "scenario": "정책 문서에서 근거를 확인한다.",
            },
            "view": "market",
            "status": "ok",
            "assessments": {
                "market": {
                    "status": "ok",
                    "claims": [
                        {
                            "technology": "KIVI",
                            "facet": "costs",
                            "text": "A sourced claim connects both records.",
                            "kind": "source_fact",
                            "references": [
                                {
                                    "chunk_id": "paper:p1:t0",
                                    "quote": "Quantization reduces cache storage while retaining task-specific accuracy.",
                                },
                                {
                                    "chunk_id": "web:s2:c10",
                                    "quote": "The snapshot describes an operator-managed host memory offload path.",
                                },
                            ],
                            "caveats": "The quoted scope is limited to the described systems.",
                            "conditions": "Only the cited configuration is covered.",
                        }
                    ],
                    "conflicts": [],
                    "gaps": [],
                }
            },
            "chunks": [
                {
                    "id": "paper:p1:t0",
                    "source_id": "paper",
                    "technology": "KIVI",
                    "text": "Quantization reduces cache storage while retaining task-specific accuracy.",
                    "page": 1,
                    "section": None,
                },
                {
                    "id": "web:s2:c10",
                    "source_id": "web",
                    "technology": "both",
                    "text": "The snapshot describes an operator-managed host memory offload path.",
                    "page": None,
                    "section": "snapshot block 2, character 10",
                },
            ],
            "sources": {
                "paper": {
                    "type": "paper_pool",
                    "authors": "Paper authors",
                    "title": "Paper title",
                    "url": "https://example.invalid/paper",
                    "version": "v1",
                    "date": "2026",
                    "accessed_at": "2026-10-07T00:00:00Z",
                },
                "web": {
                    "type": "external_web",
                    "authors": "Web authors",
                    "title": "Web title",
                    "url": "https://example.invalid/web",
                    "version": "snapshot-v1",
                    "date": None,
                    "accessed_at": "2026-10-07T00:00:00Z",
                },
            },
            "error": None,
        }
    )


def test_contract_chunk_converts_paper_and_web_locations_without_changing_text():
    pdf = {
        "id": "paper:p3:t40",
        "source_id": "paper",
        "technology": "KIVI",
        "text": "  exact PDF text stays unchanged  ",
        "page": 3,
        "section": "corpus preview is not a contract section",
        "char_start": 40,
        "char_end": 68,
        "token_start": 80,
        "score": 0.75,
    }
    web = {
        "id": "web:s2:c10",
        "source_id": "web",
        "technology": "both",
        "text": "Exact web snapshot text.",
        "page": None,
        "section": "snapshot block 2, character 10",
        "char_start": 10,
    }

    paper_chunk = contract_chunk(pdf)
    web_chunk = contract_chunk(web)

    assert paper_chunk.page == 3
    assert paper_chunk.section is None
    assert paper_chunk.text == pdf["text"]
    assert paper_chunk.model_dump()["char_start"] == 40
    assert paper_chunk.model_dump()["char_end"] == 68
    assert paper_chunk.model_dump()["token_start"] == 80
    assert "score" not in paper_chunk.model_dump()
    assert pdf["score"] == 0.75
    assert web_chunk.page is None
    assert web_chunk.section == web["section"]
    assert web_chunk.text == web["text"]


def test_contract_source_fills_unknown_metadata_and_keeps_hashes():
    source = contract_source(
        {
            "type": "paper_pool",
            "authors": "   ",
            "title": "Paper title",
            "url": "https://example.invalid/paper",
            "version": None,
            "date": None,
            "accessed_at": "2026-10-07T00:00:00Z",
            "sha256": "paper-hash",
            "text_sha256": "text-hash",
            "local_path": "/private/cache/paper.pdf",
        }
    )

    payload = source.model_dump()
    assert payload["authors"] == "저자 미표기"
    assert payload["version"] == "unknown"
    assert payload["sha256"] == "paper-hash"
    assert payload["text_sha256"] == "text-hash"
    assert "local_path" not in payload


def test_contract_source_requires_collection_time():
    with pytest.raises(ValueError, match="accessed_at"):
        contract_source(
            {
                "type": "paper_pool",
                "title": "Paper title",
                "url": "https://example.invalid/paper",
            }
        )


def test_cited_artifacts_selects_unique_referenced_chunks_and_sources():
    assessments = {
        "market": {
            "claims": [
                {
                    "references": [
                        {"chunk_id": "paper:p1:t0", "quote": "first quote"},
                        {"chunk_id": "web:s2:c10", "quote": "second quote"},
                    ]
                },
                {"references": [{"chunk_id": "paper:p1:t0", "quote": "same chunk"}]},
            ]
        }
    }
    chunk_lookup = {
        "paper:p1:t0": {
            "id": "paper:p1:t0",
            "source_id": "paper",
            "technology": "KIVI",
            "text": "Paper chunk text.",
            "page": 1,
            "section": "discarded PDF preview",
            "char_start": 0,
        },
        "web:s2:c10": {
            "id": "web:s2:c10",
            "source_id": "web",
            "technology": "both",
            "text": "Web chunk text.",
            "page": None,
            "section": "snapshot block 2, character 10",
        },
        "uncited": {
            "id": "uncited",
            "source_id": "unused",
            "technology": "KIVI",
            "text": "Not part of this result.",
            "page": 2,
            "section": None,
        },
    }
    source_registry = {
        source_id: {
            "type": source_type,
            "authors": "Author",
            "title": source_id,
            "url": f"https://example.invalid/{source_id}",
            "version": "v1",
            "date": None,
            "accessed_at": "2026-10-07T00:00:00Z",
        }
        for source_id, source_type in (("paper", "paper_pool"), ("web", "external_web"), ("unused", "paper_pool"))
    }

    chunks, sources = cited_artifacts(assessments, chunk_lookup, source_registry)

    assert [chunk.id for chunk in chunks] == ["paper:p1:t0", "web:s2:c10"]
    assert chunks[0].section is None
    assert list(sources) == ["paper", "web"]


def test_cited_artifacts_rejects_missing_chunk_or_source():
    assessments = {"market": {"claims": [{"references": [{"chunk_id": "missing", "quote": "quote"}]}]}}
    with pytest.raises(ValueError, match="missing chunk"):
        cited_artifacts(assessments, {}, {})

    assessments = {"market": {"claims": [{"references": [{"chunk_id": "known", "quote": "quote"}]}]}}
    chunk_lookup = {
        "known": {
            "id": "known",
            "source_id": "missing-source",
            "technology": "KIVI",
            "text": "A sufficiently long source text.",
            "page": 1,
            "section": None,
        }
    }
    with pytest.raises(ValueError, match="missing source"):
        cited_artifacts(assessments, chunk_lookup, {})


def test_valid_result_traces_each_claim_to_a_page_or_section():
    assert trace_errors(_result()) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda result: result["assessments"]["market"]["claims"][0]["references"][0].update(
                quote="This altered quotation is absent from the paper chunk."
            ),
            "quote not found in chunk",
        ),
        (
            lambda result: result["assessments"]["market"]["claims"][0]["references"][0].update(
                chunk_id="missing-chunk"
            ),
            "reference points to missing chunk",
        ),
        (
            lambda result: result["sources"].pop("paper"),
            "missing source paper",
        ),
        (
            lambda result: result["chunks"][0].update(section="unexpected PDF section"),
            "paper_pool chunk must have section=null",
        ),
        (
            lambda result: result["chunks"].append(
                {
                    "id": "uncited:p2",
                    "source_id": "paper",
                    "technology": "KIVI",
                    "text": "An otherwise valid but uncited chunk.",
                    "page": 2,
                    "section": None,
                }
            ),
            "chunk is not cited",
        ),
        (
            lambda result: result["chunks"].append(deepcopy(result["chunks"][0])),
            "duplicate chunk id",
        ),
        (
            lambda result: result["sources"].update(
                {
                    "unused": {
                        "type": "paper_pool",
                        "authors": "Author",
                        "title": "Unused source",
                        "url": "https://example.invalid/unused",
                        "version": "v1",
                        "date": None,
                        "accessed_at": "2026-10-07T00:00:00Z",
                    }
                }
            ),
            "unused source",
        ),
        (
            lambda result: result["chunks"][1].update(page=2),
            "external_web chunk must have page=null",
        ),
        (
            lambda result: result["chunks"][1].update(section="  "),
            "external_web chunk must have a non-blank section",
        ),
    ],
)
def test_trace_errors_reports_each_citation_graph_violation(mutate, expected):
    result = _result().model_dump(mode="python")
    mutate(result)

    assert any(expected in error for error in trace_errors(result))
