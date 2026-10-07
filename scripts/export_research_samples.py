"""Export saved legacy research nodes as deterministic agent-contract samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.researchers.contract import (  # noqa: E402
    CONTRACT_VERSION,
    ResearchResult,
    RunContext,
)
from agents.researchers.result import cited_artifacts, contract_chunk  # noqa: E402
from rag.schemas import Assessment  # noqa: E402


VIEWS = ("research", "market", "stakeholder", "domain")
RESEARCH_NODES = {
    "research_kivi": "research_kivi",
    "research_infinigen": "research_infinigen",
}
ASSESSMENT_FIELDS = ("status", "claims", "conflicts", "gaps")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc


def _check_index_hash(manifest: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    expected = manifest.get("index", {}).get("index_hash")
    actual = index.get("hash")
    chunks = index.get("chunks")
    embedding = index.get("embedding")
    if not isinstance(expected, str) or not expected:
        raise ValueError("Run manifest is missing index.index_hash")
    if not isinstance(actual, str) or not actual or not isinstance(chunks, list) or not isinstance(embedding, dict):
        raise ValueError("RAG index is missing its hash, chunks, or embedding metadata")

    # Match rag.corpus.Corpus.build's canonical hash calculation.
    calculated = hashlib.sha256(
        json.dumps({"chunks": chunks, "embedding": embedding}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if actual != expected or calculated != actual:
        raise ValueError("RAG index hash does not match the run manifest")


def _assessment(node: Mapping[str, Any], name: str) -> dict[str, Any]:
    missing = [field for field in ASSESSMENT_FIELDS if field not in node]
    if missing:
        raise ValueError(f"{name} is missing Assessment fields: {', '.join(missing)}")
    return Assessment.model_validate({field: node[field] for field in ASSESSMENT_FIELDS}).model_dump(mode="json")


def _insert_chunk(lookup: dict[str, dict[str, Any]], chunk: Mapping[str, Any]) -> None:
    chunk_id = chunk.get("id")
    source_id = chunk.get("source_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("A collection chunk is missing its id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"Chunk {chunk_id} is missing its source_id")

    current = dict(chunk)
    previous = lookup.get(chunk_id)
    if previous is not None:
        previous_contract = contract_chunk(previous).model_dump(mode="json")
        current_contract = contract_chunk(current).model_dump(mode="json")
        # Scores depend on the query that found a chunk; compare every persisted
        # contract/collection field, including hashes and offsets, as content identity.
        if previous_contract != current_contract:
            raise ValueError(f"Conflicting duplicate source chunk: {chunk_id}")
        return
    lookup[chunk_id] = current


def _chunk_lookup(
    node: Mapping[str, Any],
    index: Mapping[str, Any],
    source_registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for chunk in index["chunks"]:
        if not isinstance(chunk, dict):
            raise ValueError("RAG index contains a malformed chunk")
        source = source_registry.get(chunk.get("source_id"))
        if isinstance(source, dict) and source.get("type") == "paper_pool":
            _insert_chunk(lookup, chunk)

    searched = node.get("searched_chunks", [])
    if not isinstance(searched, list):
        raise ValueError("searched_chunks must be a list")
    for chunk in searched:
        if not isinstance(chunk, dict):
            raise ValueError("searched_chunks contains a malformed chunk")
        source = source_registry.get(chunk.get("source_id"))
        if source is None:
            raise ValueError(f"Chunk {chunk.get('id', '<unknown>')} has no source record")
        if source.get("type") == "external_web":
            _insert_chunk(lookup, chunk)
    return lookup


def _make_result(
    view: str,
    run_id: str,
    context: RunContext,
    run_dir: Path,
    index: Mapping[str, Any],
    source_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if view == "research":
        raw_assessments = {
            result_key: _assessment(_load_json(run_dir / "nodes" / f"{node_name}.json"), node_name)
            for result_key, node_name in RESEARCH_NODES.items()
        }
        raw_nodes = [_load_json(run_dir / "nodes" / f"{name}.json") for name in RESEARCH_NODES.values()]
        status = "ok" if all(node.get("status") == "ok" for node in raw_nodes) else "insufficient"
        artifact_node: Mapping[str, Any] = {}
    else:
        artifact_node = _load_json(run_dir / "nodes" / f"{view}.json")
        raw_assessments = {view: _assessment(artifact_node, view)}
        status = artifact_node.get("status")

    if status not in {"ok", "insufficient"}:
        raise ValueError(f"{view} has unsupported legacy result status: {status!r}")

    chunk_lookup = _chunk_lookup(artifact_node, index, source_registry)
    chunks, sources = cited_artifacts(raw_assessments, chunk_lookup, source_registry)
    result = ResearchResult.model_validate(
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "request_id": f"{run_id}:{view}:1",
            "attempt": 1,
            "context": context.model_dump(mode="json"),
            "view": view,
            "status": status,
            "assessments": raw_assessments,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "sources": {source_id: source.model_dump(mode="json") for source_id, source in sources.items()},
            "error": None,
        }
    )
    return result.model_dump(mode="json")


def _readme(run_id: str) -> str:
    return (
        "# ResearchResult 샘플\n\n"
        f"이 폴더의 네 JSON은 저장된 레거시 파이프라인 실행 `{run_id}`를 "
        "`agent-contract-v1` 형식으로 변환한 자료입니다. 새 ResearchAgent의 실시간 실행 결과가 아닙니다.\n\n"
        "논문 청크는 실행 manifest가 가리키는 해시와 일치하는 로컬 RAG 인덱스에서 가져왔고, "
        "웹 청크는 해당 실행의 저장된 노드 결과에서 가져왔습니다. 출처 메타데이터는 실행의 `sources.json`에서 "
        "변환했습니다. 이 실행에 사용된 자료는 공개 논문과 공개 웹 스냅샷이며 내부 문서는 포함하지 않습니다.\n\n"
        "샘플을 다시 만들려면 저장된 실행 디렉터리와 출력 디렉터리를 지정해 다음 명령을 실행합니다.\n\n"
        "```sh\n"
        ".venv/bin/python scripts/export_research_samples.py "
        f"outputs/{run_id} tests/fixtures/research\n"
        "```\n"
    )


def export_samples(run_dir: Path, out_dir: Path, index_path: Path | None = None) -> list[Path]:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    index_path = (index_path or ROOT / ".cache/rag-index/index.json").resolve()

    manifest = _load_json(run_dir / "manifest.json")
    index = _load_json(index_path)
    _check_index_hash(manifest, index)

    run_id = manifest.get("run_id")
    config = manifest.get("config")
    if not isinstance(run_id, str) or not run_id.strip() or not isinstance(config, dict):
        raise ValueError("Run manifest is missing run_id or config")
    context = RunContext.model_validate(
        {
            "technologies": config.get("technologies"),
            "domain": config.get("domain"),
            "scenario": config.get("scenario"),
        }
    )
    source_registry = _load_json(run_dir / "sources.json")
    if not isinstance(source_registry, dict):
        raise ValueError("sources.json must contain an object")

    results = {
        view: _make_result(view, run_id, context, run_dir, index, source_registry)
        for view in VIEWS
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for view, result in results.items():
        path = out_dir / f"{view}.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    readme = out_dir / "README.md"
    readme.write_text(_readme(run_id), encoding="utf-8")
    written.append(readme)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="saved legacy pipeline run directory")
    parser.add_argument("out_dir", type=Path, help="directory for contract JSON samples")
    args = parser.parse_args()
    try:
        written = export_samples(args.run_dir, args.out_dir)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.exit(2, f"export failed: {type(exc).__name__}: {exc}\n")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
