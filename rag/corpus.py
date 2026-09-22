from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import urllib.parse
import numpy as np
from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag.budget import write_json

PAPER_POOL = {"2402.02750v2", "2406.19707v1"}
EMBEDDING_PROFILES = {
    "intfloat/multilingual-e5-small": {
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "max_tokens": 512, "query_prefix": "query: ", "passage_prefix": "passage: ",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "max_tokens": 256, "query_prefix": "", "passage_prefix": "",
    },
}
WEB_URLS = {
    "https://raw.githubusercontent.com/jy-yuan/KIVI/main/README.md",
    "https://raw.githubusercontent.com/snu-comparch/InfiniGen/main/README.md",
    "https://huggingface.co/docs/transformers/kv_cache",
    "https://www.skax.co.kr/ax-services/aipmo",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def sentence_windows(text, target=1200):
    """Contiguous source windows, never a hard cut inside a long sentence.

    The target is soft: a single long sentence stays intact. One complete
    sentence overlaps between windows so comparison clauses keep context.
    """
    spans = [(m.start(), m.end()) for m in re.finditer(r'.+?(?:[.!?。！？][\"\')\]]*\s+|$)', text) if m.group().strip()]
    start = 0
    while start < len(spans):
        end = start + 1
        while end < len(spans) and spans[end][1] - spans[start][0] <= target:
            end += 1
        left, right = spans[start][0], spans[end - 1][1]
        yield left, text[left:right].strip()
        if end == len(spans):
            break
        start = max(start + 1, end - 1)


def download(url, timeout):
    from rag.source_discovery import fetch
    parsed = urllib.parse.urlsplit(url)
    # Exact registered entrypoint boundary for every redirect, before following it.
    policy = {"allow": [{"host": parsed.hostname, "paths": [parsed.path]}]}
    return fetch(url, policy, 5_000_000 if url in WEB_URLS else 25_000_000, timeout)[0]


class Corpus:
    def __init__(self, settings, config):
        self.settings, self.config = settings, config
        self.root = Path(settings.get("RAG_DATA_DIR", "data"))
        self.index = Path(settings.get("RAG_INDEX_DIR", ".cache/rag-index"))
        self.web_root = Path(settings.get("RAG_WEB_SNAPSHOT_DIR", "data/web_snapshots"))
        self.sources, self.chunks = {}, []
        self.frozen = False

    def prepare_sources(self, refresh_web=False):
        if self.frozen:
            raise ValueError("Sources are frozen during generation")
        now = datetime.now(timezone.utc).isoformat()
        pages = []
        seen = set()
        for item in self.config["papers"]:
            version = item["version"]
            if version not in PAPER_POOL or version in seen or item["url"] != f"https://arxiv.org/pdf/{version}":
                raise ValueError("Only the approved versioned PDF pool is permitted")
            seen.add(version)
            path = self.root / "papers" / f"{version}.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(download(item["url"], self.settings.integer("SOURCE_HTTP_TIMEOUT_SECONDS", 30)))
            reader = PdfReader(path)
            for number, page in enumerate(reader.pages, 1):
                text = normalized(page.extract_text(extraction_mode="plain") or "")
                if not text:
                    raise ValueError(f"Empty PDF page {version}:{number}; needs source inspection")
                pages.append({"source_id": item["id"], "technology": item["technology"], "page": number, "text": text})
            self.sources[item["id"]] = {**item, "type": "paper_pool", "pages": len(reader.pages),
                "sha256": digest(path.read_bytes()), "local_path": str(path), "accessed_at": now}
        if seen != PAPER_POOL or len(pages) > 200:
            raise ValueError("Both approved papers and a maximum of 200 total pages are required")
        write_json(self.root / "pages.json", pages)
        for item in self.config["web"]:
            if item["url"] not in WEB_URLS:
                raise ValueError("Unregistered source URL")
            path = self.web_root / (item["id"] + ".json")
            if path.exists() and not refresh_web:
                snapshot = json.loads(path.read_text())
                if snapshot["url"] != item["url"] or digest(snapshot["text"].encode()) != snapshot["text_sha256"]:
                    raise ValueError("Web snapshot integrity mismatch")
            else:
                try:
                    raw = download(item["url"], self.settings.integer("SOURCE_HTTP_TIMEOUT_SECONDS", 30))
                    raw_path = self.web_root / (item["id"] + ".raw")
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_path.write_bytes(raw)
                    decoded = raw.decode("utf-8")
                    if item["url"].endswith(".md"):
                        text = decoded
                    else:
                        soup = BeautifulSoup(decoded, "html.parser")
                        for node in soup(["script", "style", "nav", "footer", "header"]):
                            node.decompose()
                        text = (soup.find("main") or soup).get_text("\n", strip=True)
                    snapshot = {**item, "text": text, "text_sha256": digest(text.encode()),
                                "raw_sha256": digest(raw), "raw_path": str(raw_path), "accessed_at": now, "status": "ok"}
                    write_json(path, snapshot)
                except Exception as exc:
                    # Only exception class is retained; never log arbitrary remote content.
                    snapshot = {**item, "text": "", "accessed_at": now, "status": "unavailable", "error": type(exc).__name__}
                    if path.exists():
                        cached = json.loads(path.read_text())
                        if cached.get("url") == item["url"] and cached.get("status") == "ok" and digest(cached["text"].encode()) == cached.get("text_sha256"):
                            snapshot = {**cached, "refresh_failed_at": now, "freshness": "cached_after_refresh_failure",
                                        "refresh_error": type(exc).__name__}
            self.sources[item["id"]] = {k: v for k, v in snapshot.items() if k != "text"}
            self.sources[item["id"]].update(type="external_web", date="unknown", version=snapshot.get("text_sha256", "unknown"), local_path=str(path))
        discovery_path = Path(self.config.get("source_discovery_config", "config/source-discovery.yaml"))
        if discovery_path.exists():
            import yaml
            from rag.source_discovery import Discovery
            self.discovery_config = yaml.safe_load(discovery_path.read_text())
            self.sources.update(Discovery(self.discovery_config, self.web_root).collect(self.sources, refresh_web))
        write_json(self.root / "source_manifest.json", {"total_pdf_pages": len(pages), "sources": self.sources})
        self.pages = pages
        return len(pages)

    def retrieval_identity(self):
        from rag.context import policy_identity
        return {"sources": {key: {k: value.get(k) for k in
                    ("url", "sha256", "raw_sha256", "text_sha256", "version", "status")}
                for key, value in sorted(self.sources.items())},
                "discovery": getattr(self, "discovery_config", {}),
                "context": policy_identity(self.config),
                "embedding": getattr(self, "embedding", {}),
                "chunk_tokens": self.config.get("chunk_tokens"), "overlap_tokens": self.config.get("overlap_tokens")}

    def freeze_sources(self):
        # Keep verified extracted text in memory: generation cannot observe later disk edits.
        self.frozen_snapshots = {}
        for source_id, source in self.sources.items():
            if source.get("type") != "external_web" or source.get("status") != "ok":
                continue
            snapshot = json.loads(Path(source["local_path"]).read_text())
            if digest(snapshot["text"].encode()) != source.get("text_sha256"):
                raise ValueError("Frozen web snapshot integrity mismatch")
            if source.get("raw_path") and digest(Path(source["raw_path"]).read_bytes()) != source.get("raw_sha256"):
                raise ValueError("Frozen web raw integrity mismatch")
            self.frozen_snapshots[source_id] = snapshot
        self.frozen = True
        self.frozen_identity = digest(json.dumps(self.retrieval_identity(), sort_keys=True).encode())

    def assert_frozen(self):
        if self.frozen and self.frozen_identity != digest(json.dumps(self.retrieval_identity(), sort_keys=True).encode()):
            raise ValueError("Frozen source or retrieval configuration changed")

    def source_metadata(self):
        return {key: {k: value.get(k) for k in ("title", "url", "version", "date", "sha256", "text_sha256")}
                for key, value in self.sources.items()}

    def adjacent_candidates(self, hits):
        pages = {(c["source_id"], c.get("page")) for c in hits if c.get("page")}
        candidates = [c for c in self.chunks if any(c["source_id"] == source and
                      abs(c.get("page", -999) - page) == 1 for source, page in pages)]
        # Round-robin neighbor pages prevents one long early page consuming the reservation.
        groups = {}
        for chunk in candidates:
            groups.setdefault((chunk["source_id"], chunk["page"]), []).append(chunk)
        def priority(item):
            source, page = item[0]
            neighbor_rank = min(i for i, hit in enumerate(hits) if hit["source_id"] == source
                                and hit.get("page") and abs(hit["page"] - page) == 1)
            has_caption = any(c.get("captions") or re.search(r"(?:Table|Figure)\s+\d+", c["text"])
                              for c in item[1])
            return (neighbor_rank, not has_caption, page)
        ordered = [value for _, value in sorted(groups.items(), key=priority)]
        from itertools import zip_longest
        return [c for row in zip_longest(*ordered) for c in row if c]

    def evidence_tokens(self, text):
        # The budget uses the fixed operating E5 tokenizer, including for alternate encoders.
        if not hasattr(self, "budget_tokenizer"):
            if self.embedding["model"] == "intfloat/multilingual-e5-small":
                self.budget_tokenizer = self.encoder.tokenizer
            else:
                from transformers import AutoTokenizer
                self.budget_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small",
                    revision=EMBEDDING_PROFILES["intfloat/multilingual-e5-small"]["revision"],
                    local_files_only=True, trust_remote_code=False)
        return len(self.budget_tokenizer.encode(text, truncation=False))

    def load_encoder(self):
        import os
        os.environ["HF_HOME"] = self.settings.get("HF_HOME", ".cache/huggingface")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        model = self.settings.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        if model not in EMBEDDING_PROFILES:
            raise ValueError("Embedding change needs a new retrieval evaluation")
        profile = EMBEDDING_PROFILES[model]
        manifest_path = self.index / "embedding.json"
        pinned = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        revision = self.settings.get("EMBEDDING_REVISION") or (
            pinned.get("revision") if pinned.get("model") == model else None) or profile["revision"]
        from huggingface_hub.errors import LocalEntryNotFoundError
        download_options = {
            "cache_dir": str(Path(self.settings.get("HF_HOME", ".cache/huggingface")) / "hub"),
            "allow_patterns": ["*.json", "*.txt", "*.model", "*.safetensors", "README.md", "1_Pooling/*"],
            "ignore_patterns": ["onnx/*", "openvino/*"], "token": False,
        }
        try:
            location = snapshot_download(model, revision=revision, local_files_only=True, **download_options)
        except LocalEntryNotFoundError:
            location = snapshot_download(model, revision=revision, **download_options)
        self.encoder = SentenceTransformer(location, device=self.settings.get("EMBEDDING_DEVICE", "cpu"), trust_remote_code=False)
        if self.encoder.max_seq_length < profile["max_tokens"]:
            raise ValueError("Encoder limit differs from the reviewed embedding profile")
        self.embedding = {"model": model, "revision": revision, "dimension": self.encoder.get_embedding_dimension(),
                          "max_tokens": profile["max_tokens"], "normalized": True,
                          "prefixes": [profile["query_prefix"], profile["passage_prefix"]]}
        write_json(manifest_path, self.embedding)

    def build(self):
        self.load_encoder()
        size, overlap = int(self.config["chunk_tokens"]), int(self.config["overlap_tokens"])
        if not 100 <= size <= 400 or not 0 <= overlap < size:
            raise ValueError("Chunk contract requires 100..400 tokens and smaller overlap")
        tokenizer = self.encoder.tokenizer
        chunks = []
        for page in self.pages:
            # Token offsets preserve exact extracted source substrings.
            offsets = tokenizer(page["text"], add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
            for start in range(0, len(offsets), size - overlap):
                end = min(start + size, len(offsets))
                text = page["text"][offsets[start][0]:offsets[end - 1][1]]
                chunk = {**page, "text": text, "section": page["text"][:160],
                         "id": f"{page['source_id']}:p{page['page']}:t{start}", "token_start": start,
                         "char_start": offsets[start][0], "char_end": offsets[end - 1][1],
                         "captions": re.findall(r"(?:Figure|Table)\s+\d+[^.]{0,120}", text)}
                if len(tokenizer.encode(self.embedding["prefixes"][1] + text, truncation=False)) > self.embedding["max_tokens"]:
                    raise ValueError("Embedding chunk would be truncated")
                chunks.append(chunk)
        self.chunks = chunks
        key = digest(json.dumps({"chunks": chunks, "embedding": self.embedding}, sort_keys=True).encode())
        meta = self.index / "index.json"
        matrix = self.index / "vectors.npy"
        if meta.exists() and matrix.exists() and json.loads(meta.read_text()).get("hash") == key:
            self.vectors = np.load(matrix, allow_pickle=False)
        else:
            self.vectors = self.encoder.encode([self.embedding["prefixes"][1] + c["text"] for c in chunks], normalize_embeddings=True,
                batch_size=self.settings.integer("EMBEDDING_BATCH_SIZE", 16), show_progress_bar=False)
            self.index.mkdir(parents=True, exist_ok=True)
            np.save(matrix, self.vectors)
            write_json(meta, {"hash": key, "embedding": self.embedding, "chunks": chunks})
        self.index_hash = key
        return {"pages": len(self.pages), "chunks": len(chunks), "embedding": self.embedding, "index_hash": key}

    def search(self, query, technology, top_k=5):
        self.assert_frozen()
        text = self.embedding["prefixes"][0] + query
        if len(self.encoder.tokenizer.encode(text, truncation=False)) > self.embedding["max_tokens"]:
            raise ValueError("Embedding query would be truncated")
        q = self.encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        eligible = [i for i, c in enumerate(self.chunks) if c["technology"] == technology]
        scores = self.vectors[eligible] @ q
        order = np.argsort(-scores)[:top_k]
        return [{**self.chunks[eligible[i]], "score": float(scores[i])} for i in order]

    def web_search(self, query, top_k=5, expanded_only=False):
        """Bounded lexical search of registered snapshots; never claims global coverage."""
        self.assert_frozen()
        terms = set(re.findall(r"[\w가-힣]{2,}", query.lower()))
        matches = []
        for source_id, source in self.sources.items():
            if source["type"] != "external_web" or source.get("status") != "ok":
                continue
            if expanded_only and not source.get("expanded"):
                continue
            snapshot = self.frozen_snapshots[source_id] if self.frozen else json.loads(Path(source["local_path"]).read_text())
            if digest(snapshot["text"].encode()) != source.get("text_sha256"):
                raise ValueError("Frozen web snapshot changed")
            blocks = [normalized(s) for s in re.split(r"\n\s*\n|\n", snapshot["text"]) if normalized(s)]
            for i, block in enumerate(blocks):
                for offset, part in sentence_windows(block):
                    score = sum(term in part.lower() for term in terms)
                    if score:
                        matches.append({"id": f"{source_id}:s{i}:c{offset}", "source_id": source_id, "text": part,
                                        "section": f"snapshot block {i}, character {offset}", "page": None,
                                        "technology": "both", "score": score})
        ranked = sorted(matches, key=lambda c: (-c["score"], c["id"]))
        # Include the best matching block of each registered source before filling.
        chosen, seen = [], set()
        for chunk in ranked:
            if chunk["source_id"] not in seen:
                chosen.append(chunk)
                seen.add(chunk["source_id"])
        chosen.extend(c for c in ranked if c not in chosen)
        return chosen[:top_k]
