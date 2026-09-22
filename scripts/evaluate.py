"""Eight fixed retrieval probes; no paid LLM and no claim of semantic adjudication."""
from pathlib import Path
import sys,json,yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.settings import Settings
from rag.corpus import Corpus
from rag.budget import write_json

c = Corpus(Settings.load(), yaml.safe_load(Path("config/run.yaml").read_text()))
c.prepare_sources()
c.build()
results = []
for q in json.loads(Path("eval/gold.json").read_text()):
    hits = c.search(q["query"], q["technology"], 5)
    matching = [i for i,h in enumerate(hits,1) if h["page"] in q["pages"] and all(m.lower() in h["text"].lower() for m in q["markers"])]
    results.append({**q, "hit":bool(matching),"rank":min(matching) if matching else None,
                    "hits":[{"id":h["id"],"page":h["page"],"score":h["score"],"text":h["text"]} for h in hits]})
out = {"index_hash":c.index_hash,"gold_sha256":__import__('hashlib').sha256(Path('eval/gold.json').read_bytes()).hexdigest(),
       "stage":"initial; no query rewrite", "hits":sum(r["hit"] for r in results),"total":len(results),
       "mrr":sum(1/r["rank"] if r["rank"] else 0 for r in results)/len(results),
       "semantic_judgment":"pending human confirmation of gold spans", "results":results}
write_json("outputs/retrieval-evaluation.json",out)
print(json.dumps({k:v for k,v in out.items() if k!='results'}))
