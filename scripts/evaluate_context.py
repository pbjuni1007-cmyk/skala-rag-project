#!/usr/bin/env python3
"""Offline known-item diagnosis of four-facet operational context (not holdout)."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.context import build_research_context, assessment_evidence, policy_identity
from rag.corpus import Corpus


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_context(corpus, ranked, config):
    chosen, seen, used = [], set(), 0
    for items, cap in [(corpus.adjacent_candidates(ranked), config['adjacent_token_budget']), (ranked, config['evidence_token_budget'])]:
        spent = 0
        for c in items:
            cost = corpus.evidence_tokens(c['text'])
            if c['id'] not in seen and spent+cost <= cap and used+cost <= config['evidence_token_budget']:
                chosen.append(c);seen.add(c['id']);spent+=cost;used+=cost
    return chosen


def recover(evidence, chunks):
    intervals = sorted((max(evidence['start'],c['char_start']), min(evidence['end'],c['char_end'])) for c in chunks
                       if c['source_id']==evidence['source_id'] and c['page']==evidence['page']
                       and c['char_start']<evidence['end'] and c['char_end']>evidence['start'])
    covered, stop = 0, evidence['start']
    for a,b in intervals:
        covered += max(0,b-max(a,stop));stop=max(stop,b)
    return covered==evidence['end']-evidence['start']


def evaluate(dataset, contexts):
    return [{ 'id':q['id'],'group':q['group'],'technology':q['technology'],'required':q['required'],
              'complete': int(all(recover(e,contexts[q['technology']]) for e in q['evidence'])),
              'condition_recovery':statistics.mean(recover(e,contexts[q['technology']]) for e in q['evidence']),
              'spans':[recover(e,contexts[q['technology']]) for e in q['evidence']]} for q in dataset['questions']]


def verify_dataset(dataset,pages):
    lookup={(p['source_id'],p['page']):p['text'] for p in pages}
    for q in dataset['questions']:
        for e in q['evidence']:
            text=lookup[e['source_id'],e['page']]
            if text[e['start']:e['end']]!=e['quote'] or hashlib.sha256(text.encode()).hexdigest()!=e['page_sha256'] or hashlib.sha256(e['quote'].encode()).hexdigest()!=e['quote_sha256']:
                raise ValueError('Diagnostic source span/hash mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,default=Path('.cache/rag-index'))
    p.add_argument('--cache',type=Path,default=Path('.cache/huggingface/hub'))
    p.add_argument('--dataset',type=Path,default=Path('eval/context-conditions.json'))
    p.add_argument('--pages',type=Path,default=Path('data/pages.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]='2'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    dataset=read(args.dataset);pages=read(args.pages);verify_dataset(dataset,pages)
    if sha(args.pages)!=dataset['pages_sha256']:raise ValueError('Pages changed')
    index=read(args.index/'index.json');embedding=index['embedding']
    page_text={(p['source_id'],p['page']):p['text'] for p in pages}
    for c in index['chunks']:
        if page_text[c['source_id'],c['page']][c['char_start']:c['char_end']] != c['text']:
            raise ValueError('Index full chunk differs from original page')
    if embedding['model']!='intfloat/multilingual-e5-small':raise ValueError('Diagnostic requires reviewed E5 index')
    config={'evidence_token_budget':6500,'adjacent_token_budget':2800,'assessment_evidence_token_budget':6500}
    corpus=Corpus.__new__(Corpus);corpus.config=config;corpus.chunks=index['chunks'];corpus.embedding=embedding;corpus.frozen=False
    corpus.encoder=SentenceTransformer(str(args.cache/('models--'+embedding['model'].replace('/','--'))/'snapshots'/embedding['revision']),local_files_only=True,trust_remote_code=False,device='cpu')
    corpus.vectors=np.load(args.index/'vectors.npy',allow_pickle=False)
    old,new,rankings={},{},{}
    for technology,queries in dataset['queries'].items():
        groups=[corpus.search(q['query'],technology,5) for q in queries]
        ranked=[c for row in zip_longest(*groups) for c in row if c]
        rankings[technology]=[{'facet':q['facet'],'hits':[{'id':c['id'],'score':c['score']} for c in group]} for q,group in zip(queries,groups)]
        old[technology]=legacy_context(corpus,ranked,config);new[technology]=build_research_context(corpus,ranked,technology,config)
    cited_ids={r['chunk_id'] for a in dataset['assessments'].values() for c in a['claims'] for r in c['references']}
    cited=[c for c in corpus.chunks if c['id'] in cited_ids];enhanced=assessment_evidence(corpus,dataset['assessments'],config)
    views={'research_before':old,'research_after':new,'assessment_before':{t:[c for c in cited if c['technology']==t] for t in old},'assessment_after':{t:[c for c in enhanced if c['technology']==t] for t in old}}
    results={name:evaluate(dataset,ctx) for name,ctx in views.items()}
    regressions=[q['id'] for q,n in zip(results['research_before'],results['research_after']) if q['required'] and n['complete']<q['complete']]
    output={'diagnostic_only':True,'human_review_pending':True,'policy':policy_identity(config),'dataset_sha256':sha(args.dataset),'pages_sha256':sha(args.pages),'index_sha256':sha(args.index/'index.json'),'ranking':rankings,'contexts':{name:{t:{'chunk_ids':[c['id'] for c in cs],'e5_tokens':sum(corpus.evidence_tokens(c['text']) for c in cs)} for t,cs in ctx.items()} for name,ctx in views.items()},'results':results,'required_regressions':regressions,'assessment_required_regressions':[q['id'] for q,n in zip(results['assessment_before'],results['assessment_after']) if q['required'] and n['complete']<q['complete']], 'assessment_total_e5_tokens':sum(corpus.evidence_tokens(c['text']) for c in enhanced)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'required_regressions':regressions,'scores':{name:{group:statistics.mean(r['complete'] for r in rows if r['group']==group) for group in ('legacy_registered_spans','known_condition_diagnostic')} for name,rows in results.items()},'assessment_tokens':output['assessment_total_e5_tokens']},indent=2))


if __name__=='__main__':main()
