#!/usr/bin/env python3
"""Offline-capable CPU comparison; public model download is a separate explicit step."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time

MODELS = {
 'e5': {'repo':'intfloat/multilingual-e5-small','revision':'614241f622f53c4eeff9890bdc4f31cfecc418b3','limit':512,'document_prefix':'passage: ','query_prefix':'query: ','license':'mit'},
 'minilm': {'repo':'sentence-transformers/all-MiniLM-L6-v2','revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','limit':256,'document_prefix':'','query_prefix':'','license':'apache-2.0'},
}

def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guard_lengths(tokenizer, texts, limit):
    lengths = [len(tokenizer.encode(t,add_special_tokens=True,truncation=False)) for t in texts]
    if max(lengths,default=0)>limit:
        raise ValueError(f'Input would truncate: {max(lengths)} > {limit}')
    return lengths

def shared_chunks(pages, tokenizers, overlap_words=24):
    """Identical literal word-boundary spans bounded by every tokenizer incl. prefixes."""
    chunks=[]
    for page in pages:
        text=page['text']; words=list(re.finditer(r'\S+',text)); start=0
        while start<len(words):
            lo=start+1; hi=len(words); end=start
            while lo<=hi:
                mid=(lo+hi)//2; span=text[words[start].start():words[mid-1].end()]
                fits=all(len(tok.encode(spec['document_prefix']+span,add_special_tokens=True,truncation=False))<=spec['limit'] for tok,spec in tokenizers)
                if fits: end=mid;lo=mid+1
                else: hi=mid-1
            if end==start: raise ValueError('One source word exceeds a model limit')
            a=words[start].start();b=words[end-1].end()
            chunks.append({'id':f"{page['source_id']}-p{page['page']}-{a}", 'source_id':page['source_id'],'technology':page['technology'],'page':page['page'],'start':a,'end':b,'text':text[a:b]})
            if end==len(words): break
            start=max(start+1,end-overlap_words)
    return chunks

def span_coverage(evidence, chunks):
    intervals=sorted((max(evidence['start'],c['start']),min(evidence['end'],c['end'])) for c in chunks if c['source_id']==evidence['source_id'] and c['page']==evidence['page'] and c['end']>evidence['start'] and c['start']<evidence['end'])
    covered=0; stop=evidence['start']
    for a,b in intervals:
        covered+=max(0,b-max(a,stop));stop=max(stop,b)
    return covered/(evidence['end']-evidence['start'])

def metrics(question, ranked):
    evidence=question['evidence']
    relevant=[any(span_coverage(e,[c])==1 for e in evidence) for c in ranked]
    coverage=[span_coverage(e,ranked) for e in evidence]
    return {'hit_at_5':int(any(relevant)), 'mrr_at_5':next((1/(i+1) for i,r in enumerate(relevant) if r),0), 'span_coverage_at_5':statistics.mean(coverage),'complete_evidence_at_5':int(all(x==1 for x in coverage)), 'required_condition_coverage_at_5':sum(x==1 for x in coverage)/len(coverage), 'page_keyword_proxy_at_5':int(any(c['page'] in question['proxy_pages'] and any(m.lower() in c['text'].lower() for m in question['proxy_markers']) for c in ranked))}

def rank_filtered(chunks, scores, technology, k=5):
    indices=[i for i,c in enumerate(chunks) if c['technology']==technology]
    return sorted(indices,key=lambda i:(-float(scores[i]),i))[:k]

def validate_dataset(dataset,pages):
    lookup={(p['source_id'],p['page']):p for p in pages}
    for q in dataset['questions']:
        for e in q['evidence']:
            p=lookup[e['source_id'],e['page']]
            if p['text'][e['start']:e['end']]!=e['quote']: raise ValueError('Ground truth source mismatch')
            if p['technology']!=q['technology']: raise ValueError('Ground truth technology mismatch')
        if q['required_conditions']!=[e['quote'] for e in q['evidence']]: raise ValueError('Conditions must map to scored evidence spans')

def model_path(cache,key):
    s=MODELS[key]
    return Path(cache)/('models--'+s['repo'].replace('/','--'))/'snapshots'/s['revision']

def download(args):
    from huggingface_hub import snapshot_download
    log={}
    for key,s in MODELS.items():
        start=time.perf_counter()
        path=snapshot_download(s['repo'],revision=s['revision'],cache_dir=args.cache,token=False,allow_patterns=['*.json','*.txt','*.model','*.safetensors','README.md','1_Pooling/*'],ignore_patterns=['onnx/*','openvino/*'])
        log[key]={'seconds':time.perf_counter()-start,'snapshot_bytes':sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()),'cache_reuse_possible':True}
    write(args.output/'download.json',log)

def prepare(args):
    from transformers import AutoTokenizer
    pages=read(args.pages); dataset=read(args.dataset);validate_dataset(dataset,pages)
    tokenizers=[(AutoTokenizer.from_pretrained(model_path(args.cache,k),local_files_only=True),s) for k,s in MODELS.items()]
    chunks=shared_chunks(pages,tokenizers)
    for tok,s in tokenizers:
        guard_lengths(tok,[s['document_prefix']+c['text'] for c in chunks],s['limit'])
        guard_lengths(tok,[s['query_prefix']+q['queries'][lang] for q in dataset['questions'] for lang in ['en','ko']],s['limit'])
    write(args.output/'chunks.json',chunks)
    write(args.output/'protocol.json',{'dataset_sha256':sha(args.dataset),'pages_sha256':sha(args.pages),'chunks_sha256':sha(args.output/'chunks.json'),'page_count':len(pages),'chunk_count':len(chunks),'models':MODELS,'overlap_words':24,'threads':2,'query_repeats':3,'independent_questions':len(dataset['questions']),'language_queries':2*len(dataset['questions'])})

def run_model(args):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    s=MODELS[args.model];chunks=read(args.output/'chunks.json');dataset=read(args.dataset)
    start=time.perf_counter();model=SentenceTransformer(str(model_path(args.cache,args.model)),device='cpu',local_files_only=True);load_s=time.perf_counter()-start
    if model.max_seq_length!=s['limit']: raise ValueError('Unexpected model maximum sequence length')
    docs=[s['document_prefix']+c['text'] for c in chunks];lens=guard_lengths(model.tokenizer,docs,s['limit'])
    start=time.perf_counter();vectors=model.encode(docs,batch_size=16,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True);index_s=time.perf_counter()-start
    model.encode([s['query_prefix']+'warm up'],normalize_embeddings=True)
    results=[]
    for q in dataset['questions']:
        for lang,query in q['queries'].items():
            text=s['query_prefix']+query;guard_lengths(model.tokenizer,[text],s['limit']);times=[]
            for _ in range(3):
                start=time.perf_counter();v=model.encode([text],normalize_embeddings=True,show_progress_bar=False)[0];scores=vectors@v;indices=rank_filtered(chunks,scores,q['technology']);times.append(time.perf_counter()-start)
            ranked=[chunks[i] for i in indices]
            results.append({'id':q['id'],'subset':q['subset'],'language':lang,'query':query,'metrics':metrics(q,ranked),'query_seconds':times,'top5':[dict(chunks[i],cosine=float(scores[i])) for i in indices]})
    summary={}
    for lang in ['en','ko']:
        for subset in ['all','existing','new']:
            rows=[r for r in results if r['language']==lang and (subset=='all' or r['subset']==subset)]
            summary[f'{lang}_{subset}']={'n':len(rows),**{m:statistics.mean(r['metrics'][m] for r in rows) for m in rows[0]['metrics']},'warm_query_median_seconds':statistics.median(t for r in rows for t in r['query_seconds'])}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(args.output/f'{args.model}.json',{'model':s,'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'packages':{p:importlib.metadata.version(p) for p in ['sentence-transformers','transformers','torch','numpy','huggingface-hub']}},'dataset_sha256':sha(args.dataset),'chunks_sha256':sha(args.output/'chunks.json'),'model_load_seconds':load_s,'index_build_seconds':index_s,'process_peak_rss_bytes':rss if sys.platform=='darwin' else rss*1024,'vector_dimension':vectors.shape[1],'vector_bytes':vectors.nbytes,'max_document_tokens':max(lens),'summary':summary,'results':results})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',default='.cache/embedding-benchmark');p.add_argument('--pages',default='data/pages.json');p.add_argument('--dataset',default='eval/embedding-comparison.json');p.add_argument('--output',type=Path,default=Path('outputs/embedding-benchmark'));p.add_argument('--download',action='store_true');p.add_argument('--model',choices=list(MODELS));args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.download: download(args);return
    if args.model: run_model(args);return
    prepare(args)
    for key in MODELS:
        subprocess.run([sys.executable,__file__,'--cache',args.cache,'--pages',args.pages,'--dataset',args.dataset,'--output',str(args.output),'--model',key],check=True,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'})
    print(json.dumps({k:read(args.output/f'{k}.json')['summary'] for k in MODELS},indent=2))

if __name__=='__main__': main()
