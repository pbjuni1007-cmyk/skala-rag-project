import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).parents[1]/'scripts/benchmark_embeddings.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class Tokenizer:
    def encode(self,text,**kwargs): return list(range(len(text.split())+2))

class BenchmarkTests(unittest.TestCase):
    def test_union_coverage_does_not_count_overlap_twice_or_other_pages(self):
        e={'source_id':'s','page':1,'start':10,'end':30}
        chunks=[dict(source_id='s',page=1,start=5,end=22),dict(source_id='s',page=1,start=18,end=25),dict(source_id='s',page=2,start=25,end=30)]
        self.assertEqual(b.span_coverage(e,chunks),.75)
        chunks.append(dict(source_id='s',page=1,start=25,end=35))
        self.assertEqual(b.span_coverage(e,chunks),1)
    def test_mrr_requires_entire_span_but_union_can_complete(self):
        e={'source_id':'s','page':1,'start':0,'end':10}
        q={'evidence':[e],'proxy_pages':[1],'proxy_markers':['a']}
        chunks=[dict(source_id='s',page=1,start=0,end=5,text='a'),dict(source_id='s',page=1,start=5,end=10,text='a')]
        m=b.metrics(q,chunks)
        self.assertEqual(m['hit_at_5'],0);self.assertEqual(m['complete_evidence_at_5'],1)
        chunks.append(dict(source_id='s',page=1,start=0,end=10,text='a'))
        self.assertEqual(b.metrics(q,chunks)['mrr_at_5'],1/3)
    def test_filter_precedes_top_k(self):
        chunks=[{'technology':'other'},{'technology':'wanted'},{'technology':'wanted'}]
        self.assertEqual(b.rank_filtered(chunks,[.99,.3,.7],'wanted',1),[2])
    def test_common_chunks_keep_source_offsets_and_fit_both(self):
        text='one two three four five six seven eight nine ten'
        p={'source_id':'s','page':1,'technology':'T','text':text}
        configs=[(Tokenizer(),{'document_prefix':'passage: ','limit':8}),(Tokenizer(),{'document_prefix':'','limit':6})]
        chunks=b.shared_chunks([p],configs,overlap_words=1)
        self.assertGreater(len(chunks),1)
        for c in chunks:
            self.assertEqual(c['text'],text[c['start']:c['end']])
            for tok,s in configs: b.guard_lengths(tok,[s['document_prefix']+c['text']],s['limit'])
        self.assertEqual(b.span_coverage(dict(source_id='s',page=1,start=0,end=len(text)),chunks),1)
    def test_truncation_is_failure(self):
        with self.assertRaises(ValueError): b.guard_lengths(Tokenizer(),['one two three'],4)
    def test_ground_truth_matches_current_pages(self):
        root=Path(__file__).parents[1]
        b.validate_dataset(b.read(root/'eval/embedding-comparison.json'),b.read(root/'data/pages.json'))

if __name__=='__main__': unittest.main()
