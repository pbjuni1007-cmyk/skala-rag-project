import pytest
from scripts.evaluate_rewrite import compare_pairs, validate_pairs


def test_replay_reports_regressions_and_keeps_same_topk(monkeypatch):
    chunk = {'id':'evidence', 'source_id':'kivi', 'page':1, 'char_start':0, 'char_end':4, 'text':'fact'}
    class Corpus:
        calls = []
        def search(self, query, technology, k):
            self.calls.append((query, technology, k))
            return [chunk] if query == 'before' else []
        def evidence_tokens(self, text):
            return 1
    monkeypatch.setattr('scripts.evaluate_rewrite.build_research_context', lambda c,r,t,cfg:r[:1])
    case = {'id':'observed', 'technology':'KIVI', **{view:[{'technology':'KIVI','facet':f,'query':view}
            for f in ['mechanism','limitation','conditions','maturity']] for view in ['before','after']}}
    dataset = {'questions':[{'id':'condition','technology':'KIVI','evidence':[
        {'source_id':'kivi','page':1,'start':0,'end':4}]}]}
    corpus = Corpus()
    result = compare_pairs(corpus, {'top_k':5}, [case], dataset)
    assert result['regressed'] == 1 and result['improved'] == 0
    assert result['results'][0]['condition_delta'] == -1
    assert len(corpus.calls) == 8 and all(k == 5 for _,_,k in corpus.calls)
    case['after'][0]['technology'] = 'InfiniGen'
    with pytest.raises(ValueError, match='Technology'):
        validate_pairs([case])


def test_empty_pairs_are_not_reported_as_success():
    with pytest.raises(ValueError, match='No observed'):
        validate_pairs([])
