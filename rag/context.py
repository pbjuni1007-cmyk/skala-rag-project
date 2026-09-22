"""Bounded full-chunk experiment context; no paper-specific facts or page rules."""
import re

CONTEXT_POLICY = 'full-chunk-caption-setup-neighbors-v2'
CAPTION = re.compile(r'\b(?:Figure|Table)\s+\d+\s*[:.]', re.I)
STRONG_SETUP = re.compile(r'\b(?:experimental setup|input prompt length|output length|input tokens|output tokens|batch size|wall.clock|workloads?)\b', re.I)
SETUP = re.compile(r'\b(?:experimental setup|we use|we evaluate|hardware|batch size|wall.clock|input tokens|output tokens|workloads?|data ?sets?|input prompt length|output length)\b', re.I)


def policy_identity(config):
    return {'policy': CONTEXT_POLICY, 'evidence_token_budget': config.get('evidence_token_budget', 6500),
            'adjacent_token_budget': config.get('adjacent_token_budget', 2800),
            'assessment_evidence_token_budget': config.get('assessment_evidence_token_budget', 6500)}


def _unique(chunks):
    seen = set()
    return [c for c in chunks if not (c['id'] in seen or seen.add(c['id']))]


def companion_candidates(corpus, seeds):
    """Immediate source neighbors, prioritizing experimental setup and captions.

    Neighbors follow original page/offset order, including page boundaries. Only
    neighbors of supplied seeds are considered; a bounded setup fallback considers at most three chunks per source.
    """
    by_source = {}
    for c in corpus.chunks:
        if c.get('page') is not None:
            by_source.setdefault(c['source_id'], []).append(c)
    positions = {}
    for group in by_source.values():
        group.sort(key=lambda c: (c['page'], c.get('char_start', c.get('token_start', 0))))
        positions.update({c['id']: (group, i) for i,c in enumerate(group)})
    candidates = {}
    for rank, seed in enumerate(_unique(seeds)):
        if seed['id'] not in positions:
            continue
        group, i = positions[seed['id']]
        for delta in (1, -1):
            j = i + delta
            if not 0 <= j < len(group):
                continue
            c = group[j]
            score = min(8, len(SETUP.findall(c['text'])))
            priority = (-score, rank, delta < 0)
            if c['id'] not in candidates or priority < candidates[c['id']][0]:
                candidates[c['id']] = (priority, c)
    ordered = [c for _, c in sorted(candidates.values(), key=lambda item: item[0])]
    captions = [c for c in ordered if CAPTION.search(c['text'])]

    def setup_key(c):
        terms = {m.lower() for m in STRONG_SETUP.findall(c['text'])}
        paired_lengths = bool(terms & {'input tokens', 'input prompt length'}) and bool(terms & {'output tokens', 'output length'})
        return (-int(paired_lengths), -len(terms), c['source_id'], c['page'], c.get('char_start', 0))

    # At most three same-source full chunks, ranked by distinct setting fields.
    # A paired input/output length is stronger than repeated generic mentions.
    fallback = []
    for source in sorted({c['source_id'] for c in seeds}):
        matches = [c for c in by_source.get(source, []) if len(set(STRONG_SETUP.findall(c['text']))) >= 2]
        fallback.extend(sorted(matches, key=setup_key)[:3])
    setups = sorted(_unique(fallback + [c for c in ordered if SETUP.search(c['text'])]), key=setup_key)
    # Reserve alternating opportunities for captions and setup paragraphs.
    # Original full chunks only; neighbor expansion is not recursive.
    paired = []
    for i in range(max(len(captions), len(setups))):
        if i < len(captions):
            paired.append(captions[i])
        if i < len(setups):
            paired.append(setups[i])
    return _unique(paired + ordered)


def _pack(corpus, required, companions, ranked, budget, companion_budget):
    if budget <= 0 or companion_budget < 0 or companion_budget > budget:
        raise ValueError('Invalid evidence/companion token budget')
    selected, seen, used = [], set(), 0
    def take(items, limit, mandatory=False):
        nonlocal used
        spent = 0
        for c in items:
            if c['id'] in seen:
                continue
            cost = corpus.evidence_tokens(c['text'])
            if used + cost > budget or spent + cost > limit:
                if mandatory:
                    raise ValueError('Required cited full chunks exceed evidence token budget')
                continue
            selected.append(c);seen.add(c['id']);used += cost;spent += cost
        return spent
    take(required, budget, True)
    take(companions, min(companion_budget, budget-used))
    take(ranked, budget-used)
    return selected


def build_research_context(corpus, ranked, technology, config):
    seeds = _unique([c for c in ranked if c.get('technology') == technology])
    budget = config.get('evidence_token_budget', 6500)
    reserve = config.get('adjacent_token_budget', 2800)
    # Start from the legacy 6500/2800 selection before bounded replacement.
    baseline = _pack(corpus, [], corpus.adjacent_candidates(seeds), seeds, budget, reserve)
    companions = companion_candidates(corpus, _unique(seeds + baseline))
    # Up to two cross-page-only choices can yield to a caption/setup pair.
    # Ranked hits keep their original order and full source text.
    ranked_ids = {c['id'] for c in seeds}
    additions = [c for c in companions if c['id'] not in {b['id'] for b in baseline}][:2]
    retained = list(baseline)
    needed = sum(corpus.evidence_tokens(c['text']) for c in additions)
    removed = 0
    for c in reversed(baseline):
        if sum(corpus.evidence_tokens(b['text']) for b in retained) + needed <= budget:
            break
        if c['id'] not in ranked_ids and removed < 2:
            retained.remove(c)
            removed += 1
    return _pack(corpus, retained, companions, [], budget, budget)


def assessment_evidence(corpus, tech_assessment, config=None):
    config = config or getattr(corpus, 'config', {})
    ids = {r['chunk_id'] for a in tech_assessment.values() for c in a['claims'] for r in c['references']}
    cited = [c for c in corpus.chunks if c['id'] in ids]
    if ids - {c['id'] for c in cited}:
        raise ValueError('A cited chunk is absent from the frozen corpus')
    return _pack(corpus, cited, companion_candidates(corpus, cited), [],
                 config.get('assessment_evidence_token_budget', 6500), config.get('adjacent_token_budget', 2800))
