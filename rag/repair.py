"""Bounded, typed corrections merged into an immutable full assessment.

A repair round may contain several independent units. Each unit is attempted
once; no unit can change unrequested fields or omit another claim. Source text
is never shortened. All assembled output is checked by the original validator.
"""
from copy import deepcopy
from dataclasses import dataclass
import re
from pydantic import Field
from rag.schemas import Strict, Claim, Section
from rag.corpus import normalized
from rag.request_budget import deduplicate_chunks

REPAIR_INSTRUCTIONS = (
    '주어진 오류 항목만 수정하고 지정된 patch schema로 답하라. '
    '요청한 문장·필드 외에는 변경하지 말고 사실·실험조건·한계를 보존하라. '
    '인용은 제공된 전체 원문에서 그대로 복사하며 하이픈·띄어쓰기를 추정해서 바꾸지 마라. '
    '지정된 target_id를 정확히 한 번씩 반환하고 새 ID·근거·주장을 추가하지 마라.'
)


class CitationPatch(Strict):
    target_id: str
    quote: str


class CitationPatches(Strict):
    patches: list[CitationPatch] = Field(min_length=1)


class ReferencePatch(Strict):
    target_id: str
    chunk_id: str
    quote: str


class ReferencePatches(Strict):
    patches: list[ReferencePatch] = Field(min_length=1, max_length=1)


class ClaimPatch(Strict):
    target_id: str
    claim: Claim


class ClaimPatches(Strict):
    patches: list[ClaimPatch] = Field(min_length=1)


class LayoutPatch(Strict):
    summary_claim_ids: list[str]
    sections: list[Section]


@dataclass
class RepairUnit:
    kind: str
    target_id: str
    schema: type
    content: dict
    collection: str = ''
    index: int = -1
    reference_index: int = -1
    allowed_fields: tuple = ()


def restore_sufficient(value, content):
    """Restore original sufficient claims in code, without changing caller state.

    Malformed/missing/duplicate identity sets are NOT silently repaired; their
    original structural validator must reject them. Existing valid identities
    are put in original order so preservation checking is deterministic.
    """
    previous = content.get('previous_assessment') if isinstance(content, dict) else None
    if not isinstance(previous, dict) or not isinstance(value.get('claims'), list):
        return value
    originals = previous.get('claims', [])
    old_pairs = [(c['technology'], c['facet']) for c in originals]
    new_pairs = [(c['technology'], c['facet']) for c in value['claims']]
    if len(set(old_pairs)) != len(old_pairs) or len(set(new_pairs)) != len(new_pairs) or set(old_pairs) != set(new_pairs):
        return value
    missing = content.get('missing_facets', [])
    current = dict(zip(new_pairs, value['claims']))
    result = deepcopy(value)
    result['claims'] = [deepcopy(old if old['facet'] not in missing else current[pair])
                        for pair, old in zip(old_pairs, originals)]
    return result


def plan_repairs(value, errors, content, source_chunks):
    """Return precise units, or None for schemas/errors requiring whole repair."""
    collection = 'claims' if 'claims' in value else 'synthesis_claims' if 'synthesis_claims' in value else None
    if not collection or not errors or not all(isinstance(e, str) for e in errors):
        return None
    claim_errors = {}
    layout_errors = []
    for error in errors:
        match = re.fullmatch(r'claim (\d+): (.+)', error)
        if match:
            index = int(match[1])
            if index >= len(value[collection]):
                return None
            claim_errors.setdefault(index, []).append(match[2])
        else:
            layout_errors.append(error)
    if layout_errors and collection != 'synthesis_claims':
        return None
    # Report layout errors can be repaired without resending full claim prose.
    # Unrecognized non-layout errors still go through the bounded generic path.
    layout_markers = ('SUMMARY', 'Report must', 'report sections', 'body section', 'Every claim',
                      'wrong perspective section', 'Unknown report claim', 'both technologies',
                      'retain all four', 'Final body', 'claim must not repeat', 'Empty report', 'section order')
    if any(not any(marker.lower() in e.lower() for marker in layout_markers) for e in layout_errors):
        return None
    chunks = deduplicate_chunks({'chunks': source_chunks})['chunks']
    lookup = {c['id']: c for c in chunks}
    units = []
    for index, failures in claim_errors.items():
        claim = value[collection][index]
        if all(f.startswith('quotation does not exist in supplied chunk ') for f in failures):
            bad = [(j, ref) for j, ref in enumerate(claim['references'])
                   if ref['chunk_id'] not in lookup or len(normalized(ref['quote'])) < 12
                   or normalized(ref['quote']) not in normalized(lookup[ref['chunk_id']]['text'])]
            if not bad:
                return None
            for j, ref in bad:
                target = f'{collection}:{index}:reference:{j}'
                if ref['chunk_id'] not in lookup:
                    relevant = [deepcopy(c) for c in chunks
                                if c.get('technology') in (None, claim['technology'], 'both')]
                    units.append(RepairUnit('reference', target, ReferencePatches,
                        {'target_id': target, 'claim': deepcopy(claim), 'reference': deepcopy(ref),
                         'errors': failures, 'chunks': relevant,
                         'correction': 'Replace only this invalid reference with a supplied chunk ID and its exact quotation.'},
                        collection, index, j))
                    continue
                units.append(RepairUnit('citation', target, CitationPatches,
                    {'target_id': target, 'claim': deepcopy(claim), 'reference': deepcopy(ref),
                     'errors': failures, 'chunks': [deepcopy(lookup[ref['chunk_id']])]}, collection, index, j))
        else:
            fields = set()
            for failure in failures:
                if failure == 'numerical result requires experimental conditions':
                    fields.add('conditions')
                elif failure == 'inference/scenario/unknown must disclose limitations':
                    fields.add('caveats')
                elif failure == 'missing source evidence':
                    fields.add('references')
                else:
                    return None
            target = f'{collection}:{index}'
            relevant = [deepcopy(c) for c in chunks if c.get('technology') in (None, claim['technology'], 'both')]
            units.append(RepairUnit('claim', target, ClaimPatches,
                {'target_id': target, 'claim': deepcopy(claim), 'allowed_fields': sorted(fields),
                 'errors': failures, 'chunks': relevant}, collection, index, allowed_fields=tuple(sorted(fields))))
    if layout_errors:
        metadata = [{key: c[key] for key in ('id', 'technology', 'facet', 'perspective') if key in c}
                    for c in content.get('claims', [])]
        metadata += [{'id': f'synthesis-{i+1}', 'technology': c['technology'], 'facet': c['facet'], 'perspective': 'synthesis'}
                     for i, c in enumerate(value['synthesis_claims'])]
        units.append(RepairUnit('layout', 'report-layout', LayoutPatch,
            {'current_layout': {k: deepcopy(value[k]) for k in ('summary_claim_ids', 'sections')},
             'claim_metadata': metadata, 'errors': layout_errors,
             'required_order': ['기술 성숙도', '시장성', '이해관계자', '도메인 적용', '관점 간 상충과 한계'],
             'summary_count': '2..3; use all existing claims exactly once in their perspective body'}))
    return units or None


def source_verbatim_quote(quote, source):
    """Restore only PDF word-wrap hyphens through one unique source span.

    The saved quotation is the original source substring, never a rewritten
    source. Other wording, numbers and punctuation cannot be repaired here.
    """
    if normalized(quote) in normalized(source):
        return quote
    text = normalized(quote)
    pattern = []
    for i, char in enumerate(text):
        if i and text[i - 1].isalpha() and char.isalpha():
            pattern.append(r'(?:-\s+)?')
        if char == '-' and i and i + 1 < len(text) and text[i - 1].isalpha() and text[i + 1].isalpha():
            pattern.append(r'-\s*')
        else:
            pattern.append(r'\s+' if char.isspace() else re.escape(char))
    matches = list(re.finditer(''.join(pattern), source)) if text else []
    return matches[0].group() if len(matches) == 1 else quote


def restore_verbatim_references(value, source_chunks):
    """Restore uniquely aligned typography only; preserve IDs and all claims."""
    result = deepcopy(value)
    lookup = {chunk['id']: chunk['text'] for chunk in source_chunks}
    for collection in ('claims', 'synthesis_claims'):
        for claim in result.get(collection, []):
            for ref in claim.get('references', []):
                if ref['chunk_id'] in lookup:
                    ref['quote'] = source_verbatim_quote(ref['quote'], lookup[ref['chunk_id']])
    return result


def apply_patch(value, unit, patch):
    """Copy-and-merge; reject absent/duplicate/unknown IDs or non-target changes."""
    patch = unit.schema.model_validate(patch).model_dump()
    result = deepcopy(value)
    if unit.kind == 'layout':
        result.update(patch)
        return result
    patches = patch['patches']
    if len(patches) != 1 or patches[0]['target_id'] != unit.target_id:
        raise ValueError('Patch IDs must match the single requested target exactly once')
    item = patches[0]
    if unit.kind == 'reference':
        claim = result[unit.collection][unit.index]
        chunk = next((c for c in unit.content['chunks'] if c['id'] == item['chunk_id']), None)
        if chunk is None or chunk.get('technology') not in (None, claim['technology'], 'both'):
            raise ValueError('Reference patch requires a supplied same-technology source chunk')
        verbatim = source_verbatim_quote(item['quote'], chunk['text'])
        quote = normalized(verbatim)
        if len(quote) < 12 or quote not in normalized(chunk['text']):
            raise ValueError('Reference patch quotation must exist in the supplied full source chunk')
        claim['references'][unit.reference_index] = {'chunk_id': item['chunk_id'], 'quote': verbatim}
    elif unit.kind == 'citation':
        original_ref = result[unit.collection][unit.index]['references'][unit.reference_index]
        original_ref['quote'] = item['quote']
    else:
        old, new = result[unit.collection][unit.index], item['claim']
        if any(new[k] != old[k] for k in old if k not in unit.allowed_fields):
            raise ValueError('Patch attempted to modify immutable/unrequested claim fields')
        if 'references' in unit.allowed_fields and any(ref not in new['references'] for ref in old['references']):
            raise ValueError('Patch may not drop existing source references')
        result[unit.collection][unit.index] = new
    return result
