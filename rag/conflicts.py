"""Preserve conflict provenance without guessing semantic links or resolution."""
import re
from pydantic import ValidationError
from rag.schemas import ConflictRecord


def build_conflict_records(assessments, claims):
    records = []
    for perspective, assessment in assessments.items():
        owned = [cid for cid, claim in claims.items() if claim.get('perspective') == perspective]
        for index, text in enumerate(assessment.get('conflicts', []), 1):
            explicit = [cid for cid in claims if re.search(
                r'(?<![\w-])' + re.escape(cid) + r'(?![\w-])', text)]
            candidates = list(dict.fromkeys(owned + explicit))
            records.append(ConflictRecord(
                id=f'conflict-{perspective}-{index}', perspective=perspective, text=text,
                candidate_claim_ids=candidates, explicit_claim_ids=explicit, verified_claim_ids=[],
                conditions={cid: claims[cid].get('conditions', '') for cid in candidates},
                evidence_ids={cid: list(claims[cid].get('evidence_ids', [])) for cid in candidates},
                type='unclassified', status='unresolved', reviewed_by='',
                rationale='소유 관점과 원문에 명시된 주장 ID로 후보를 연결했습니다. 의미 관계·유형·해소 여부는 미검수입니다.',
            ).model_dump())
    return records


def conflict_record_errors(records, claims, assessments):
    """Validate identities and review prerequisites, not semantic truth."""
    errors = []
    expected = {r['id']: r for r in build_conflict_records(assessments, claims)}
    parsed = []
    for index, value in enumerate(records):
        try:
            parsed.append(ConflictRecord.model_validate(value).model_dump())
        except ValidationError:
            errors.append(f'conflict record {index}: invalid structure')
    ids = [r['id'] for r in parsed]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        errors.append('Every original conflict must have exactly one record; duplicate or unknown IDs are forbidden')
    for record in parsed:
        cid = record['id']
        original = expected.get(cid)
        if original is None:
            continue
        for field in ('perspective', 'text', 'candidate_claim_ids', 'explicit_claim_ids'):
            if record[field] != original[field]:
                errors.append(f'{cid}: original {field} must be preserved')
        for field in ('candidate_claim_ids', 'explicit_claim_ids', 'verified_claim_ids'):
            values = record[field]
            if len(values) != len(set(values)) or any(v not in claims for v in values):
                errors.append(f'{cid}: invalid or duplicate {field}')
        linked = set(record['candidate_claim_ids'] + record['verified_claim_ids'])
        for field, default in (('conditions', ''), ('evidence_ids', [])):
            snapshot = {key: claims[key].get(field, default) for key in linked if key in claims}
            if record[field] != snapshot:
                errors.append(f'{cid}: {field} must preserve the linked claims exactly')
        if not record['rationale'].strip():
            errors.append(f'{cid}: rationale is required')
        reviewed = record['reviewed_by'].strip() and record['rationale'].strip()
        if (record['verified_claim_ids'] or record['type'] != 'unclassified') and not reviewed:
            errors.append(f'{cid}: verified links and classification require a named human review')
        if record['status'] == 'resolved':
            supported = [claims[key] for key in record['verified_claim_ids'] if key in claims]
            if (not reviewed or record['type'] == 'unclassified' or len(supported) < 2
                    or any(c.get('kind') == 'unknown' or not c.get('evidence_ids') for c in supported)):
                errors.append(f'{cid}: resolved requires reviewed classification and two verified sourced claims beyond unknown')
    return errors


def conflicts_for_joined(joined):
    """Old saved runs remain renderable without silently inventing provenance."""
    assessments = joined.get('assessments')
    if not assessments:
        assessments = {'legacy': {'conflicts': joined.get('conflicts', [])}}
    records = joined.get('conflict_records')
    if records is None:
        records = build_conflict_records(assessments, joined['claims'])
    errors = conflict_record_errors(records, joined['claims'], assessments)
    if errors:
        raise ValueError('Conflict review validation failed: ' + '; '.join(errors))
    return records


def conflict_review_markdown(records, claims):
    lines = ['# 상충 검수', '',
             '후보 연결은 같은 관점에서 나온 주장 또는 원문에 명시된 ID입니다. 의미 관계가 검증됐다는 뜻은 아닙니다.', '',
             '유형·검증 연결·해소 판단은 사람이 원문과 조건을 대조해 기록합니다. 구조 검사는 의미적 진실을 인증하지 않습니다.', '']
    if not records:
        lines += ['등록된 상충은 없습니다. 상충이 실제로 없다는 판정은 아닙니다.', '']
    for record in records:
        lines += [f"## {record['id']}", '', f"**소유 관점:** {record['perspective']}", '',
                  '**상충 원문:** ' + record['text'], '',
                  f"**유형:** {record['type']} · **해소 상태:** {record['status']}", '',
                  '**판단 근거:** ' + record['rationale'], '',
                  '**검수자:** ' + (record['reviewed_by'] or '미검수'), '',
                  '**원문 명시 ID:** ' + (', '.join(record['explicit_claim_ids']) or '없음'), '',
                  '**검증된 관계:** ' + (', '.join(record['verified_claim_ids']) or '없음'), '']
        for cid in dict.fromkeys(record['candidate_claim_ids'] + record['verified_claim_ids']):
            claim = claims[cid]
            label = '검증 연결' if cid in record['verified_claim_ids'] else '후보·미검증'
            lines += [f"### [{cid}](citation_review.md#{cid}) · {label}", '',
                      claim['text'], '', '**조건:** ' + record['conditions'][cid], '',
                      '**한계:** ' + claim.get('caveats', ''), '',
                      '**근거 ID:** ' + (', '.join(record['evidence_ids'][cid]) or '없음'), '']
    return '\n'.join(lines)
