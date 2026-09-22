from rag.tracing import submit
"""Bounded facet reassessment when the complete request exceeds input budget."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag.budget import write_json
from rag.evidence import PERSPECTIVE_FACETS, validate_assessment
from rag.request_budget import InputBudgetExceeded, STRUCTURED_REPAIR_HEADROOM, deduplicate_chunks
from rag.schemas import Assessment


def reassess_facets(pipeline, purpose, instructions, content, check, base_instructions):
    content = deduplicate_chunks(content)
    schema = Assessment.model_json_schema()
    try:
        pipeline._preflight(purpose, base_instructions + instructions, content, schema,
                            STRUCTURED_REPAIR_HEADROOM)
    except InputBudgetExceeded as exc:
        # Only initial input overflow is splittable. Never repeat a generation or
        # a failed correction, and never split a protocol/part-count violation.
        if exc.diagnostic['reason'] != 'input_limit':
            raise
    else:
        return pipeline.structured(purpose, Assessment, instructions, content, check)

    original = deepcopy(content['previous_assessment'])
    missing = content['missing_facets']
    name = purpose.removesuffix('_reassessment')
    if not 1 <= len(missing) <= 3 or len(set(missing)) != len(missing) or not set(missing) <= PERSPECTIVE_FACETS[name]:
        raise ValueError('Facet reassessment requires 1..3 distinct known missing facets')
    completed, planned = {}, []
    evidence_path = pipeline.out / 'reassessments' / (purpose + '-split.json')

    def save(status, error=None):
        write_json(evidence_path, {'status': status, 'original_assessment': original,
                   'missing_facets': missing, 'completed_facets': completed,
                   'planned_facets': [facet for facet, *_ in planned], 'error': error})

    try:
        for facet in missing:
            previous = [deepcopy(c) for c in original['claims'] if c['facet'] == facet]
            expected = {('KIVI', facet), ('InfiniGen', facet)}
            if len(previous) != 2 or {(c['technology'], c['facet']) for c in previous} != expected:
                raise ValueError('Original facet must contain exactly one claim per technology')
            unit = {k: deepcopy(v) for k, v in content.items()
                    if k not in {'previous_assessment', 'missing_facets'}}
            # Old aggregate gaps/conflicts are retained locally for the final
            # merge. Relevant reasons remain in these two complete prior claims.
            unit.update(previous_assessment={'status': original['status'], 'claims': previous,
                                             'conflicts': [], 'gaps': []}, missing_facets=[facet])
            unit_instructions = instructions + (
                f' 이번 분할 요청의 범위는 {facet} 하나다. 전체 여섯 주장 작성 지시 대신 '
                'KIVI와 InfiniGen의 이 facet 주장만 각각 하나씩, 총 두 개를 반환하라. '
                '다른 facet의 주장은 코드가 보존한다. 원문·조건을 유지하고 이 facet의 상충과 공백만 작성하라.')
            unit_purpose = f'{purpose}_facet_{facet}'
            pipeline._preflight(unit_purpose, base_instructions + unit_instructions, unit, schema,
                                STRUCTURED_REPAIR_HEADROOM)
            planned.append((facet, unit_purpose, unit_instructions, unit))
        save('preflight_passed')
        # Preflight every complete unit before allowing any generation. Only
        # this coordinator writes progress; workers never mutate shared state.
        def run_unit(facet, unit_purpose, unit_instructions, unit):
            def validate(value):
                errors = validate_assessment(value, content['chunks'])
                pairs = [(c['technology'], c['facet']) for c in value['claims']]
                if len(pairs) != 2 or set(pairs) != {('KIVI', facet), ('InfiniGen', facet)}:
                    errors.append('Facet reassessment requires exactly two claims with the requested technology/facet identities')
                return errors
            return pipeline.structured(unit_purpose, Assessment, unit_instructions, unit, validate)

        failures = {}
        with ThreadPoolExecutor(max_workers=min(len(planned), pipeline.settings.integer('RAG_MAX_CONCURRENCY', 3))) as pool:
            futures = {submit(pool, run_unit, *unit): unit[0] for unit in planned}
            for future in as_completed(futures):
                facet = futures[future]
                try:
                    completed[facet] = future.result()
                except Exception as exc:
                    failures[facet] = exc
                completed = {f: completed[f] for f in missing if f in completed}
                save('partial')
        # Drain all workers before failing so successful sibling evidence survives.
        if failures:
            raise next(failures[f] for f in missing if f in failures)
        replacements = {(c['technology'], c['facet']): c for result in completed.values() for c in result['claims']}
        merged = deepcopy(original)
        merged['claims'] = [deepcopy(replacements.get((c['technology'], c['facet']), c)) for c in original['claims']]
        for field in ('conflicts', 'gaps'):
            merged[field] = list(dict.fromkeys(original[field] + [v for r in completed.values() for v in r[field]]))
        merged['status'] = 'insufficient' if (any(r['status'] == 'insufficient' for r in completed.values()) or
                                              any(c['kind'] == 'unknown' for c in merged['claims'])) else 'ok'
        merged = Assessment.model_validate(merged).model_dump()
        errors = check(merged)
        if errors:
            raise ValueError('Merged facet reassessment failed full validation: ' + '; '.join(errors))
        save('validated')
        return merged
    except Exception as exc:
        save('failed', str(exc))
        raise
