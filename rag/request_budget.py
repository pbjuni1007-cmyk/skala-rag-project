"""Request accounting shared by generation and bounded repair planning."""
from copy import deepcopy
import json

PROTOCOL_MARGIN = 512
STRUCTURED_REPAIR_HEADROOM = 512


class InputBudgetExceeded(ValueError):
    def __init__(self, purpose, count, limit, reserve=0, reason='input_limit'):
        self.diagnostic = {'type': 'InputBudgetExceeded', 'purpose': purpose,
                           'input_tokens': count, 'input_limit': limit,
                           'protocol_margin': PROTOCOL_MARGIN, 'repair_reserve': reserve,
                           'available_input_tokens': max(0, limit - PROTOCOL_MARGIN - reserve),
                           'reason': reason, 'generation_sent': False}
        super().__init__(json.dumps(self.diagnostic, ensure_ascii=False))


def enforce_input_budget(purpose, count, limit, reserve=0):
    if type(count) is not int or count < 0:
        raise ValueError('Provider returned an invalid exact input token count')
    if type(reserve) is not int or reserve < 0:
        raise ValueError('Invalid repair input reserve')
    if count > limit - PROTOCOL_MARGIN - reserve:
        raise InputBudgetExceeded(purpose, count, limit, reserve)
    return count


def deduplicate_chunks(content):
    """Remove only byte-equivalent repeated chunks, without changing caller state.

    Same ID with differing text/metadata is a conflict, never silently merged.
    Distinct IDs and overlapping source spans remain intact (references depend
    on those IDs and metadata). All first-occurrence full source text survives.
    """
    if not isinstance(content, dict) or not isinstance(content.get('chunks'), list):
        return content
    seen, unique = {}, []
    for chunk in content['chunks']:
        key = chunk['id']
        if key in seen:
            if seen[key] != chunk:
                raise ValueError(f'Conflicting duplicate source chunk: {key}')
            continue
        seen[key] = chunk
        unique.append(chunk)
    if len(unique) == len(content['chunks']):
        return content
    result = deepcopy(content)
    result['chunks'] = deepcopy(unique)
    return result
