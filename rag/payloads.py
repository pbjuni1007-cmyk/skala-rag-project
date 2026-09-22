"""Lossless reference tables for cross-perspective synthesis inputs."""
from copy import deepcopy
import json


def synthesis_payload(claims, conflicts):
    table, identifiers, packed = {}, {}, []
    for original in claims:
        claim = deepcopy(original)
        ids = []
        for ref in claim.pop('references'):
            key = json.dumps(ref, ensure_ascii=False, sort_keys=True)
            if key not in identifiers:
                identifier = f'e{len(table) + 1}'
                identifiers[key] = identifier
                table[identifier] = deepcopy(ref)
            ids.append(identifiers[key])
        claim['reference_ids'] = ids
        packed.append(claim)
    return {'claims': packed, 'reference_table': table, 'conflicts': deepcopy(conflicts)}
