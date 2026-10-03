import re, json

CATEGORIES = 'harassment harassment_threatening hate hate_threatening self_harm self_harm_instructions self_harm_intent sexual sexual_minors violence violence_graphic'.split()

def normalize(scores):
    result = {}
    for name, value in scores.items():
        key = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        if key in result and result[key] != value:
            raise ValueError(f'Conflicting category aliases: {name}')
        result[key] = value
    if set(result) != set(CATEGORIES):
        raise ValueError(f'Unexpected category set: {sorted(result)}')
    return {name: result[name] for name in CATEGORIES}

def canonical(scores):
    return json.dumps(normalize(scores), sort_keys=True, separators=(',', ':'), allow_nan=False)
