"""Frozen task-aware ranking; independent from image/mask validity checks."""
import hashlib
import math
from collections import Counter


def task_score(mask_iou, confidence):
    values = (float(mask_iou), float(confidence))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid IoU/confidence')
    return 2.0 - sum(values)


def tie_key(seed, identifier):
    return hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest()


def select(rows, count=3000, seed=416):
    if len(rows) < count or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Insufficient/duplicate candidate IDs')
    for row in rows:
        if row.get('valid') is not True:
            raise ValueError('Only the common audited valid pool may enter selection')
        if abs(task_score(row['mask_iou'], row['confidence']) - row['score']) > 1e-9:
            raise ValueError('Score does not match frozen formula')
    random = sorted(rows, key=lambda r: (tie_key(seed, r['id']), r['id']))[:count]
    task = sorted(rows, key=lambda r: (-r['score'], tie_key(seed, r['id']), r['id']))[:count]
    def stats(selected):
        return {'domains': dict(Counter(r['domain'] for r in selected)),
                'scales': dict(Counter(r['scale_bucket'] for r in selected)),
                'no_predictions': sum(r['prediction_count'] == 0 for r in selected),
                'objects': dict(Counter(r['reference'] for r in selected))}
    return {'random': [r['id'] for r in random], 'task': [r['id'] for r in task],
        'overlap': len({r['id'] for r in random} & {r['id'] for r in task}),
        'random_statistics': stats(random), 'task_statistics': stats(task),
        'note': 'Distribution differences are reported, not silently rebalanced. '
                'No-prediction cases score2; this can enrich difficult cases and is not proof of semantic validity.'}
