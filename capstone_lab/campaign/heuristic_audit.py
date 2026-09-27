"""Read-only post-generation audit; reuse successfully verified image hashes."""
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import read_json, inside
from .io import atomic_json


def audit(root, output):
    run = root / 'artifacts/s8_heuristic_formal/run_v1'
    state = read_json(run / 'campaign_state.json')
    if state['status'] != 'SUCCEEDED':
        raise ValueError('Completed formal campaign required')
    output.mkdir(parents=True, exist_ok=False)
    summaries, samples, all_hashes = {}, [], defaultdict(list)
    for subset in ('train748', 'low187'):
        base = run / 'datasets' / subset
        report = read_json(base / 'report.json')
        plans = {p['sample_id']: p for p in read_json(base / 'plan.json')['plans']}
        source_file = root / ('manifests/low187_v1.jsonl' if subset == 'low187' else 'manifests/real_split_observed_v1.jsonl')
        allowed = {r['image']: r['image_sha256'] for r in map(json.loads, source_file.read_text(encoding='utf-8').splitlines()) if r['original_split'] == 'train'}
        conditions = ['A1'] if subset == 'low187' else ['A'+str(i) for i in range(6)]
        for condition in conditions:
            tag = subset + '_' + condition
            rows, pairs, hashes, usage = [], Counter(), defaultdict(list), Counter()
            distributions = {k: Counter() for k in ('domain','mode','difficulty','scale_bucket')}
            candidates = []
            for sid in sorted(plans):
                key = sid + '/' + condition
                directory = base / 'samples' / key
                h = report['output_hashes'][key]
                for name in ('image.png','mask.png','polygon.txt','metadata.json','COMMITTED.json'):
                    if not (directory / name).is_file():
                        raise ValueError('Missing output: '+str(directory / name))
                if not (directory.parent / 'PAIR.json').is_file():
                    raise ValueError('Missing paired commit')
                m = read_json(directory / 'metadata.json')
                if m['subset'] != subset or m['condition_id'] != condition or m['sample_id'] != sid:
                    raise ValueError('Metadata identity mismatch')
                fg = m['foreground']
                if allowed.get(fg['source_image']) != fg['source_sha256']:
                    raise ValueError('Source outside allowed train subset')
                if fg['foreground_id'] != m['foreground_id'] or m['roundtrip_iou'] < .9:
                    raise ValueError('Foreground identity or mask alignment mismatch')
                for field in distributions:
                    if m['plan'][field] != plans[sid][field]:
                        raise ValueError('Actual output differs from plan')
                    distributions[field][m['plan'][field]] += 1
                pairs[(m['foreground_id'], m['background_id'])] += 1
                usage[m['foreground_id']] += 1
                hashes[h['image']].append(sid)
                all_hashes[h['image']].append(tag+'/'+sid)
                row = {'image': (directory/'image.png').relative_to(root).as_posix(),
                       'label': (directory/'polygon.txt').relative_to(root).as_posix(),
                       'mask': (directory/'mask.png').relative_to(root).as_posix(),
                       'image_sha256': h['image'], 'label_sha256': h['polygon'],
                       'mask_sha256': h['mask'], 'original_split':'train',
                       'kind':'synthetic', 'subset':subset, 'condition':condition}
                rows.append(row)
                candidates.append((hashlib.sha256((tag+sid).encode()).hexdigest(), m, row))
            if len(rows) != 3000 or report['counts'][condition] != 3000:
                raise ValueError('Count mismatch')
            duplicates = [v for v in hashes.values() if len(v)>1]
            summaries[tag] = {'count':len(rows), 'distributions':distributions,
                'source_violations':0, 'object_count':len(usage), 'object_usage':dict(usage),
                'same_object_background_repeat_excess':sum(n-1 for n in pairs.values()),
                'exact_image_duplicate_groups':duplicates}
            atomic_json(output/(tag+'_manifest.json'), rows, immutable=True)
            # Deterministic greedy coverage of all observed categorical strata.
            covered = set()
            for _, m, row in sorted(candidates):
                strata = {(k,m['plan'][k]) for k in distributions}
                if strata <= covered:
                    continue
                covered |= strata
                samples.append(dict(row, sample_id=m['sample_id'], tag=tag,
                                    plan={k:m['plan'][k] for k in distributions}))
        summaries[subset+'_candidate_rejections'] = dict(Counter(f.get('rejected','UNKNOWN') for f in report['failures']))
    result = {'status':'AUDIT_RECORDED', 'hash_basis':'existing SUCCEEDED campaign; full image rehash not repeated',
              'datasets':summaries, 'cross_condition_exact_duplicate_groups':
              [v for v in all_hashes.values() if len(v)>1],
              'visual_review':'PENDING', 'final_generation_failures':0}
    atomic_json(output/'summary.json', result, immutable=True)
    atomic_json(output/'samples.json', samples, immutable=True)
    print(json.dumps({'status':result['status'],'samples':len(samples),
        'datasets':{k:{f:v[f] for f in ('count','object_count','same_object_background_repeat_excess','exact_image_duplicate_groups')} for k,v in summaries.items() if 'count' in v}}))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path.cwd()
    audit(root, inside(root,args.output))
