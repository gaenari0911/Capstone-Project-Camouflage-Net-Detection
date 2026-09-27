"""Read-only Real-only audit/Judge freeze and fail-closed synthesis capacity gate."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import math
from pathlib import Path
import sqlite3
import statistics

from capstone_lab.config import sha256_file
from .contracts import contract, digest, inside, read_json
from .io import atomic_json
from .seed_policy import load_queue_policy
from .supervisor import verify_result

RUN = 'artifacts/s8_real_campaign/run_v1'
OUT = 'artifacts/s8_post_real/run_v1'
METRICS = ('mask_precision', 'mask_recall', 'mask_f1', 'mask_map50', 'mask_map50_95')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def aggregate(records):
    result = {}
    for experiment in ('M0', 'L0'):
        selected = [r for r in records if r['experiment'] == experiment]
        require(sorted(r['seed'] for r in selected) == [0, 1, 2], 'missing/duplicate seeds')
        result[experiment] = {key: {'mean': statistics.mean(r[key] for r in selected),
                                   'sample_std_ddof1': statistics.stdev(r[key] for r in selected),
                                   'n': 3} for key in METRICS}
    return result


def legacy_capacity(source, counts, target=3000):
    tree = ast.parse(source)
    defaults = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'MAX_OBJECT_USAGE' for t in n.targets)]
    initial = ast.literal_eval(defaults[-1].value)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_build_object_quota')
    ceilings = [n.comparators[0].value for n in ast.walk(function)
                if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name)
                and n.left.id == 'max_usage' and len(n.ops) == 1 and isinstance(n.ops[0], ast.Lt)
                and isinstance(n.comparators[0], ast.Constant)]
    require(len(ceilings) == 1 and type(ceilings[0]) is int, 'unknown quota implementation')
    ceiling = max(initial, ceilings[0])
    capacity = sum(counts.values()) * ceiling
    return {'status': 'BLOCKED_POLICY_DECISION' if capacity < target else 'CAPACITY_ONLY_PASS',
            'target_per_condition': target, 'foreground_counts': dict(counts),
            'legacy_default_limit': initial, 'legacy_relaxation_ceiling': ceiling,
            'capacity_upper_bound_per_ranked_condition': capacity,
            'shortfall_at_least': max(0, target-capacity),
            'applies_to': ['A1', 'A2', 'A3', 'A4', 'A5'],
            'A0': 'random control has no quota ranking; does not rescue paired capacity',
            'domain_constraints_can_only_reduce_capacity': True,
            'minimum_average_reuse': target / sum(counts.values()),
            'minimum_global_ceiling_necessary_not_sufficient': math.ceil(target / sum(counts.values())),
            'formal_launch_permitted': False}


def audit(root):
    run, output = root / RUN, root / OUT
    config, binding = contract(root, root / 'configs/campaigns/s8_real_only_v1.json', run / 'runtime')
    require(read_json(run / 'contract.json') == binding, 'frozen campaign binding mismatch')
    require(read_json(run / 'config.json') == config, 'frozen config mismatch')
    policy = load_queue_policy(root, run, config)
    require(policy is not None, 'missing approved cancellation policy')
    with sqlite3.connect((run / 'state.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        import json
        jobs = {name: json.loads(raw) for name, raw in db.execute('SELECT id,record FROM jobs')}
    expected = {'M0_seed0', 'M0_seed1', 'M0_seed2', 'L0_seed0', 'L0_seed1', 'L0_seed2', 'H0_R_seed0', 'H1_R_seed0'}
    cancelled = {'H0_R_seed1', 'H0_R_seed2', 'H1_R_seed1', 'H1_R_seed2'}
    require(set(jobs) == expected | cancelled, 'unexpected job set')
    records = []
    for job in config['jobs']:
        name, record = job['id'], jobs[job['id']]
        if name in cancelled:
            require(record['state'] == 'CANCELLED_BY_USER' and record['attempts'] == 0,
                    'cancelled job unexpectedly executed')
            continue
        require(record['state'] == 'SUCCEEDED', 'incomplete Real-only job')
        marker_hash = verify_result(run, job, digest(binding), record)
        directory = run / 'jobs' / name / 'training_state'
        pointer, result = read_json(directory / 'current.json'), read_json(directory / 'result.json')
        training_contract = read_json(directory / 'contract.json')
        require(pointer['contract'] == training_contract, 'training contract mismatch')
        require(result['contract_sha256'] == digest(training_contract), 'training binding mismatch')
        require(result['epochs'] == pointer['epoch'] == 150 and result['formal_training_completed'], 'wrong epochs')
        require(result['best_mask'] == pointer['best_mask'], 'best pointer mismatch')
        history = result['history']
        require([h['epoch'] for h in history] == list(range(1, 151)), 'history incomplete')
        require(all(math.isfinite(h['mask_map50_95']) for h in history), 'nonfinite score')
        chosen = max(history, key=lambda h: h['mask_map50_95'])
        best = pointer['best_mask']
        require(chosen['epoch'] == best['epoch'] and chosen['mask_map50_95'] == best['score'], 'best selection mismatch')
        require(sha256_file(inside(directory, best['file'])) == best['sha256'], 'best checkpoint changed')
        require(sha256_file(inside(directory, pointer['last']['file'])) == pointer['last']['sha256'], 'last changed')
        metrics = chosen['metrics']
        require(metrics['images'] == 100 and metrics['inference_boundary_calls'] == 0, 'validation contract mismatch')
        row = {'job_id': name, 'experiment': result['experiment'], 'seed': result['seed'],
               'epochs': 150, 'best_epoch': best['epoch'], 'attempts': record['attempts'],
               'checkpoint': (directory / best['file']).relative_to(root).as_posix(),
               'checkpoint_sha256': best['sha256'], 'completion_marker_sha256': marker_hash,
               'training_contract_sha256': digest(training_contract),
               'mask_precision': metrics['mask_precision'], 'mask_recall': metrics['mask_recall'],
               'mask_f1': metrics['mask_f1'], 'mask_map50': metrics['library_metrics']['metrics/mAP50(M)'],
               'mask_map50_95': best['score'], 'optimizer_steps': history[-1]['global_step'],
               'epoch_wall_seconds_sum': sum(h['wall_seconds'] for h in history)}
        records.append(row)
    records.sort(key=lambda r: r['job_id'])
    report = {'status': 'VERIFIED', 'split': 'val100', 'test_inference_executed': False,
              'checkpoint_verification': 'completion-bound SHA256 of bytes; no tensor deserialization or new inference',
              'campaign_contract_sha256': digest(binding), 'completed_count': 8,
              'cancelled_never_started': sorted(cancelled), 'records': records,
              'aggregate': aggregate(records)}
    atomic_json(output / 'real_only_audit.json', report, immutable=True)
    judge = next(r for r in records if r['job_id'] == 'L0_seed0')
    directory = run / 'jobs/L0_seed0/training_state'
    training = read_json(directory / 'contract.json')
    manifest = {'status': 'FROZEN', 'selection_rule': 'preapproved L0_seed0 best_mask; no cross-seed reselection',
                'checkpoint': judge['checkpoint'], 'checkpoint_sha256': judge['checkpoint_sha256'],
                'checkpoint_format': 'custom epoch checkpoint; loader implementation must use saved model state, not assume standalone YOLO weights',
                'best_epoch': judge['best_epoch'], 'model': 'custom DualHeadSegment n / nc=1 / H2 distance BCE',
                'training_contract': (directory / 'contract.json').relative_to(root).as_posix(),
                'training_contract_file_sha256': sha256_file(directory / 'contract.json'),
                'training_contract_payload': training,
                'information_budget': {'training': 'low187', 'training_images': 187, 'validation_images': 100,
                                       'low187_sha256': sha256_file(root / 'manifests/low187_v1.jsonl')},
                'preprocessing': {'implementation': RUN + '/runtime/capstone_lab/campaign/data.py',
                                  'sha256': sha256_file(run / 'runtime/capstone_lab/campaign/data.py'),
                                  'imgsz': 640, 'augment': False, 'rect': False, 'stride': 32,
                                  'tensor_scale': 'float()/255', 'effective_args': training['effective_args']},
                'scoring_implementation_status': 'NOT_IMPLEMENTED_OR_EXECUTED',
                'candidate_inference_executed': False, 'learner_initialization_permitted': False,
                'audit_sha256': sha256_file(output / 'real_only_audit.json')}
    atomic_json(output / 'judge_manifest.json', manifest, immutable=True)
    provenance_path = root / 'artifacts/s8_source_audit/run_v1/low187_object_provenance.json'
    provenance = read_json(provenance_path)
    require(provenance['status'] == 'VERIFIED' and not provenance['rejected'], 'source audit not verified')
    low_path = root / 'manifests/low187_v1.jsonl'
    require(provenance['low187_sha256'] == sha256_file(low_path), 'low187 changed')
    import json
    low = [json.loads(line) for line in low_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(low) == 187 and len(provenance['records']) == 187, 'not exactly187 sources')
    require({r['source_image'] for r in provenance['records']} == {r['image'] for r in low}, 'source budget mismatch')
    for row in provenance['records']:
        for key, hashkey in [('object_image', 'object_image_sha256'), ('object_mask', 'object_mask_sha256'),
                             ('source_image', 'source_sha256'), ('label', 'label_sha256')]:
            require(sha256_file(inside(root, row[key])) == row[hashkey], 'source changed: ' + row[key])
    legacy = root / 'build_demo_synthetic_segmentation.py'
    capacity = legacy_capacity(legacy.read_text(encoding='utf-8'), Counter(r['environment'] for r in provenance['records']))
    capacity.update(legacy_source_sha256=sha256_file(legacy), source_proof_sha256=sha256_file(provenance_path),
                    formal_generated_images=0, worker_count=0, eta_seconds=None,
                    recommended_decision='Keep shared187 and3000; approve a deterministic domain-aware quota policy scaled to targets, common to A1-A5; preserve A0 no-score random control. Not applied.')
    atomic_json(output / 'synthesis_capacity_gate.json', capacity, immutable=True)
    return {'audit': report['status'], 'judge': manifest['status'], 'synthesis': capacity,
            'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    import json
    print(json.dumps(audit(args.root.resolve()), indent=2))
