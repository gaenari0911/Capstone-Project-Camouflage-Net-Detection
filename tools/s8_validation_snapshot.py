"""Read-only aggregation of completed training-time Validation records. No inference."""
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capstone_lab.campaign.io import atomic_json
from anydoor_runtime import sha256


def main():
    rows = []
    for campaign in ('s8_real_campaign', 's8_heuristic_training'):
        for path in sorted((ROOT / 'artifacts' / campaign / 'run_v1/jobs').glob('*/training_state/result.json')):
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('status') != 'SUCCEEDED' or not result.get('formal_training_completed'):
                continue
            best = result['best_mask']
            checkpoint = path.parent / best['file']
            if sha256(checkpoint) != best['sha256']:
                raise ValueError(f'Best checkpoint hash mismatch: {checkpoint}')
            epoch_rows = [row for row in result['history'] if row['epoch'] == best['epoch']]
            if len(epoch_rows) != 1 or abs(epoch_rows[0]['mask_map50_95'] - best['score']) > 1e-12:
                raise ValueError(f'Best epoch does not match recorded metric: {path}')
            metrics = epoch_rows[0]['metrics']
            if metrics['images'] != 100:
                raise ValueError('Expected fixed Validation100, not Test')
            rows.append({'job': path.parent.parent.name, 'experiment': result['experiment'], 'seed': result['seed'],
                'epochs': result['epochs'], 'best_epoch': best['epoch'], 'mask_map50_95': best['score'],
                'mask_map50': metrics['library_metrics']['metrics/mAP50(M)'],
                'mask_precision': metrics['mask_precision'], 'mask_recall': metrics['mask_recall'], 'mask_f1': metrics['mask_f1'],
                'operating_confidence': metrics['operating_confidence'], 'operating_mask_iou': metrics['operating_mask_iou'],
                'checkpoint': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': best['sha256'],
                'result_path': path.relative_to(ROOT).as_posix(), 'result_sha256': sha256(path),
                'initial_state': result['initial'], 'contract_sha256': result['contract_sha256']})
    if len(rows) != 22 or len({r['job'] for r in rows}) != 22:
        raise ValueError(f'Expected 22 unique completed jobs, found {len(rows)}')
    groups = defaultdict(list)
    for row in rows:
        groups[row['experiment']].append(row['mask_map50_95'])
    summary = {key: {'n': len(values), 'mean': statistics.mean(values),
                     'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
               for key, values in sorted(groups.items())}
    paired = {}
    by_id = {row['job']: row for row in rows}
    for synthetic, real in (('M1', 'M0'), ('L1', 'L0')):
        differences = [by_id[f'{synthetic}_seed{seed}']['mask_map50_95'] - by_id[f'{real}_seed{seed}']['mask_map50_95'] for seed in (0, 1, 2)]
        paired[f'{synthetic}-{real}'] = {'seed_differences': differences, 'mean': statistics.mean(differences),
                                        'sample_sd': statistics.stdev(differences)}
    out = ROOT / 'artifacts/s8_validation_preliminary/run_v1'
    payload = {'status': 'PRELIMINARY_RECORDED_VALIDATION_NOT_FINAL_TEST', 'rows': rows,
        'groups': summary, 'paired_heuristic_differences': paired,
        'remaining_training_jobs': 12, 'test_inference_performed': False,
        'caveats': ['Best epochs selected on this same Validation set; not independent Test estimates.',
            'Recorded training-time metrics, not newly rerun checkpoint inference.',
            'A screening40 versus M/L/H150 are different budgets; A/H are single-seed.',
            'Formal final evaluation freeze awaits generative training and contract audit.']}
    atomic_json(out / 'summary.json', payload, immutable=True)
    report = ['# Preliminary recorded Validation — 22 completed training jobs', '',
        'No new inference or Test access. Checkpoint SHA256 and selected-epoch metric identity verified.', '',
        '| Job | Epochs | Best epoch | Mask mAP50-95 | Mask mAP50 | Mask F1 @ conf .25 / IoU .5 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        report.append(f'| {row["job"]} | {row["epochs"]} | {row["best_epoch"]} | {row["mask_map50_95"]:.6f} | {row["mask_map50"]:.6f} | {row["mask_f1"]:.6f} |')
    report.extend(['', '## Caveats', '', *['- ' + text for text in payload['caveats']]])
    target = out / 'review.md'
    content = '\n'.join(report) + '\n'
    if target.exists() and target.read_text(encoding='utf-8') != content:
        raise ValueError('Do not overwrite changed review')
    if not target.exists():
        target.write_text(content, encoding='utf-8')
    print(json.dumps({'count': len(rows), 'groups': summary, 'paired': paired, 'path': str(out)}, indent=2))


if __name__ == '__main__':
    main()
