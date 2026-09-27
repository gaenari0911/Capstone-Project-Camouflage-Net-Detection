"""Inspect official collage silhouette WITHOUT modifying generator inputs/outputs.

The target mask only defines a bounding box in official AnyDoor. The collage
silhouette is therefore a more informative proxy, but is still NOT output GT.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

import anydoor_pilot as pilot
from anydoor_runtime import activate, load_pair_functions, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', default='artifacts/s8_anydoor_pilot/run_v1')
    parser.add_argument('--out', default='artifacts/s8_anydoor_geometry_review/run_v1')
    args = parser.parse_args()
    source_run = (pilot.ROOT / args.source_run).resolve()
    out = (pilot.ROOT / args.out).resolve()
    source_run.relative_to(pilot.ROOT / 'artifacts')
    out.relative_to(pilot.ROOT / 'artifacts')
    activate(pilot.SOURCE)
    process_pairs, _ = load_pair_functions(pilot.SOURCE)
    rows = []
    manifest = pilot.read(source_run / 'manifest.json')
    for record in manifest['records']:
        sample = source_run / 'samples' / record['id']
        if not (sample / 'COMMITTED.json').exists():
            continue
        marker = pilot.read(sample / 'COMMITTED.json')
        if sha256(sample / 'output.png') != marker['files']['output.png']:
            raise ValueError('Generated output hash changed')
        source_inputs = pilot.inputs(record, manifest['settings'].get('input_geometry', 'reference_mask'))
        captured = {}
        def trace(frame, event, arg):
            if event == 'return' and frame.f_code is process_pairs.__code__:
                for key in ('ref_mask_compose', 'tar_box_yyxx', 'tar_box_yyxx_crop'):
                    captured[key] = frame.f_locals[key]
        old_profile = sys.getprofile()
        random.seed(record['seed'])
        np.random.seed(record['seed'])
        try:
            sys.setprofile(trace)
            process_pairs(*source_inputs)
        finally:
            sys.setprofile(old_profile)
        cy1, cy2, cx1, cx2 = captured['tar_box_yyxx_crop']
        y1, y2, x1, x2 = captured['tar_box_yyxx']
        proxy = np.zeros(source_inputs[3].shape, np.uint8)
        proxy[cy1 + y1:cy1 + y2, cx1 + x1:cx1 + x2] = captured['ref_mask_compose']
        output = cv2.cvtColor(cv2.imdecode(np.fromfile(sample / 'output.png', np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        contours, _ = cv2.findContours(proxy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.drawContours(output.copy(), contours, -1, (0, 255, 255), 2)
        destination = out / record['id']
        destination.mkdir(parents=True, exist_ok=True)
        for name, value, rgb in (('conditioning_silhouette.png', proxy * 255, False), ('conditioning_overlay.png', overlay, True)):
            target = destination / name
            if not target.exists():
                pilot.save_png(target, value, rgb=rgb)
        requested = source_inputs[3] > 0
        intersection = np.count_nonzero(requested & (proxy > 0))
        union = np.count_nonzero(requested | (proxy > 0))
        row = {'id': record['id'], 'source_run': pilot.rel(source_run), 'generated_output_sha256': marker['files']['output.png'],
               'input_vs_collage_mask_iou': intersection / union, 'output_ground_truth_status': 'UNVERIFIED',
               'note': 'Cyan contour follows the actual official collage reference silhouette BEFORE512 interpolation. '
                       'It is diagnostic conditioning geometry, not a segmentation of generated pixels.'}
        pilot.atomic_json(destination / 'metadata.json', row, immutable=True)
        rows.append(row)
    pilot.atomic_json(out / 'summary.json', {'status': 'DIAGNOSTIC_PROXY_ONLY', 'count': len(rows), 'records': rows})
    print(json.dumps({'count': len(rows), 'first_rows': rows[:2], 'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
