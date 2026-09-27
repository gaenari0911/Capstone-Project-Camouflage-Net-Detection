"""Bounded12-pair shape-conditioning repair, never a formal dataset release."""
from copy import deepcopy
import argparse
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import anydoor_pilot as pilot
from anydoor_runtime import sha256

RUN = pilot.ROOT / 'artifacts/s8_anydoor_shape_probe/run_v1'
BASE = pilot.ROOT / 'artifacts/s8_anydoor_pilot/run_v2'


def prepare():
    manifest = deepcopy(pilot.read(BASE / 'manifest.json'))
    ids = [f'{subset}_{index:03d}' for index in (1, 16, 24, 32, 40, 48)
           for subset in ('train748', 'low187')]
    manifest['records'] = [next(row for row in manifest['records'] if row['id'] == id) for id in ids]
    manifest['purpose'] = '12_PAIR_SHAPE_CONTROL_REPAIR_DIAGNOSTIC_NOT_TRAINING'
    manifest['settings']['shape_control'] = True
    manifest['comparison'] = {'baseline': pilot.rel(BASE), 'same_sources_seeds_crop_rgb_conditioning': True,
        'changed': 'Only hint fourth channel uses the target silhouette with nearest resizing; padding=-1 per official demo.',
        'not_exact_demo_pipeline': 'Baseline crop policy is preserved to isolate shape conditioning.',
        'quality_approval': 'PENDING', 'annotation_status': 'UNVERIFIED_NOT_GT'}
    pilot.atomic_json(RUN / 'manifest.json', manifest, immutable=True)
    print(json.dumps({'run': str(RUN), 'count': len(ids)}))


def review():
    records = pilot.read(RUN / 'manifest.json')['records']
    status = pilot.read(RUN / 'status.json')
    if status['completed'] != len(records) or status['state'] != 'PILOT_GENERATED_AWAITING_REVIEW':
        raise ValueError('Incomplete diagnostic')
    out = pilot.ROOT / 'artifacts/s8_anydoor_shape_review/run_v1'
    out.mkdir(parents=True, exist_ok=True)
    checks, hashes = 0, []
    for row in records:
        for run in (BASE, RUN):
            directory = run / 'samples' / row['id']
            proof = pilot.read(directory / 'COMMITTED.json')
            for file, digest in proof['files'].items():
                if sha256(directory / file) != digest:
                    raise ValueError(f'Hash mismatch: {directory / file}')
                checks += 1
        for name in ('reference.png', 'reference_mask.png', 'background.png', 'input_placement_mask.png'):
            if sha256(BASE / 'samples' / row['id'] / name) != sha256(RUN / 'samples' / row['id'] / name):
                raise ValueError('Paired inputs differ')
        meta = pilot.read(RUN / 'samples' / row['id'] / 'metadata.json')
        old = pilot.read(BASE / 'samples' / row['id'] / 'metadata.json')
        if meta['crop_yyxx'] != old['crop_yyxx'] or not meta['shape_control']:
            raise ValueError('Crop or conditioning regression')
        hashes.append(sha256(RUN / 'samples' / row['id'] / 'output.png'))
    for page in range(3):
        sheet = Image.new('RGB', (1200, 1280), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(records[page * 4:(page + 1) * 4]):
            y = index * 320
            cy1, cy2, cx1, cx2 = pilot.read(RUN / 'samples' / row['id'] / 'metadata.json')['crop_yyxx']
            draw.text((5, y + 2), row['id'] + ' / ' + row['domain'] + ' | RED IS INPUT, NOT GT', fill='black')
            paths = [BASE / 'samples' / row['id'] / 'reference.png',
                     BASE / 'samples' / row['id'] / 'output_overlay.png',
                     RUN / 'samples' / row['id'] / 'output_overlay.png']
            for column, (label, path) in enumerate(zip(('REFERENCE', 'BOX CONDITION baseline', 'SHAPE CONDITION repair'), paths)):
                with Image.open(path) as source:
                    img = source.convert('RGB')
                if column:
                    img = img.crop((cx1, cy1, cx2, cy2))
                img.thumbnail((390, 280))
                draw.text((column * 400 + 5, y + 18), label, fill='black')
                sheet.paste(img, (column * 400 + (400 - img.width) // 2, y + 36))
        path = out / f'comparison_{page + 1}.png'
        with path.open('xb') as stream:
            sheet.save(stream, format='PNG')
    pilot.atomic_json(out / 'technical_review.json', {'status': 'TECHNICAL_PASS_SEMANTIC_REVIEW_PENDING',
        'paired_count': len(records), 'hash_checks': checks, 'same_inputs_and_crop': True,
        'exact_duplicate_excess': len(hashes) - len(set(hashes)), 'verified_output_gt': False,
        'formal_release': False, 'user_quality_approval': 'PENDING'}, immutable=True)
    print(json.dumps({'review': str(out), 'hash_checks': checks}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'review'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else review()
