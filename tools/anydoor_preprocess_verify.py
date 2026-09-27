"""Verify all100 diagnostic inputs through the unmodified official preprocessing."""
import json
import random
import time

import numpy as np

import anydoor_pilot as pilot
from anydoor_runtime import activate, load_pair_functions, sha256


def main():
    run = pilot.ROOT / 'artifacts/s8_anydoor_pilot/run_v1'
    manifest = pilot.read(run / 'manifest.json')
    activate(pilot.SOURCE)
    process_pairs, crop_back = load_pair_functions(pilot.SOURCE)
    started = time.monotonic()
    records = []
    for record in manifest['records']:
        inputs = pilot.inputs(record)
        np.random.seed(record['seed'])
        random.seed(record['seed'])
        item = process_pairs(*inputs)
        if item['ref'].shape != (224, 224, 3) or item['hint'].shape != (512, 512, 4):
            raise ValueError(f'Unexpected official input shapes: {record["id"]}')
        if not all(np.isfinite(item[key]).all() for key in ('ref', 'hint', 'jpg')):
            raise ValueError('Nonfinite preprocessing')
        restored = crop_back(np.full((511, 512, 3), 127, np.uint8), inputs[2].copy(),
                             item['extra_sizes'], item['tar_box_yyxx_crop'])
        if restored.shape != inputs[2].shape:
            raise ValueError('Crop-back dimension mismatch')
        np.random.seed(record['seed'])
        random.seed(record['seed'])
        repeat = process_pairs(*inputs)
        if any(not np.array_equal(item[key], repeat[key]) for key in item):
            raise ValueError('Preprocessing is not repeatable with the same sample seed')
        records.append({'id': record['id'], 'crop': item['tar_box_yyxx_crop'].tolist(),
                        'input_mask_pixels': int(inputs[3].sum())})
    report = {'status': 'VERIFIED_OFFICIAL_PREPROCESSING_NOT_GENERATION', 'count': len(records),
              'elapsed_seconds': time.monotonic() - started, 'manifest_sha256': sha256(run / 'manifest.json'),
              'records': records, 'semantic_gt_verified': False}
    pilot.atomic_json(pilot.SETUP / 'preprocessing_verified.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}))


if __name__ == '__main__':
    main()
