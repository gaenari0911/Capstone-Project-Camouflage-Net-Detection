"""Two-case full-precision diagnostic; no new sources or large generation."""
from copy import deepcopy
import json
import anydoor_pilot as pilot


def main():
    baseline = pilot.ROOT / 'artifacts/s8_anydoor_square_probe/run_v1'
    manifest = deepcopy(pilot.read(baseline / 'manifest.json'))
    by_id = {r['id']: r for r in manifest['records']}
    manifest['records'] = [by_id[id] for id in ('low187_016', 'train748_048')]
    manifest['purpose'] = '2_IMAGE_FP32_FIDELITY_DIAGNOSTIC_NOT_FORMAL_POOL'
    manifest['settings']['precision'] = 'fp32_offload'
    manifest['comparison'] = {'baseline': pilot.rel(baseline), 'same_sources_and_seeds_and_geometry': True,
        'change': 'Full original checkpoint floating-point precision, no autocast, official low_vram_shift.'}
    out = pilot.ROOT / 'artifacts/s8_anydoor_fp32_probe/run_v1'
    pilot.atomic_json(out / 'manifest.json', manifest, immutable=True)
    print(json.dumps({'count': 2, 'path': str(out)}))


if __name__ == '__main__':
    main()
