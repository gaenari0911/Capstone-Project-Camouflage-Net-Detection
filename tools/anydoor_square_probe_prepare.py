"""Preserved diagnostic contrast:12 same source/seed pairs, square input region."""
from copy import deepcopy
import json
import anydoor_pilot as pilot


def main():
    baseline = pilot.ROOT / 'artifacts/s8_anydoor_pilot/run_v2'
    manifest = deepcopy(pilot.read(baseline / 'manifest.json'))
    indices = [1, 16, 24, 32, 40, 48]
    ids = [f'{subset}_{index:03d}' for index in indices for subset in ('train748', 'low187')]
    by_id = {row['id']: row for row in manifest['records']}
    manifest['records'] = [by_id[id] for id in ids]
    manifest['purpose'] = '12_IMAGE_DIAGNOSTIC_INPUT_GEOMETRY_CONTRAST_NOT_FORMAL_POOL'
    manifest['settings']['input_geometry'] = 'square_region'
    manifest['settings']['placement'] = 'Centered square bounding region prevents double aspect compression; side=max requested object bbox side, capped at80percent of smaller canvas dimension. Actual object area may differ from the recorded heuristic plan.'
    manifest['comparison'] = {'baseline': pilot.rel(baseline), 'same_sources_and_seeds': True,
        'caveat': 'Input region/crop/actual object scale changes; not a single-variable performance ablation or GT validation.'}
    out = pilot.ROOT / 'artifacts/s8_anydoor_square_probe/run_v1'
    pilot.atomic_json(out / 'manifest.json', manifest, immutable=True)
    print(json.dumps({'count': len(ids), 'path': str(out)}))


if __name__ == '__main__':
    main()
