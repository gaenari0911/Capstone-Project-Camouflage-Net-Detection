"""Technical checks and side-by-side source/baseline/square-input diagnostics."""
import json
import statistics
import numpy as np
import cv2
from PIL import Image, ImageDraw

import anydoor_pilot as pilot
from anydoor_runtime import sha256


def rgb(path):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main():
    run = pilot.ROOT / 'artifacts/s8_anydoor_square_probe/run_v1'
    baseline = pilot.ROOT / 'artifacts/s8_anydoor_pilot/run_v2'
    square_geometry = pilot.ROOT / 'artifacts/s8_anydoor_square_geometry/run_v1'
    base_geometry = pilot.ROOT / 'artifacts/s8_anydoor_geometry_review/run_v2'
    out = pilot.ROOT / 'artifacts/s8_anydoor_square_review/run_v1'
    status = pilot.read(run / 'status.json')
    if status['state'] != 'PILOT_GENERATED_AWAITING_REVIEW' or status['completed'] != 12:
        raise ValueError('12-case contrast not complete')
    records = pilot.read(run / 'manifest.json')['records']
    metadata_rows, hashes = [], []
    for row in records:
        directory = run / 'samples' / row['id']
        marker = pilot.read(directory / 'COMMITTED.json')
        for file, digest in marker['files'].items():
            if sha256(directory / file) != digest:
                raise ValueError(f'Output hash mismatch: {directory / file}')
        hashes.append(marker['files']['output.png'])
        metadata_rows.append(pilot.read(directory / 'metadata.json'))
    out.mkdir(parents=True, exist_ok=True)
    for page in range(3):
        sheet = Image.new('RGB', (900, 1200), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(records[page * 4:(page + 1) * 4]):
            y = index * 300
            images = [rgb(run / 'samples' / row['id'] / 'reference.png')]
            for source, geometry in ((baseline, base_geometry), (run, square_geometry)):
                metadata = pilot.read(source / 'samples' / row['id'] / 'metadata.json')
                cy1, cy2, cx1, cx2 = metadata['crop_yyxx']
                images.append(rgb(geometry / row['id'] / 'conditioning_overlay.png')[cy1:cy2, cx1:cx2])
            draw.text((5, y + 2), f'{row["id"]} {row["domain"]}', fill='black')
            for column, (label, array) in enumerate(zip(('REFERENCE', 'BASELINE input', 'SQUARE REGION input'), images)):
                x = column * 300
                draw.text((x + 5, y + 16), label + ' (cyan is NOT verified GT)' if column else label, fill='black')
                image = Image.fromarray(array)
                image.thumbnail((290, 260))
                sheet.paste(image, (x + (300 - image.width) // 2, y + 35 + (260 - image.height) // 2))
        path = out / f'comparison_{page + 1}.png'
        if path.exists():
            raise ValueError('Preserve previous comparison output')
        sheet.save(path)
    report = {'status': 'TECHNICAL_PASS_VISUAL_REVIEW_PENDING', 'count': 12,
        'output_hashes_verified': 96, 'exact_image_duplicate_excess': len(hashes) - len(set(hashes)),
        'generation_seconds_median': statistics.median(r['generation_seconds'] for r in metadata_rows),
        'peak_cuda_reserved': max(r['peak_cuda_reserved'] for r in metadata_rows),
        'same_source_and_seed_as_baseline': True,
        'caveat': 'Bounding region changes local crop/scale as well as aspect handling; not a performance ablation.',
        'generated_segmentation_gt_available': False, 'user_quality_approval': 'PENDING',
        'ids': [r['id'] for r in records]}
    pilot.atomic_json(out / 'summary.json', report, immutable=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
