"""Audit completed pilot files and prepare prespecified24-case visual review sheets."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
from PIL import Image, ImageDraw

import anydoor_pilot as pilot
from anydoor_runtime import sha256


def rgb(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    run = pilot.ROOT / 'artifacts/s8_anydoor_pilot/run_v2'
    geometry = pilot.ROOT / 'artifacts/s8_anydoor_geometry_review/run_v2'
    out = pilot.ROOT / 'artifacts/s8_anydoor_pilot_audit/run_v1'
    manifest = pilot.read(run / 'manifest.json')
    records = manifest['records']
    status = pilot.read(run / 'status.json')
    if status['state'] != 'PILOT_GENERATED_AWAITING_REVIEW' or status['completed'] != 100:
        raise ValueError('Wait for all100 pilot samples before final audit')
    image_hashes, bindings, metadata_rows, source_files = [], set(), [], {}
    for record in records:
        directory = run / 'samples' / record['id']
        marker = pilot.read(directory / 'COMMITTED.json')
        bindings.add(marker['binding'])
        for name, digest in marker['files'].items():
            if sha256(directory / name) != digest:
                raise ValueError(f'Artifact integrity mismatch: {directory / name}')
        metadata = pilot.read(directory / 'metadata.json')
        metadata_rows.append(metadata)
        image_hashes.append(marker['files']['output.png'])
        source_files.update(record['source_hashes'])
        generated, background = rgb(directory / 'output.png'), rgb(directory / 'background.png')
        if generated.shape != background.shape:
            raise ValueError('Output shape mismatch')
        y1, y2, x1, x2 = metadata['crop_yyxx']
        difference = np.any(generated != background, axis=2)
        difference[y1:y2, x1:x2] = False
        if np.any(difference):
            raise ValueError('Unexpected edits outside official crop')
        if not (geometry / record['id'] / 'conditioning_overlay.png').exists():
            raise ValueError('Run geometry review for the full completed pilot first')
    if len(bindings) != 1:
        raise ValueError('Mixed runtime bindings')
    for path, digest in source_files.items():
        if sha256(pilot.ROOT / path) != digest:
            raise ValueError(f'Source integrity mismatch: {path}')
    selected = []
    for domain in sorted({r['domain'] for r in records}):
        for subset in ('train748', 'low187'):
            cases = [r for r in records if r['domain'] == domain and r['subset'] == subset]
            # Pick both ends of available scale order, independent of output quality.
            cases.sort(key=lambda r: (r['scale_ratio'], r['id']))
            selected.extend([cases[0], cases[-1]])
    out.mkdir(parents=True, exist_ok=True)
    for page in range(4):
        sheet = Image.new('RGB', (1200, 960), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(selected[page * 6:(page + 1) * 6]):
            col, row = index % 2, index // 2
            x, y = col * 600, row * 320
            sample = run / 'samples' / record['id']
            metadata = pilot.read(sample / 'metadata.json')
            reference = Image.fromarray(rgb(sample / 'reference.png'))
            generated = rgb(geometry / record['id'] / 'conditioning_overlay.png')
            y1, y2, x1, x2 = metadata['crop_yyxx']
            crop = Image.fromarray(generated[y1:y2, x1:x2])
            for offset, image in ((0, reference), (300, crop)):
                image.thumbnail((290, 265))
                sheet.paste(image, (x + offset + (290 - image.width) // 2, y + 40 + (265 - image.height) // 2))
            draw.text((x + 5, y + 5), f'{record["id"]} {record["domain"]} scale={record["scale_ratio"]:.3f}', fill='black')
            draw.text((x + 5, y + 20), 'Reference                 Output crop + CYAN conditioning proxy (NOT GT)', fill='black')
        target = out / f'contact_sheet_{page + 1}.png'
        if target.exists():
            raise ValueError('Preserve existing visual sheet; choose a new audit version')
        sheet.save(target)
    seconds = [r['generation_seconds'] for r in metadata_rows]
    report = {'status': 'TECHNICAL_AUDIT_PASS_SEMANTIC_QUALITY_PENDING', 'count': len(records),
        'subsets': dict(Counter(r['subset'] for r in records)),
        'domains': dict(Counter(r['domain'] for r in records)),
        'exact_output_duplicate_excess': len(image_hashes) - len(set(image_hashes)),
        'source_files_hash_verified': len(source_files), 'output_files_hash_verified': sum(len(pilot.read(run / 'samples' / r['id'] / 'COMMITTED.json')['files']) for r in records),
        'changed_pixels_outside_official_crop': 0,
        'generation_seconds_median': statistics.median(seconds), 'generation_seconds_mean': statistics.mean(seconds),
        'peak_process_rss': max(r['peak_process_rss'] for r in metadata_rows),
        'peak_cuda_allocated': max(r['peak_cuda_allocated'] for r in metadata_rows),
        'peak_cuda_reserved': max(r['peak_cuda_reserved'] for r in metadata_rows),
        'low_ram_exact_parity_samples': sum(r['low_ram_loader_parity'] == 'EXACT_OUTPUT_HASH_MATCH' for r in metadata_rows),
        'selected_visual_review_ids': [r['id'] for r in selected],
        'visual_review_status': 'SHEETS_PREPARED_NOT_AUTOMATICALLY_REVIEWED',
        'user_quality_approval': 'PENDING', 'formal_generation_started': False,
        'annotations_are_verified_gt': False, 'manifest_sha256': sha256(run / 'manifest.json')}
    pilot.atomic_json(out / 'summary.json', report, immutable=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
