"""Bounded, resumable AnyDoor pilot. This is NOT a formal candidate campaign."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capstone_lab.campaign.io import RunLock, atomic_json
from anydoor_runtime import SOURCE_REVISION, OFFICIAL_SHA256, sha256

SETUP = ROOT / 'artifacts/s8_anydoor_setup/run_v1'
SOURCE = SETUP / 'source' / ('AnyDoor-' + SOURCE_REVISION)
WEIGHTS = SETUP / 'weights/anydoor_official.ckpt'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def prepare(run):
    """Balanced diagnostic pairs, not a scored/training selection manifest."""
    records = []
    for subset in ('train748', 'low187'):
        provenance_path = ROOT / f'artifacts/s8_source_audit/split_pools_v1/{subset}_object_provenance.json'
        provenance = read(provenance_path)
        allowed = {row['object_image']: row for row in provenance['records']}
        grouped = defaultdict(list)
        base = ROOT / f'artifacts/s8_heuristic_formal/run_v1/datasets/{subset}/samples'
        for path in sorted(base.glob('*/A1/metadata.json')):
            metadata = read(path)
            fg = metadata['foreground']
            if fg['image_path'] not in allowed:
                raise ValueError(f'Forbidden {subset} source: {fg}')
            grouped[(metadata['plan']['domain'], metadata['plan']['scale_bucket'])].append((path, metadata))
        selected = []
        while len(selected) < 50:
            progressed = False
            for key in sorted(grouped):
                if grouped[key] and len(selected) < 50:
                    selected.append(grouped[key].pop(0))
                    progressed = True
            if not progressed:
                raise ValueError('Insufficient diagnostic coverage')
        for number, (path, metadata) in enumerate(selected):
            fg, bg = metadata['foreground'], metadata['background']
            proof = allowed[fg['image_path']]
            files = {fg['image_path']: proof['object_image_sha256'],
                     fg['mask_path']: proof['object_mask_sha256'],
                     proof['source_image']: proof['source_sha256'],
                     proof['label']: proof['label_sha256'], bg['path']: bg['sha256']}
            for file, digest in files.items():
                if sha256(ROOT / file) != digest:
                    raise ValueError(f'Source integrity failed: {file}')
            records.append({'id': f'{subset}_{number + 1:03d}', 'subset': subset,
                'seed': int(hashlib.sha256(f'anydoor-pilot-v1:{subset}:{number}'.encode()).hexdigest()[:8], 16),
                'domain': bg['domain'], 'environment': fg['environment'],
                'scale_bucket': metadata['plan']['scale_bucket'],
                'scale_ratio': metadata['plan']['scale_ratio'],
                'reference': fg['image_path'], 'reference_mask': fg['mask_path'],
                'background': bg['path'], 'source_image': fg['source_image'],
                'source_hashes': files, 'provenance_path': rel(provenance_path),
                'provenance_sha256': sha256(provenance_path),
                'diagnostic_pair_origin': rel(path), 'origin_sha256': sha256(path)})
    # Interleave subsets for an informative first smoke and early visual review.
    records = [records[index + offset] for index in range(50) for offset in (0, 50)]
    manifest = {'schema_version': 1, 'purpose': 'PILOT_ONLY_NOT_TRAINING',
        'records': records, 'source_revision': SOURCE_REVISION, 'weights_sha256': OFFICIAL_SHA256,
        'settings': {'ddim_steps': 50, 'guidance_scale': 5.0, 'eta': 0.0,
            'control_strength': 1.0, 'local_resolution': 512, 'background_long_side': 1280,
            'precision': 'fp16_weights_autocast_sdpa', 'gpu_concurrency': 1,
            'placement': 'centered resized original mask; area uses recorded bbox scale ratio',
            'official_crop_row_drop_preserved': True},
        'annotation_status': 'INPUT_PLACEMENT_ONLY_NOT_VERIFIED_GT',
        'selection_caveat': 'Diagnostic pairs reuse A1 chosen source pairs, not formal candidates. '
            'Original backgrounds/no heuristic rendering. Balanced domain/scale coverage is not natural frequency.',
        'time_ceiling_hours': None, 'pilot_user_approval': 'PENDING'}
    atomic_json(run / 'manifest.json', manifest, immutable=True)
    print(json.dumps({'status': 'PREPARED', 'count': len(records),
                      'domains': dict(Counter(r['domain'] for r in records)), 'path': str(run)}))


def inputs(record, input_geometry='reference_mask'):
    import cv2
    import numpy as np
    def rgb(path):
        image = cv2.imdecode(np.fromfile(ROOT / path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reference = rgb(record['reference'])
    mask = cv2.imdecode(np.fromfile(ROOT / record['reference_mask'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 128).astype(np.uint8)
    if mask.shape != reference.shape[:2] or mask.sum() == 0:
        raise ValueError('Reference mask mismatch/empty')
    background = rgb(record['background'])
    height, width = background.shape[:2]
    factor = 1280 / max(height, width)
    background = cv2.resize(background, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    height, width = background.shape[:2]
    ys, xs = np.where(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = (height * width * record['scale_ratio'] / crop.size) ** .5
    scale = min(scale, .8 * width / crop.shape[1], .8 * height / crop.shape[0])
    resized = cv2.resize(crop, (max(8, round(crop.shape[1] * scale)), max(8, round(crop.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    target = np.zeros((height, width), np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    target[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    if input_geometry == 'square_region':
        side = min(max(resized.shape), int(.8 * min(height, width)))
        target[:] = 0
        y, x = (height - side) // 2, (width - side) // 2
        target[y:y + side, x:x + side] = 1
    elif input_geometry != 'reference_mask':
        raise ValueError(f'Unknown input geometry: {input_geometry}')
    return reference, mask, background, target


def save_png(path, array, rgb=True):
    import cv2
    if rgb:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode('.png', array)
    if not ok:
        raise ValueError(f'PNG encoding failed: {path}')
    with Path(path).open('xb') as stream:
        stream.write(encoded.tobytes())
        stream.flush()
        os.fsync(stream.fileno())


def budget():
    from capstone_lab.campaign.resources import budget_roots
    # DirEntry caches Windows directory metadata; repeated Path.stat on every
    # historical artifact makes a safety check unnecessarily slow.
    total = 0
    pending = list(budget_roots(ROOT))
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError(f'Unexpected symlink in budget roots: {entry.path}')
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
    free = shutil.disk_usage(ROOT).free
    if total > 100 * 2**30 or free < 20 * 2**30:
        raise ValueError(f'Disk capacity guard: used={total}, free={free}')
    return {'s8_bytes': total, 'free_bytes': free}


def committed(directory, binding):
    marker = directory / 'COMMITTED.json'
    if not marker.exists():
        return False
    proof = read(marker)
    if proof['binding'] != binding:
        raise ValueError(f'Changed runtime/manifest: {directory}')
    for file, digest in proof['files'].items():
        if sha256(directory / file) != digest:
            raise ValueError(f'Output integrity failed: {directory / file}')
    return True


def gallery(run, records):
    sections = []
    for record in records:
        directory = run / 'samples' / record['id']
        if not (directory / 'COMMITTED.json').exists():
            continue
        prefix = 'samples/' + record['id'] + '/'
        images = ''.join(f'<figure><img width="420" src="{prefix}{name}.png"><figcaption>{name}</figcaption></figure>'
                         for name in ('reference', 'background', 'placement_overlay', 'output', 'output_overlay'))
        sections.append(f'<h2>{html.escape(record["id"])} / {html.escape(record["domain"])} / {html.escape(record["scale_bucket"])}</h2><div class="row">{images}</div>')
    content = '<meta charset="utf-8"><title>AnyDoor pilot review</title><style>.row{display:flex;flex-wrap:wrap}figure{margin:5px}img{max-width:100%}</style><h1>Pilot only — input mask is NOT verified output GT</h1><p>Check identity/net texture, output silhouette versus red input contour, background seams and unwanted objects. User approval PENDING.</p>' + ''.join(sections)
    from capstone_lab.campaign.io import atomic_replace
    temp = run / ('review.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(content, encoding='utf-8')
    atomic_replace(temp, run / 'review.html')


def execute(run, limit, parity_run=None):
    import cv2
    import numpy as np
    import psutil
    import torch
    from anydoor_runtime import load_model, generate
    from capstone_lab.campaign.detached import job_info
    from capstone_lab.campaign.resources import gpu_memory
    if not 1 <= limit <= 100:
        raise ValueError('Pilot must stay within 1..100; no hidden formal generation')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    manifest = read(run / 'manifest.json')
    precision = manifest['settings']['precision']
    if precision not in {'fp16_weights_autocast_sdpa', 'fp32_offload'}:
        raise ValueError('Unknown inference precision')
    parity_markers = {} if parity_run is None else {
        p.parent.name: sha256(p) for p in sorted(parity_run.glob('samples/*/COMMITTED.json'))}
    if parity_run is not None:
        if len(parity_markers) < 2 or sha256(parity_run / 'manifest.json') != sha256(run / 'manifest.json'):
            raise ValueError('Parity smoke requires at least2 committed samples with the same input manifest')
    binding_record = {'manifest': sha256(run / 'manifest.json'),
        'parity_reference': None if parity_run is None else rel(parity_run), 'parity_markers': parity_markers,
        'environment': sha256(SETUP / 'environment_verified.json'),
        'time_approval': sha256(ROOT / 'configs/approvals/s8_time_limit_removed_v1.json'),
        'runtime': sha256(Path(__file__).with_name('anydoor_runtime.py')),
        'runner': sha256(__file__), 'source': {p.relative_to(SOURCE).as_posix(): sha256(p) for p in sorted(SOURCE.rglob('*'))
                    if p.is_file() and p.suffix in {'.py', '.yaml'}}, 'torch': torch.__version__}
    binding = hashlib.sha256(json.dumps(binding_record, sort_keys=True).encode()).hexdigest()
    records = manifest['records'][:limit]
    started = time.monotonic()
    with RunLock(run / 'run.lock'), RunLock(ROOT / 'artifacts/s8_gpu.lock'):
        telemetry = budget()
        _, used = gpu_memory()
        if used > 2500 or psutil.virtual_memory().available < 9 * 2**30:
            raise ValueError(f'Insufficient idle GPU/RAM: GPU used={used} MiB')
        atomic_json(run / 'binding.json', binding_record, immutable=True)
        base_status = {'pid': os.getpid(), 'job_info': job_info(), 'requested': limit,
                       'planned': len(manifest['records']), 'time_ceiling_hours': None, 'disk': telemetry}
        atomic_json(run / 'status.json', {**base_status, 'state': 'LOADING', 'completed': 0})
        runtime = load_model(SOURCE, WEIGHTS, precision)
        load_seconds = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats()
        for index, record in enumerate(records):
            directory = run / 'samples' / record['id']
            if committed(directory, binding):
                continue
            for path, digest in record['source_hashes'].items():
                if sha256(ROOT / path) != digest:
                    raise ValueError(f'Source changed: {path}')
            if index % 10 == 0:
                telemetry = budget()
            atomic_json(run / 'status.json', {**base_status, 'state': 'GENERATING', 'completed': index,
                'active_sample': record['id'], 'utc': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': time.monotonic() - started, 'disk': telemetry})
            attempt = run / 'attempts' / (record['id'] + '_' + uuid.uuid4().hex)
            attempt.mkdir(parents=True, exist_ok=False)
            sample_started = time.monotonic()
            reference, mask, background, target = inputs(record, manifest['settings'].get('input_geometry', 'reference_mask'))
            generated, info = generate(runtime, reference, mask, background, target, record['seed'], precision,
                shape_control=manifest['settings'].get('shape_control', False))
            if generated.shape != background.shape or np.array_equal(generated, background):
                raise ValueError('Invalid/unchanged generated output')
            contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            def overlay(image):
                return cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 2)
            for name, value in (('reference', reference), ('background', background), ('placement_overlay', overlay(background)),
                                ('output', generated), ('output_overlay', overlay(generated))):
                save_png(attempt / f'{name}.png', value)
            save_png(attempt / 'reference_mask.png', mask * 255, rgb=False)
            save_png(attempt / 'input_placement_mask.png', target * 255, rgb=False)
            parity = 'NOT_IN_REFERENCE_SMOKE'
            if record['id'] in parity_markers:
                reference_directory = parity_run / 'samples' / record['id']
                proof = read(reference_directory / 'COMMITTED.json')
                if sha256(reference_directory / 'COMMITTED.json') != parity_markers[record['id']]:
                    raise ValueError('Parity reference changed')
                expected_hash = proof['files']['output.png']
                if sha256(reference_directory / 'output.png') != expected_hash or sha256(attempt / 'output.png') != expected_hash:
                    raise ValueError(f'Output parity failed after low-RAM loading: {record["id"]}')
                parity = 'EXACT_OUTPUT_HASH_MATCH'
            atomic_json(attempt / 'metadata.json', {**record, **info, 'annotation_status': manifest['annotation_status'],
                'low_ram_loader_parity': parity,
                'generation_seconds': time.monotonic() - sample_started,
                'peak_cuda_allocated': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved': torch.cuda.max_memory_reserved(), 'attention_test': runtime[-1],
                'process_rss': psutil.Process().memory_info().rss,
                'peak_process_rss': getattr(psutil.Process().memory_info(), 'peak_wset', None),
                'load_seconds': load_seconds})
            proof = {'binding': binding, 'files': {p.name: sha256(p) for p in attempt.iterdir() if p.is_file()}}
            atomic_json(attempt / 'COMMITTED.json', proof, immutable=True)
            directory.parent.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                raise ValueError(f'Uncommitted destination preserved, manual inspection required: {directory}')
            attempt.rename(directory)
            gallery(run, records)
            print(json.dumps({'completed': index + 1, 'requested': limit, 'sample': record['id'],
                              'seconds': time.monotonic() - sample_started}), flush=True)
        atomic_json(run / 'status.json', {**base_status, 'state': 'PILOT_GENERATED_AWAITING_REVIEW' if limit == len(manifest['records']) else 'SMOKE_GENERATED',
            'completed': len(records), 'elapsed_seconds': time.monotonic() - started,
            'user_quality_approval': 'PENDING', 'formal_candidate_generation': 'NOT_STARTED'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--run', default='artifacts/s8_anydoor_pilot/run_v1')
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--parity-run')
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    run.relative_to(ROOT / 'artifacts')
    run.mkdir(parents=True, exist_ok=True)
    try:
        parity_run = None if args.parity_run is None else (ROOT / args.parity_run).resolve()
        if parity_run is not None:
            parity_run.relative_to(ROOT / 'artifacts')
        prepare(run) if args.command == 'prepare' else execute(run, args.limit, parity_run)
    except Exception:
        error = traceback.format_exc()
        atomic_json(run / ('failure_' + uuid.uuid4().hex + '.json'), {'traceback': error, 'utc': datetime.now(timezone.utc).isoformat()})
        # Do not overwrite another running owner's live status on lock failure.
        status_path = run / 'status.json'
        if status_path.exists():
            status = read(status_path)
            if status.get('pid') == os.getpid():
                atomic_json(status_path, {**status, 'state': 'FAILED', 'error': error,
                                         'utc': datetime.now(timezone.utc).isoformat()})
        print(error, file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
