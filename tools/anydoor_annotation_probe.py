"""Independent SAM2 proposals/calibration. NO automatic GT or training release."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw
import anydoor_pilot as pilot
from anydoor_runtime import sha256

SETUP = pilot.ROOT / 'artifacts/s8_anydoor_annotation_setup/run_v1'
SOURCE = SETUP / 'source/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4'
WEIGHTS = SETUP / 'sam2.1_hiera_large.pt'
WEIGHT_SHA = '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
RUN = pilot.ROOT / 'artifacts/s8_anydoor_shape_probe/run_v1'
OUT = pilot.ROOT / 'artifacts/s8_anydoor_annotation_probe/run_v1'


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def prompt(mask):
    y, x = np.where(mask)
    if not len(y):
        raise ValueError('Empty prompt')
    h, w = mask.shape
    dx, dy = (x.max() - x.min() + 1) * .2, (y.max() - y.min() + 1) * .2
    box = np.array([max(0, x.min() - dx), max(0, y.min() - dy),
                    min(w - 1, x.max() + dx), min(h - 1, y.max() + dy)], np.float32)
    # An interior point, not the box center which may lie outside a thin mask.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(distance.argmax(), distance.shape)
    return box, np.array([[px, py]], np.float32)


def annotation_is_releasable(decision, image_sha, mask_sha):
    """Fail closed: model confidence/proposal agreement is never approval."""
    return (decision.get('status') == 'VERIFIED_ANNOTATION'
        and decision.get('image_sha256') == image_sha and decision.get('mask_sha256') == mask_sha
        and decision.get('identity_review') == 'PASS' and decision.get('extra_object_review') == 'PASS'
        and decision.get('mask_review') == 'PASS' and decision.get('reviewer_role') == 'human'
        and bool(decision.get('reviewer')))


def main():
    import torch
    from capstone_lab.campaign.resources import gpu_memory
    started = time.monotonic()
    manifest = pilot.read(RUN / 'manifest.json')
    if pilot.read(RUN / 'status.json')['state'] != 'PILOT_GENERATED_AWAITING_REVIEW':
        raise ValueError('Generation incomplete')
    if sha256(WEIGHTS) != WEIGHT_SHA:
        raise ValueError('SAM2 weight hash mismatch')
    sys.path[:0] = [str(SETUP / 'vendor'), str(SOURCE)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    with pilot.RunLock(OUT / 'run.lock'), pilot.RunLock(pilot.ROOT / 'artifacts/s8_gpu.lock'):
        if gpu_memory()[1] > 2500:
            raise ValueError('GPU not idle')
        binding = {'source_commit': SOURCE.name, 'weights_sha256': WEIGHT_SHA,
            'source_files': {p.relative_to(SOURCE).as_posix(): sha256(p) for p in SOURCE.rglob('*')
                             if p.is_file() and p.suffix in {'.py', '.yaml'}},
            'runner_sha256': sha256(__file__), 'manifest_sha256': sha256(RUN / 'manifest.json'),
            'torch': torch.__version__, 'postprocessing': 'No hole filling/sprinkle removal/custom CUDA extension',
            'source_dataset_training': 'NONE', 'judge_used': False, 'test_used': False}
        pilot.atomic_json(OUT / 'binding.json', binding, immutable=True)
        model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', str(WEIGHTS), apply_postprocessing=False)
        predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
        rows = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for record in manifest['records']:
                sample = RUN / 'samples' / record['id']
                proof = pilot.read(sample / 'COMMITTED.json')
                for file, digest in proof['files'].items():
                    if sha256(sample / file) != digest:
                        raise ValueError('Generated input integrity failure')
                destination = OUT / 'samples' / record['id']
                if (destination / 'COMMITTED.json').exists():
                    committed = pilot.read(destination / 'COMMITTED.json')
                    if committed['binding_sha256'] != sha256(OUT / 'binding.json'):
                        raise ValueError('Changed annotation runtime')
                    for file, digest in committed['files'].items():
                        if sha256(destination / file) != digest:
                            raise ValueError('Proposal integrity failure')
                    rows.append(pilot.read(destination / 'metadata.json'))
                    continue
                if destination.exists():
                    raise ValueError('Preserve partial sample; use a new version after inspection')
                destination.mkdir(parents=True)
                reference, reference_mask, _, target = pilot.inputs(record)
                with Image.open(sample / 'output.png') as img:
                    output = np.asarray(img.convert('RGB'))
                proposals = {}
                for name, rgb, seed_mask in (('reference', reference, reference_mask), ('output', output, target)):
                    predictor.set_image(rgb)
                    box, points = prompt(seed_mask)
                    variants = {}
                    for variant, use_points in (('box', False), ('point_box', True)):
                        masks, scores, _ = predictor.predict(box=box,
                            point_coords=points if use_points else None,
                            point_labels=np.ones(1, np.int32) if use_points else None,
                            multimask_output=True)
                        if not np.isfinite(scores).all():
                            raise ValueError('Nonfinite annotation scores')
                        selected = int(np.argmax(scores))
                        variants[variant] = (masks[selected], scores.tolist())
                        for index, mask in enumerate(masks):
                            pilot.save_png(destination / f'{name}_{variant}_{index}.png', mask.astype(np.uint8) * 255, rgb=False)
                    chosen, scores = variants['point_box']
                    pilot.save_png(destination / f'{name}_proposal.png', chosen.astype(np.uint8) * 255, rgb=False)
                    contours, _ = cv2.findContours(chosen.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    overlay = cv2.drawContours(rgb.copy(), contours, -1, (0, 255, 255), 2)
                    pilot.save_png(destination / f'{name}_overlay.png', overlay)
                    proposals[name] = {'model_predicted_ious': scores,
                        'prompt_variant_agreement': iou(chosen, variants['box'][0]),
                        'iou_to_reference_gt' if name == 'reference' else 'iou_to_INPUT_NOT_GT': iou(chosen, seed_mask),
                        'box': box.tolist(), 'positive_point': points.tolist()}
                row = {'id': record['id'], 'subset': record['subset'], 'domain': record['domain'],
                    'image_sha256': proof['files']['output.png'],
                    'proposal_sha256': sha256(destination / 'output_proposal.png'),
                    'annotation_status': 'PROPOSAL_REQUIRES_SEMANTIC_AND_MASK_REVIEW',
                    'training_eligible': False, 'measurements': proposals}
                pilot.atomic_json(destination / 'metadata.json', row, immutable=True)
                pilot.atomic_json(destination / 'COMMITTED.json', {'binding_sha256': sha256(OUT / 'binding.json'),
                    'files': {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}}, immutable=True)
                rows.append(row)
                print(json.dumps(row), flush=True)
        for page in range(3):
            sheet = Image.new('RGB', (1200, 1280), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, record in enumerate(manifest['records'][page*4:(page+1)*4]):
                y = index * 320
                draw.text((5, y + 2), record['id'] + ' | CYAN IS SAM PROPOSAL, NOT VERIFIED GT', fill='black')
                paths = [OUT / 'samples' / record['id'] / 'reference_overlay.png',
                    RUN / 'samples' / record['id'] / 'output_overlay.png',
                    OUT / 'samples' / record['id'] / 'output_overlay.png']
                meta = pilot.read(RUN / 'samples' / record['id'] / 'metadata.json')
                cy1, cy2, cx1, cx2 = meta['crop_yyxx']
                for col, path in enumerate(paths):
                    with Image.open(path) as src:
                        img = src.convert('RGB')
                    if col:
                        img = img.crop((cx1, cy1, cx2, cy2))
                    img.thumbnail((390, 280))
                    sheet.paste(img, (col * 400 + (400-img.width)//2, y + 28))
            with (OUT / f'review_{page+1}.png').open('xb') as stream:
                sheet.save(stream, format='PNG')
        pilot.atomic_json(OUT / 'summary.json', {'state': 'PROPOSALS_COMPLETE_NOT_GT', 'count': len(rows),
            'training_eligible': 0, 'elapsed_seconds': time.monotonic()-started,
            'reference_calibration_mean_iou': float(np.mean([r['measurements']['reference']['iou_to_reference_gt'] for r in rows])),
            'calibration_caveat': '12 diagnostic source crops only; not held-out accuracy or generated-mask GT.',
            'records': rows}, immutable=True)


if __name__ == '__main__':
    main()
