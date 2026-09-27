"""Explicit new generative release. Historical mixed manifests stay unchanged."""
from .contracts import read_json, inside, digest, code_records
from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError

BASE = 'artifacts/s8_anydoor_campaign/run_v1'
EXPERIMENTS = {'M2': ('real748', 'train748', 'random'), 'M3': ('real748', 'train748', 'task'),
               'L2': ('low187', 'low187', 'random'), 'L3': ('low187', 'low187', 'task')}


def binding(root, experiment):
    real, subset, selection = EXPERIMENTS[experiment]
    path = root / BASE / 'selection' / (experiment + '.json')
    approval = root / 'configs/approvals/s8_anydoor_shape_campaign_v1.json'
    return {'manifest': path.relative_to(root).as_posix(), 'sha256': sha256_file(path),
            'approval_sha256': sha256_file(approval), 'mixture': [real, subset, selection, 'H2', 150],
            'sampling': 'one visit per image per epoch; ordinary shuffle; no reweighting',
            'annotation': 'user-accepted approximate placement mask; SAM2 not used'}


def records(root, experiment, smoke=False):
    from .data import manifest_records
    info = binding(root, experiment)
    rows = read_json(root / info['manifest'])
    if len(rows) != 3000 or len({r['image'] for r in rows}) != 3000:
        raise ArtifactError('Generative manifest must contain3000 unique samples')
    for row in rows:
        if row['subset'] != EXPERIMENTS[experiment][1] or row['original_split'] != 'train':
            raise ArtifactError('Generative source policy mismatch')
        for key in ('image', 'label', 'mask'):
            path = inside(root, row[key])
            if not path.is_relative_to(root / BASE / 'pools' / row['subset']):
                raise ArtifactError('Unapproved generative source path')
            if sha256_file(path) != row[key + '_sha256']:
                raise ArtifactError('Generative artifact changed')
    real = manifest_records(root, EXPERIMENTS[experiment][0])
    if smoke:
        # Whole epochs on a declared32-image diagnostic mixture, NOT formal epochs.
        real.sort(key=lambda r: len((root / r['label']).read_text().splitlines()), reverse=True)
        rows.sort(key=lambda r: len((root / r['label']).read_text().splitlines()), reverse=True)
        return real[:16] + rows[:16]
    return real + rows


def require_release(root):
    gate = read_json(root / BASE / 'training_release.json')
    if gate.get('status') != 'VERIFIED_GENERATIVE_RELEASE' or gate.get('code_sha256') != digest(code_records(root)):
        raise ArtifactError('Generative training has no matching verified release')
    for file, expected in gate['evidence'].items():
        if sha256_file(inside(root, file)) != expected:
            raise ArtifactError('Generative release evidence changed')
