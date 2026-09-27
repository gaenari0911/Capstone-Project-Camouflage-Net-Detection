"""Approved heuristic mixtures: each real/synthetic image once per epoch."""
from pathlib import Path
from .contracts import read_json, inside, digest
from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError

MIXTURES = {**{'A'+str(i):('real748','train748','A'+str(i),'H2',40) for i in range(6)},
            'M1':('real748','train748','A1','H2',150),
            'L1':('low187','low187','A1','H2',150),
            'H0_S':('real748','train748','A1','H0',150),
            'H1_S':('real748','train748','A1','H1',150)}
AUDIT = Path('artifacts/s8_heuristic_audit/run_v1')


def records(root, experiment):
    from .data import manifest_records
    real, subset, condition, _, _ = MIXTURES[experiment]
    manifest = root/AUDIT/(subset+'_'+condition+'_manifest.json')
    rows = read_json(manifest)
    if len(rows)!=3000 or len({r['image'] for r in rows})!=3000:
        raise ArtifactError('Invalid synthetic manifest count')
    for row in rows:
        if row['subset']!=subset or row['condition']!=condition or row['original_split']!='train':
            raise ArtifactError('Mixed source policy mismatch')
        directory = inside(root,row['image']).parent
        expected = root/'artifacts/s8_heuristic_formal/run_v1/datasets'/subset/'samples'
        if not directory.is_relative_to(expected) or directory.name!=condition:
            raise ArtifactError('Unexpected synthetic source path')
        for key, filename in [('image','image.png'),('label','polygon.txt'),('mask','mask.png')]:
            path=inside(root,row[key])
            if path!=directory/filename or sha256_file(path)!=row[key+'_sha256']:
                raise ArtifactError('Mixed artifact mismatch: '+str(path))
    return manifest_records(root,real)+rows


def binding(root, experiment):
    _, subset, condition, _, _ = MIXTURES[experiment]
    path = root/AUDIT/(subset+'_'+condition+'_manifest.json')
    return {'manifest':path.relative_to(root).as_posix(), 'sha256':sha256_file(path),
            'sampling':'one visit per image per epoch; ordinary shuffle; no reweighting',
            'mixture':list(MIXTURES[experiment])}
