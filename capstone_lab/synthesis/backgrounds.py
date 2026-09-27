"""Immutable extraction and decode audit of the original FULL background archive."""
from pathlib import Path, PurePosixPath
import hashlib
import zipfile
import os
import shutil
import json

from capstone_lab.config import sha256_file
from capstone_lab.campaign.io import atomic_json, atomic_replace, RunLock
from capstone_lab.campaign.contracts import read_json

DOMAINS = {'snow', 'forest_dense', 'grass_field', 'leaf_ground', 'rocky', 'mixed'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp'}


def prepare(root, archive):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    out = root / 'artifacts/s8_full_backgrounds/v1'
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out / 'extract.lock'):
        archive_hash = sha256_file(archive)
        records, excluded, invalid = [], [], []
        with zipfile.ZipFile(archive) as z:
            members = sorted(z.infolist(), key=lambda i: i.filename)
            for i, member in enumerate(members):
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename)
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('unsafe archive member')
                if name.suffix.lower() not in EXTENSIONS:
                    excluded.append(member.filename)
                    continue
                domain = [p for p in name.parts[:-1] if p in DOMAINS]
                if len(domain) != 1:
                    raise ValueError('ambiguous background domain: ' + member.filename)
                if shutil.disk_usage(out).free < (21 * 2**30 + member.file_size):
                    raise ValueError('insufficient disk headroom for source extraction')
                content = z.read(member)  # ZipFile checks CRC; never execute archive contents.
                h = hashlib.sha256(content).hexdigest()
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    invalid.append({'member': member.filename, 'reason': 'decode_failed'})
                    continue
                dest = out / 'images' / domain[0] / (hashlib.sha256(member.filename.encode()).hexdigest()[:20] + name.suffix.lower())
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    if sha256_file(dest) != h:
                        raise ValueError('existing extracted source changed')
                else:
                    tmp = dest.with_suffix(dest.suffix + '.partial')
                    with tmp.open('wb') as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    atomic_replace(tmp, dest)
                records.append({'background_id': member.filename, 'domain': domain[0],
                                'environment': 'snow' if domain[0] == 'snow' else 'non_snow',
                                'path': dest.relative_to(root).as_posix(), 'sha256': h,
                                'bytes': len(content), 'width': image.shape[1], 'height': image.shape[0]})
                if len(records) % 100 == 0:
                    print('decoded backgrounds', len(records), flush=True)
        if sha256_file(archive) != archive_hash:
            raise ValueError('archive changed during extraction')
        if {r['domain'] for r in records} != DOMAINS:
            raise ValueError('missing FULL background domain')
        payload = {'status': 'VERIFIED' if not invalid else 'BLOCKED_DECODE_FAILURE',
                   'archive_sha256': archive_hash, 'archive_bytes': archive.stat().st_size,
                   'backgrounds': records, 'excluded_unsupported': excluded, 'invalid': invalid}
        atomic_json(out / 'manifest.json', payload, immutable=True)
        return {'status': payload['status'], 'backgrounds': len(records), 'invalid': invalid,
                'manifest': str(out / 'manifest.json')}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(Path.cwd().resolve(), a.archive.resolve()), indent=2))
