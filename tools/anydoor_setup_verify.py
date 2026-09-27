"""Record environment provenance and verify the offline-copied wheel payload."""
import base64
import csv
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capstone_lab.campaign.io import atomic_json
from anydoor_runtime import sha256, activate, verify_attention, SOURCE_REVISION, OFFICIAL_SHA256


def main():
    import torch
    import torchvision
    out = ROOT / 'artifacts/s8_anydoor_setup/run_v1'
    payloads = {}
    for package in ('torch', 'torchvision'):
        distribution = metadata.distribution(package)
        checked = 0
        for row in csv.reader(distribution.read_text('RECORD').splitlines()):
            name, digest, size = row
            name = name.replace(chr(92), '/')
            if not digest or not name.startswith((package + '/', 'torchgen/')):
                continue
            path = Path(distribution.locate_file(name))
            algorithm, encoded = digest.split('=', 1)
            if algorithm != 'sha256':
                raise ValueError('Unknown wheel digest algorithm')
            expected = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).hex()
            if path.stat().st_size != int(size) or sha256(path) != expected:
                raise ValueError(f'Copied wheel content mismatch: {path}')
            checked += 1
        if checked == 0:
            raise ValueError(f'No wheel payload hashes verified: {package}')
        payloads[package] = {'version': distribution.version, 'verified_record_files': checked,
            'offline_copy_source': '.venv-s5/Lib/site-packages', 'source_environment_modified': False}
    activate(out / 'source' / ('AnyDoor-' + SOURCE_REVISION))
    from cldm.cldm import ControlLDM
    from ldm.modules.encoders.modules import FrozenDinoV2Encoder
    matrix = torch.randn(64, 64, device='cuda')
    result = matrix @ matrix
    torch.cuda.synchronize()
    if not bool(result.isfinite().all()):
        raise ValueError('CUDA arithmetic failed')
    packages = {dist.metadata['Name']: dist.version for dist in metadata.distributions()}
    report = {'status': 'VERIFIED_ENVIRONMENT_NOT_MODEL_INFERENCE', 'python': sys.version,
        'executable': sys.executable, 'torch_path': torch.__file__, 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'attention': verify_attention(),
        'copied_wheel_verification': payloads, 'packages': packages,
        'source_revision': SOURCE_REVISION, 'source_archive_sha256': sha256(out / 'official_source.zip'),
        'official_checkpoint_expected_sha256': OFFICIAL_SHA256,
        'source_license': 'MIT', 'official_weight_space_card_license': 'Apache-2.0',
        'upstream_component_licenses': 'Bundled dinov2 LICENSE/MODEL_CARD declares CC-BY-NC4.0; '
            'preserve attribution/noncommercial restrictions. This capstone research setup '
            'does not authorize commercial use or redistribution.',
        'compatibility': 'timm0.6.12 dataclass defaults fail Python3.11; replaced by0.9.16. '
                         'No xformers; inference adapter uses tested PyTorch SDPA.',
        'actual_model_inference': 'PENDING_WEIGHTS_AND_SMOKE'}
    atomic_json(out / 'environment_verified.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'packages'}, indent=2))


if __name__ == '__main__':
    main()
