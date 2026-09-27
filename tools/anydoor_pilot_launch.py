"""Explicitly detached pilot launcher; does not auto-approve formal generation."""
import argparse
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capstone_lab.campaign.io import RunLock, atomic_json
from capstone_lab.campaign.detached import spawn, process_in_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='artifacts/s8_anydoor_pilot/run_v1')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--parity-run')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        raise ValueError('Pilot only: 1..100 samples')
    run = (ROOT / args.run).resolve()
    run.relative_to(ROOT / 'artifacts')
    if not (run / 'manifest.json').exists():
        raise ValueError('Prepare and verify inputs before launching')
    with RunLock(run / 'launch.lock'):
        # An actual live worker, not a stale PID file, owns this OS-held lock.
        with RunLock(run / 'run.lock'):
            pass
        log = run / 'launches' / uuid.uuid4().hex
        log.mkdir(parents=True)
        command = [str(ROOT / '.venv-s8-anydoor/Scripts/python.exe'), '-u',
                   str(ROOT / 'tools/anydoor_pilot.py'), 'run', '--run', str(run), '--limit', str(args.limit)]
        if args.parity_run:
            command.extend(['--parity-run', args.parity_run])
        with (log / 'stdout.log').open('xb') as stdout, (log / 'stderr.log').open('xb') as stderr:
            child = spawn(command, cwd=str(ROOT), stdout=stdout, stderr=stderr)
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError(f'Pilot exited early; inspect {log}')
        in_job = process_in_job(child.pid)
        if in_job:
            child.terminate()
            raise RuntimeError('Refusing session-owned worker')
        record = {'pid': child.pid, 'in_job': in_job, 'logs': str(log), 'command': command,
                  'reboot_autostart': False, 'automatic_quality_approval': False}
        atomic_json(log / 'launch.json', record, immutable=True)
        atomic_json(run / 'latest_launch.json', record)
        print(record)


if __name__ == '__main__':
    main()
