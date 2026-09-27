"""Steps6-9 sequential stage engine. Explicitly preserves historical runs."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capstone_lab.campaign.contracts import read_json, inside, digest, code_records
from capstone_lab.campaign.io import RunLock, atomic_json
from capstone_lab.campaign.detached import spawn, process_in_job
from capstone_lab.config import sha256_file

BASE = ROOT / 'artifacts/s8_anydoor_campaign/run_v1'
APPROVAL = ROOT / 'configs/approvals/s8_anydoor_shape_campaign_v1.json'
TRAIN_PYTHON = ROOT / '.venv-s5/python.exe'
GEN_PYTHON = ROOT / '.venv-s8-anydoor/Scripts/python.exe'


def stages():
    result = ['pilot', 'generate_train748', 'generate_low187', 'audit', 'score_train748', 'score_low187', 'selection', 'train_preflight']
    result += [f'train_{exp}_seed{seed}' for exp in ('M2', 'M3', 'L2', 'L3') for seed in (0, 1, 2)]
    return result + ['freeze', 'test', 'report']


def freeze_judge():
    directory=ROOT/'artifacts/s8_real_campaign/run_v1/jobs/L0_seed0/training_state'
    result=read_json(directory/'result.json');contract=read_json(directory/'contract.json')
    pointer=read_json(directory/'current.json')
    if (result.get('status')!='SUCCEEDED' or result.get('epochs')!=150
        or result.get('experiment')!='L0' or result.get('seed')!=0
        or contract.get('training_subset')!='low187'
        or result['contract_sha256']!=digest(contract) or result['best_mask']!=pointer['best_mask']):
        raise ValueError('Invalid completed low187 Judge contract')
    best=result['best_mask'];checkpoint=inside(directory,best['file'])
    if sha256_file(checkpoint)!=best['sha256']:raise ValueError('Judge checkpoint changed')
    chosen=max(result['history'],key=lambda r:r['mask_map50_95'])
    if chosen['epoch']!=best['epoch'] or chosen['mask_map50_95']!=best['score']:
        raise ValueError('Saved best does not match earliest maximum Val score')
    atomic_json(BASE/'judge_manifest.json',{'status':'FROZEN','checkpoint':checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256':best['sha256'],'best_epoch':best['epoch'],'source':'preapproved L0_seed0 best_mask',
        'training_subset':'low187','low187_sha256':sha256_file(ROOT/'manifests/low187_v1.jsonl'),
        'contract_sha256':result['contract_sha256'],'result_sha256':sha256_file(directory/'result.json'),
        'selection_uses_Test':False,'learner_initialization_from_judge':False},immutable=True)


def source_binding():
    files = code_records(ROOT)
    for path in sorted((ROOT / 'tools').glob('*.py')):
        if 'anydoor' in path.name or path.name == 's8_validation_snapshot.py':
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    files[APPROVAL.relative_to(ROOT).as_posix()] = sha256_file(APPROVAL)
    source=ROOT/'artifacts/s8_anydoor_setup/run_v1/source/AnyDoor-44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5'
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in {'.py','.yaml'}:
            files[path.relative_to(ROOT).as_posix()]=sha256_file(path)
    for file in ('artifacts/s8_anydoor_campaign/run_v1/judge_manifest.json',
                 'manifests/low187_v1.jsonl', 'manifests/real_split_observed_v1.jsonl',
                 'artifacts/s8_source_audit/split_pools_v1/train748_object_provenance.json',
                 'artifacts/s8_source_audit/split_pools_v1/low187_object_provenance.json',
                 'artifacts/s8_full_backgrounds/v1/manifest.json',
                 'artifacts/s8_anydoor_shape_review/run_v1/agent_review.json'):
        files[file] = sha256_file(ROOT / file)
    return files


def verify_binding():
    binding = read_json(BASE / 'binding.json')
    for file, expected in binding.items():
        if sha256_file(inside(ROOT, file)) != expected:
            raise ValueError('Frozen pipeline input/code changed: ' + file)
    return digest(binding)


def prepare():
    approval = read_json(APPROVAL)
    if approval['sam2'] or not approval['shape_control']:
        raise ValueError('Unexpected generation method')
    BASE.mkdir(parents=True, exist_ok=True)
    freeze_judge()
    atomic_json(BASE / 'binding.json', source_binding(), immutable=True)
    atomic_json(BASE / 'plan.json', {'stages': stages(), 'training_jobs': 12, 'epochs': 150,
        'candidate_images': 12000, 'pilot_separate': 100, 'sam2': False,
        'active_time_ceiling': None, 'gpu_concurrency': 1,
        'disk_budget_gib': 100, 'minimum_free_gib': 20,
        'approval_sha256': sha256_file(APPROVAL), 'per_job_approval': False,
        'technical_release': 'Must pass tests/preflight before start; runtime dataset preflight before training',
        'status': 'PREPARED_NOT_STARTED'}, immutable=True)
    return {'status': 'PREPARED_NOT_STARTED', 'path': str(BASE), 'stages': stages()}


def mark_done(stage, files):
    atomic_json(BASE / 'done' / (stage + '.json'), {'stage': stage, 'status': 'SUCCEEDED',
        'binding': verify_binding(), 'files': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}}, immutable=True)


def is_done(stage):
    path = BASE / 'done' / (stage + '.json')
    if not path.exists():
        return False
    row = read_json(path)
    if row['stage'] != stage or row['status'] != 'SUCCEEDED' or row['binding'] != verify_binding() or not row['files']:
        raise ValueError('Invalid stage completion')
    for file, expected in row['files'].items():
        if sha256_file(inside(ROOT, file)) != expected:
            raise ValueError('Completed stage output changed: ' + file)
    return True


def resource_guard():
    # No168hour ceiling. Keep previously approved capacity protection.
    from anydoor_pilot import budget
    return budget()


def work(stage):
    if stage not in stages():
        raise ValueError('Unknown stage')
    with RunLock(BASE / 'worker.lock'):
        verify_binding()
        index = stages().index(stage)
        if any(not is_done(previous) for previous in stages()[:index]):
            raise ValueError('Unfinished dependency')
        if is_done(stage):
            return
        resource_guard()
        from s8_anydoor_stages import dispatch
        files = dispatch(stage)
        mark_done(stage, files)


def owner_alive(owner):
    import psutil
    try:
        process = psutil.Process(owner['pid'])
        return (abs(process.create_time() - owner['created']) < .001
            and str(Path(__file__).resolve()) in process.cmdline()
            and owner['stage'] in process.cmdline() and process.is_running())
    except psutil.NoSuchProcess:
        return False


def serve():
    import psutil
    with RunLock(BASE / 'supervisor.lock'):
        verify_binding()
        for stage in stages():
            if is_done(stage):
                continue
            for attempt in range(1, 4):
                if (BASE / 'pause.request').exists():
                    atomic_json(BASE / 'status.json', {'status': 'PAUSED', 'stage': stage})
                    return
                worker_record = BASE / 'worker_owner.json'
                child = None
                if worker_record.exists() and owner_alive(read_json(worker_record)):
                    owner = read_json(worker_record)
                    if owner['stage'] != stage:
                        raise ValueError('Different live stage owns campaign')
                else:
                    log = BASE / 'logs' / (stage + '_' + uuid.uuid4().hex)
                    log.mkdir(parents=True)
                    python = GEN_PYTHON if stage == 'pilot' or stage.startswith('generate_') else TRAIN_PYTHON
                    command = [str(python), '-u', '-X', 'faulthandler', str(Path(__file__).resolve()), 'work', '--stage', stage]
                    with (log / 'stdout.log').open('xb') as stdout, (log / 'stderr.log').open('xb') as stderr:
                        child = spawn(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONFAULTHANDLER': '1'})
                    owner = {'pid': child.pid, 'created': psutil.Process(child.pid).create_time(),
                             'stage': stage, 'log': str(log), 'attempt': attempt}
                    atomic_json(worker_record, owner)
                while owner_alive(owner):
                    atomic_json(BASE / 'status.json', {'status': 'RUNNING', 'stage': stage, 'owner': owner,
                        'completed_stages': stages().index(stage), 'total_stages': len(stages()), 'updated': time.time()})
                    time.sleep(5)
                    if child is not None and child.poll() is not None:
                        break
                if is_done(stage):
                    break
                error_file = BASE / 'errors' / (stage + '.json')
                error = read_json(error_file) if error_file.exists() else {}
                if error.get('updated',0)<owner['created']:
                    error={'retryable': True, 'error': 'Worker disappeared without completion; logs/partial outputs retained.'}
                if not error.get('retryable', False) or attempt == 3:
                    atomic_json(BASE / 'status.json', {'status': 'BLOCKED', 'stage': stage, 'error': error,
                        'action': 'Fix cause; do not edit frozen code/inputs or delete completed data. No downstream stage launched.'})
                    return
                time.sleep(10 * attempt)
            else:
                return
        atomic_json(BASE / 'status.json', {'status': 'SUCCEEDED', 'completed_stages': len(stages()), 'updated': time.time()})


def watchdog():
    with RunLock(BASE / 'watchdog.lock'):
        for restart in range(4):
            status = read_json(BASE / 'status.json') if (BASE / 'status.json').exists() else {}
            if status.get('status') in {'SUCCEEDED', 'PAUSED', 'BLOCKED'}:
                return
            log = BASE / 'watchdog_logs' / uuid.uuid4().hex
            log.mkdir(parents=True)
            command = [str(TRAIN_PYTHON), '-u', '-X', 'faulthandler', str(Path(__file__).resolve()), 'serve']
            with (log / 'stdout.log').open('xb') as stdout, (log / 'stderr.log').open('xb') as stderr:
                child = spawn(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            while child.poll() is None:
                time.sleep(5)
            status = read_json(BASE / 'status.json') if (BASE / 'status.json').exists() else {}
            if status.get('status') in {'SUCCEEDED', 'PAUSED', 'BLOCKED'}:
                return
            time.sleep(10)
        atomic_json(BASE / 'status.json', {'status': 'BLOCKED', 'error': 'Supervisor repeatedly exited; durable worker ownership retained.'})


def start():
    verify_binding()
    release = read_json(BASE / 'implementation_release.json')
    if release.get('status') != 'VERIFIED' or release['binding'] != verify_binding():
        raise ValueError('Implementation not verified; refusing formal launch')
    for file, expected in release['evidence'].items():
        if sha256_file(inside(ROOT, file)) != expected:
            raise ValueError('Preflight evidence changed')
    with RunLock(BASE / 'launch.lock'):
        with RunLock(BASE / 'watchdog.lock'):
            pass
        # This launches but never kills existing workers. They are adopted by ownership.
        atomic_json(BASE / 'status.json', {'status': 'STARTING', 'updated': time.time()})
        log = BASE / 'launches' / uuid.uuid4().hex
        log.mkdir(parents=True)
        with (log / 'stdout.log').open('xb') as stdout, (log / 'stderr.log').open('xb') as stderr:
            child = spawn([str(TRAIN_PYTHON), '-u', '-X', 'faulthandler', str(Path(__file__).resolve()), 'watchdog'],
                          cwd=ROOT, stdout=stdout, stderr=stderr)
        time.sleep(1)
        if child.poll() is not None or process_in_job(child.pid):
            raise RuntimeError('Detached watchdog handshake failed; inspect logs before retry')
        return {'status': 'STARTED', 'pid': child.pid, 'in_job': False, 'logs': str(log), 'reboot_autostart': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'start', 'serve', 'watchdog', 'work', 'status', 'check'])
    parser.add_argument('--stage')
    args = parser.parse_args()
    if args.action == 'prepare': result = prepare()
    elif args.action == 'start': result = start()
    elif args.action == 'serve': result = serve()
    elif args.action == 'watchdog': result = watchdog()
    elif args.action == 'work':
        try:
            result = work(args.stage)
        except Exception as exc:
            atomic_json(BASE / 'errors' / (str(args.stage) + '.json'), {'error': traceback.format_exc(),
                'retryable': isinstance(exc, (PermissionError, TimeoutError, ConnectionError)), 'updated': time.time()})
            raise
    elif args.action == 'check': result = {'binding': verify_binding(), 'stages': stages(), 'no_jobs_started': True}
    else: result = read_json(BASE / 'status.json') if (BASE / 'status.json').exists() else {'status': 'NOT_STARTED'}
    if result is not None: print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
