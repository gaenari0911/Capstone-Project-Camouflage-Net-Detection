"""Evidence-bound, restartable launcher for the approved Heuristic21000 campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid

import psutil

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError, StateError
from capstone_lab.campaign.contracts import digest, inside, read_json
from capstone_lab.campaign.detached import job_info, process_in_job, spawn
from capstone_lab.campaign.io import RunLock, atomic_json, atomic_replace
from capstone_lab.campaign.resources import budget_roots
from capstone_lab.campaign.supervisor import directory_bytes
from .full import execute, memory


RELEASE = Path('artifacts/s8_heuristic_release/run_v1/summary.json')
CAMPAIGN = Path('artifacts/s8_heuristic_release/run_v1/formal_campaign.json')
FORMAL_ROOT = Path('artifacts/s8_heuristic_formal/run_v1')
BASELINE = Path('artifacts/s8_heuristic_verification/run_v2/summary.json')
REAL_STATUS = Path('artifacts/s8_real_campaign/run_v1/status.json')
MAX_ACTIVE_SECONDS = 604800.0
MAX_NEW_BYTES = 100 * 2**30
MIN_FREE_BYTES = 20 * 2**30
TERMINAL = {'SUCCEEDED', 'PAUSED', 'BLOCKED_RESOURCES', 'BLOCKED_INTEGRITY',
            'PARTIAL_FAILURE'}


def _require(value, message):
    if not value:
        raise ArtifactError(message)


def _record(pid, token, module):
    process = psutil.Process(pid)
    command = process.cmdline()
    _require(token in command and module in command, 'process ownership command mismatch')
    _require(not process_in_job(pid), 'process remained in Windows session job')
    return {'pid': pid, 'created': process.create_time(), 'token': token,
            'module': module, 'command': command, 'in_job': process_in_job(pid)}


def _owned(record):
    try:
        process = psutil.Process(record['pid'])
        if abs(process.create_time() - record['created']) > .001:
            return None
        command = process.cmdline()
        if record['token'] not in command or record['module'] not in command:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.NoSuchProcess, KeyError):
        return None
    except psutil.AccessDenied as exc:
        raise StateError('cannot prove process ownership') from exc


def _runtime_files(root):
    files = sorted((root/'capstone_lab').rglob('*.py'))
    files.append(root/'build_demo_synthetic_segmentation.py')
    return files


def _input_hashes(root):
    paths = [
        'configs/campaigns/s8_heuristic_policy_v1.json',
        'configs/approvals/s8_foreground_amendment_v1.json',
        'configs/approvals/s8_low_diversity_v1.json',
        'artifacts/s8_source_audit/split_pools_v1/split_source_summary.json',
        'artifacts/s8_source_audit/split_pools_v1/train748_object_provenance.json',
        'artifacts/s8_source_audit/split_pools_v1/low187_object_provenance.json',
        'artifacts/s8_full_backgrounds/v1/manifest.json',
        'manifests/real_split_observed_v1.jsonl',
        'manifests/low187_v1.jsonl',
        'manifests/object_environment_resolution_v1.jsonl',
    ]
    return {p: sha256_file(root/p) for p in paths}


def _environment():
    import cv2
    import numpy
    return {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
            'executable_sha256': sha256_file(Path(sys.executable)),
            'numpy': numpy.__version__, 'opencv': cv2.__version__,
            'platform': platform.platform()}


def _verify_baseline(root):
    evidence = read_json(root/BASELINE)
    _require(evidence['status'] == 'PARTIALLY_VERIFIED_NOT_FORMAL_RELEASE',
             'baseline evidence status changed')
    _require(not evidence['formal_generation_started'] and
             not evidence['training_or_inference_started'], 'baseline scope changed')
    allowed_changes = {'capstone_lab/synthesis/full.py','capstone_lab/synthesis/smoke.py'}
    mismatches = []
    for relative, expected in evidence['input_code_hashes'].items():
        observed = sha256_file(root/relative)
        if observed != expected and relative not in allowed_changes:
            mismatches.append(relative)
    _require(not mismatches, 'unreviewed baseline changes: '+','.join(mismatches))
    for subset in ('train748', 'low187'):
        pair = evidence['pairs'][subset]
        _require(pair['worker_output_hashes_equal'], 'baseline worker hash mismatch')
        for run, report_hash in zip(pair['runs'], pair['report_hashes']):
            _require(sha256_file(root/run/'report.json') == report_hash,
                     'baseline report changed: '+run)
    return evidence


def _bootstrap_text():
    return ("import runpy,sys\n"
            "if __name__ == '__main__':\n"
            "    module=sys.argv.pop(1)\n"
            "    runpy.run_module(module,run_name='__main__')\n")


def freeze_release(root, output):
    """Run acceptance tests and atomically freeze code before permitting production."""
    root, output = root.resolve(), output.resolve()
    _require(output == (root/RELEASE).parent.resolve(), 'release path is fixed')
    if output.exists():
        raise ArtifactError('release output already exists; never overwrite a release')
    baseline = _verify_baseline(root)
    from .acceptance import verify_acceptance
    acceptance = verify_acceptance(root)
    before = {p.relative_to(root).as_posix():sha256_file(p) for p in _runtime_files(root)}
    inputs_before = _input_hashes(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent/('attempt_'+uuid.uuid4().hex)
    stage.mkdir()
    logs = stage/'acceptance.log'
    modules = ['tests.test_s8_formal_synthesis','tests.test_s8_full_synthesis',
               'tests.test_s8_source_policy','tests.test_s3_synthesis',
               'tests.test_s4_smoke','tests.test_s8_detached','tests.test_s8_campaign',
               'tests.test_s8_seed_policy']
    command = [sys.executable, '-m', 'unittest', *modules, '-v']
    with logs.open('xb') as stream:
        result = subprocess.run(command, cwd=root, stdout=stream,
                                stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    _require(result.returncode == 0, 'formal acceptance tests failed; inspect acceptance.log')
    _require(before == {p.relative_to(root).as_posix():sha256_file(p) for p in _runtime_files(root)}
             and inputs_before == _input_hashes(root), 'code/input changed during acceptance')
    pending = stage/('.runtime_pending_'+uuid.uuid4().hex)
    pending.mkdir()
    for source in _runtime_files(root):
        relative = source.relative_to(root)
        destination = pending/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (pending/'bootstrap.py').write_text(_bootstrap_text(), encoding='utf-8', newline='\n')
    runtime_records = {p.relative_to(pending).as_posix(): sha256_file(p)
                       for p in sorted(pending.rglob('*')) if p.is_file()}
    runtime_manifest = {'schema_version': 1, 'files': runtime_records,
                        'code_sha256': digest(runtime_records)}
    atomic_json(pending/'runtime_manifest.json', runtime_manifest, immutable=True)
    runtime_records['runtime_manifest.json'] = sha256_file(pending/'runtime_manifest.json')
    atomic_replace(pending, stage/'runtime')
    real = read_json(root/REAL_STATUS)
    _require(real['status'] == 'SUCCEEDED' and real['counts'].get('SUCCEEDED') == 8,
             'completed Real-only campaign evidence missing')
    body = {
        'schema_version': 1,
        'status': 'VERIFIED_FORMAL_GENERATOR',
        'baseline_summary_sha256': sha256_file(root/BASELINE),
        'baseline_contract_hashes': {k:v['contract_hashes'] for k,v in baseline['pairs'].items()},
        'acceptance_log_sha256': sha256_file(logs),
        'acceptance_modules': modules,
        'runtime_acceptance': acceptance,
        'runtime_manifest_sha256': sha256_file(stage/'runtime/runtime_manifest.json'),
        'runtime_code_sha256': runtime_manifest['code_sha256'],
        'environment': _environment(),
        'profile_evidence': {
            p:sha256_file(root/p) for p in [
                'artifacts/s8_heuristic_preflight/formal_profile_w1_v1/report.json',
                'artifacts/s8_heuristic_preflight/formal_profile_w2_v1/report.json']},
        'input_hashes': _input_hashes(root),
        'prior_real_active_seconds': real['active_seconds'],
        'budgets': {'active_seconds_max': MAX_ACTIVE_SECONDS,
                    'new_bytes_max': MAX_NEW_BYTES, 'min_free_bytes': MIN_FREE_BYTES},
        'formal_counts': {'train748': {'plans':3000,'conditions':['A0','A1','A2','A3','A4','A5']},
                          'low187': {'plans':3000,'conditions':['A1']}},
        'workers': 2, 'cpu_heavy_max': 6, 'io_max': 2,
        'ssh_disconnect': 'NOT_PERFORMED', 'power_cut': 'NOT_PERFORMED',
        'mixed_training_loader': 'NOT_IMPLEMENTED_OR_VERIFIED',
    }
    config = {'schema_version':1, 'status':'APPROVED_FORMAL_HEURISTIC21000',
              'output':FORMAL_ROOT.as_posix(), 'release':RELEASE.as_posix(),
              'workers':2,
              'cpu_heavy_max':6, 'io_max':2,
              'jobs':[{'id':'train748_A0_A5_3000','subset':'train748','count':3000},
                      {'id':'low187_A1_3000','subset':'low187','count':3000}]}
    body['campaign_contract'] = dict(config)
    summary = {**body, 'release_sha256': digest(body)}
    atomic_json(stage/'summary.json', summary, immutable=True)
    config['release_sha256']=summary['release_sha256']
    atomic_json(stage/'formal_campaign.json', config, immutable=True)
    atomic_replace(stage, output)
    return summary


def verify_release(root, release_path=RELEASE):
    root = root.resolve()
    path = inside(root, release_path)
    release = read_json(path)
    body = {k:v for k,v in release.items() if k != 'release_sha256'}
    _require(release.get('status') == 'VERIFIED_FORMAL_GENERATOR' and
             release.get('release_sha256') == digest(body), 'release digest/status mismatch')
    _require(_environment() == release['environment'], 'Python/NumPy/OpenCV runtime changed')
    for relative, expected in release['input_hashes'].items():
        _require(sha256_file(root/relative) == expected, 'formal input changed: '+relative)
    runtime = path.parent/'runtime'
    manifest_path = runtime/'runtime_manifest.json'
    _require(sha256_file(manifest_path) == release['runtime_manifest_sha256'],
             'runtime manifest changed')
    _require(sha256_file(path.parent/'acceptance.log') == release['acceptance_log_sha256'],
             'acceptance log changed')
    manifest = read_json(manifest_path)
    records = dict(manifest['files'])
    _require(digest(records) == release['runtime_code_sha256'], 'runtime code digest changed')
    for relative, expected in records.items():
        _require(sha256_file(runtime/relative) == expected, 'runtime file changed: '+relative)
    return release, runtime


def _campaign(root):
    acceptance = os.environ.get('CAPSTONE_SYNTHESIS_ACCEPTANCE')
    if acceptance:
        path = Path(acceptance).resolve()
        _require(path.is_relative_to(root/'artifacts/s8_heuristic_preflight'), 'unsafe acceptance config')
        config = read_json(path)
        _require(config.get('mode') == 'BOUNDED_RUNTIME_ACCEPTANCE', 'invalid acceptance mode')
        _require(config['workers'] in (1,2), 'acceptance worker limit')
        _require([j['subset'] for j in config['jobs']] == ['train748','low187'] and
                 all(1 <= j['count'] <= 3 for j in config['jobs']), 'acceptance count/scope')
        run = inside(root, config['output'])
        _require(run.is_relative_to(path.parent) and run != path.parent, 'unsafe acceptance output')
        runtime = path.parent/'runtime'
        manifest = read_json(runtime/'runtime_manifest.json')
        _require(digest(manifest['files']) == config['runtime_code_sha256'], 'acceptance runtime changed')
        for relative, expected in manifest['files'].items():
            _require(sha256_file(inside(runtime, relative)) == expected, 'acceptance runtime bytes changed')
        for relative, expected in config['input_hashes'].items():
            _require(sha256_file(inside(root,relative)) == expected, 'acceptance input changed')
        release = {'status':'BOUNDED_RUNTIME_ACCEPTANCE','release_sha256':digest(config),
                   'prior_real_active_seconds':config['prior_real_active_seconds']}
        return config,release,runtime,run
    config = read_json(root/CAMPAIGN)
    _require(config['status'] == 'APPROVED_FORMAL_HEURISTIC21000', 'campaign not approved')
    release, runtime = verify_release(root, Path(config['release']))
    _require(config['release_sha256'] == release['release_sha256'], 'campaign release mismatch')
    _require({k:v for k,v in config.items() if k!='release_sha256'} == release['campaign_contract'],
             'campaign config changed')
    _require(config['workers'] <= config['cpu_heavy_max'] <= 6 and config['io_max'] <= 2,
             'campaign admission limits invalid')
    run = inside(root, config['output'])
    _require(run.resolve() == (root/FORMAL_ROOT).resolve(), 'formal output changed')
    return config, release, runtime, run


def _initial_state(config):
    return {'status':'PLANNED','jobs':{j['id']:{'state':'PLANNED','attempts':0}
                                      for j in config['jobs']},'updated':time.time()}


def _load_state(run, config):
    path=run/'campaign_state.json'
    state=read_json(path) if path.exists() else _initial_state(config)
    _require(set(state['jobs']) == {j['id'] for j in config['jobs']}, 'campaign state jobs changed')
    return state


def _ledger_tick(root, run, release, seconds=0.0, active=False):
    with RunLock(run/'ledger.lock'):
        path=run/'active_time_ledger.json'
        if path.exists():
            ledger=read_json(path)
            _require(ledger['release_sha256'] == release['release_sha256'], 'ledger release mismatch')
        else:
            ledger={'schema_version':1,'release_sha256':release['release_sha256'],
                    'prior_real_active_seconds':release['prior_real_active_seconds'],
                    'formal_active_seconds':0.0,'ticks':0}
        ledger['formal_active_seconds'] += max(0.0, seconds)
        ledger['ticks'] += 1
        ledger['active'] = active
        ledger['total_active_seconds'] = (ledger['prior_real_active_seconds']+
                                          ledger['formal_active_seconds'])
        ledger['remaining_active_seconds'] = max(0.0, MAX_ACTIVE_SECONDS-ledger['total_active_seconds'])
        ledger['updated'] = time.time()
        atomic_json(path,ledger)
        return ledger


class _Ticker:
    def __init__(self, root, run, release):
        self.root,self.run,self.release=root,run,release
        self.stop=threading.Event(); self.thread=None
    def __enter__(self):
        # Charge a safety window BEFORE dispatch. An abrupt death retains the
        # unused reservation, so restarts cannot erase unflushed active time.
        self.last=time.monotonic()
        self.reserved=600.0
        self.lease=uuid.uuid4().hex
        self.inputs=_input_hashes(self.root)
        self.next_input_check=0.0
        self._save(0,begin=True)
        return self
    def _save(self, elapsed, *, begin=False, close=False):
        with RunLock(self.run/'ledger.lock'):
            path=self.run/'active_time_ledger.json'
            ledger=read_json(path) if path.exists() else {
                'release_sha256':self.release['release_sha256'],
                'prior_real_active_seconds':self.release['prior_real_active_seconds'],
                'formal_active_seconds':0.0,'ticks':0}
            _require(ledger['release_sha256']==self.release['release_sha256'], 'ledger release mismatch')
            if not begin: _require(ledger.get('lease')==self.lease,'ledger lease changed')
            value=ledger['formal_active_seconds']+elapsed+(self.reserved if begin else 0)
            if close: value-=self.reserved
            _require(ledger['prior_real_active_seconds']+value <= MAX_ACTIVE_SECONDS,'TIME_BUDGET_STOP')
            ledger.update(formal_active_seconds=value,lease=self.lease,active=not close,
                          reserved_seconds=0 if close else self.reserved,
                          total_active_seconds=ledger['prior_real_active_seconds']+value,
                          updated=time.time(),ticks=ledger['ticks']+1)
            ledger['remaining_active_seconds']=MAX_ACTIVE_SECONDS-ledger['total_active_seconds']
            atomic_json(path,ledger)
    def check(self):
        now=time.monotonic()
        self._save(now-self.last)
        self.last=now
        if (self.run/'control.json').exists() and read_json(self.run/'control.json')['action']=='pause':
            raise InterruptedError('requested campaign pause')
        from .full import resource_guard, POLICY
        resource_guard(self.root,read_json(self.root/POLICY),2,disk=False)
        if now>=self.next_input_check:
            _require(_input_hashes(self.root)==self.inputs,'formal input changed during execution')
            self.next_input_check=now+10
    def __exit__(self,*_):
        self._save(time.monotonic()-self.last,close=True)


def _resource_state(root, run, release, workers):
    ledger=_ledger_tick(root,run,release)
    used=sum(directory_bytes(p) for p in budget_roots(root))
    free=shutil.disk_usage(root).free
    snapshot=memory()
    snapshot.update(new_bytes=used,free_bytes=free,
                    total_active_seconds=ledger['total_active_seconds'])
    _require(ledger['total_active_seconds'] < MAX_ACTIVE_SECONDS, 'TIME_BUDGET_STOP')
    _require(used < MAX_NEW_BYTES, 'DISK_BUDGET_STOP')
    _require(free >= MIN_FREE_BYTES, 'DISK_FREE_STOP')
    _require(workers <= 6, 'CPU_ADMISSION_STOP')
    return snapshot


def worker(root, token, job_id, attempt_id):
    config,release,runtime,run=_campaign(root)
    _require(Path(__file__).resolve().is_relative_to(runtime.resolve()), 'worker is not frozen runtime')
    job=next(j for j in config['jobs'] if j['id']==job_id)
    _require(read_json(run/'token.json')['token']==token, 'worker token mismatch')
    output=run/'datasets'/job['subset']
    attempt=inside(run,attempt_id)
    _require(attempt.is_relative_to(run/'jobs'/job_id), 'worker attempt escapes job')
    atomic_json(attempt/'owner.json',_record(os.getpid(),token,'capstone_lab.synthesis.formal'),immutable=True)
    deadline=time.monotonic()+30
    while not (attempt/'dispatch.json').exists():
        if time.monotonic()>deadline: raise StateError('worker dispatch handshake timeout')
        time.sleep(.1)
    _require(read_json(attempt/'dispatch.json')['owner']==read_json(attempt/'owner.json'),
             'worker dispatch owner mismatch')
    complete_status='FORMAL_DATASET_COMPLETE' if release['status']=='VERIFIED_FORMAL_GENERATOR' else 'VERIFIED_SMALL_RENDER'
    try:
        _resource_state(root,run,release,config['workers'])
        with RunLock(root/'artifacts/s8_formal_ledger/synthesis_heavy.lock'), _Ticker(root,run,release) as ticker:
            result=execute(root,output,job['subset'],config['workers'],job['count'],
                           formal_release=release if release['status']=='VERIFIED_FORMAL_GENERATOR' else None,
                           code_root=runtime,pause_path=run/'control.json',checkpoint=ticker.check)
        status=result['status']
        payload={'job_id':job_id,'status':status,'report':
                 (output/'report.json').relative_to(run).as_posix() if (output/'report.json').exists() else None,
                 'release_sha256':release['release_sha256'],'finished':time.time()}
        atomic_json(attempt/'result.json',payload,immutable=True)
        return 0 if status in {complete_status,'PAUSED'} else 2
    except InterruptedError as exc:
        atomic_json(attempt/'result.json',{'job_id':job_id,'status':'PAUSED','error':str(exc),
                    'release_sha256':release['release_sha256'],'finished':time.time()},immutable=True)
        return 0
    except Exception as exc:
        text=f'{type(exc).__name__}: {exc}'
        from concurrent.futures.process import BrokenProcessPool
        kind='RETRYABLE_FAILURE' if isinstance(exc,(BrokenProcessPool,TimeoutError)) else (
            'BLOCKED_RESOURCES' if any(x in text for x in ('BUDGET_STOP','FREE_STOP','SAFE_WAIT','ADMISSION_STOP'))
            else 'BLOCKED_INTEGRITY')
        atomic_json(attempt/'result.json',{'job_id':job_id,'status':kind,
                    'error':text,'release_sha256':release['release_sha256'],'finished':time.time()},immutable=True)
        raise


def _validate_complete(run, job, release, result_path=None):
    result=read_json(result_path or run/'jobs'/job['id']/'result.json')
    _require(result['release_sha256']==release['release_sha256'], 'job result release mismatch')
    expected_status='FORMAL_DATASET_COMPLETE' if release['status']=='VERIFIED_FORMAL_GENERATOR' else 'VERIFIED_SMALL_RENDER'
    if result['status']!=expected_status: return result['status']
    report=read_json(inside(run,result['report']))
    expected={c:job['count'] for c in ('A0','A1','A2','A3','A4','A5')} if job['subset']=='train748' else {'A1':job['count']}
    _require(report['status']==expected_status and
             report['counts']==expected, 'formal report incomplete')
    _require(len(report['output_hashes']) == sum(expected.values()), 'formal output index incomplete')
    names={'image':'image.png','mask':'mask.png','polygon':'polygon.txt','metadata':'metadata.json'}
    for key, hashes in report['output_hashes'].items():
        for name, expected_hash in hashes.items():
            _require(name in names and sha256_file(run/'datasets'/job['subset']/'samples'/key/names[name]) == expected_hash,
                     'formal output changed: '+key+'/'+name)
    return 'SUCCEEDED'

def cleanup_pool(output, parent):
    """Only terminate registered pool children of this exact dead worker."""
    removed=[]
    for path in (output/'.pool').glob('*.json'):
        record=read_json(path)
        if record['parent']!=parent['pid'] or record['created']<parent['created']: continue
        try:
            process=psutil.Process(record['pid'])
            if abs(process.create_time()-record['created'])>.001 or process.cmdline()!=record['command']: continue
            process.terminate()
            try: process.wait(timeout=15)
            except psutil.TimeoutExpired: raise StateError('registered pool child did not stop')
            removed.append(record)
        except psutil.NoSuchProcess: pass
    return removed


def supervise(root, token):
    config,release,runtime,run=_campaign(root)
    _require(Path(__file__).resolve().is_relative_to(runtime.resolve()), 'supervisor is not frozen runtime')
    with RunLock(run/'supervisor.lock'):
        atomic_json(run/'supervisor_owner.json',_record(os.getpid(),token,'capstone_lab.synthesis.formal'))
        state=_load_state(run,config)
        while True:
            # Never move to the next dataset until its predecessor is verified.
            jobs=[j for j in config['jobs'] if state['jobs'][j['id']]['state']!='SUCCEEDED']
            if not jobs:
                for completed in config['jobs']:
                    row=state['jobs'][completed['id']]
                    _require(_validate_complete(run,completed,release,inside(run,row['result']))=='SUCCEEDED',
                             'completed job changed')
                state['status']='SUCCEEDED'
                break
            job=jobs[0]
            row=state['jobs'][job['id']]
            handle=None
            attempt=inside(run,row['attempt_dir']) if row.get('attempt_dir') else None
            if attempt and (attempt/'owner.json').exists():
                row['owner']=read_json(attempt/'owner.json')
            live=_owned(row.get('owner',{})) if row.get('owner') else None
            if not live and row.get('owner'):
                cleanup_pool(run/'datasets'/job['subset'],row['owner'])
            if live and attempt and not (attempt/'dispatch.json').exists():
                atomic_json(attempt/'dispatch.json',{'owner':row['owner']},immutable=True)
            if not live and attempt and (attempt/'result.json').exists() and not row.get('handled'):
                observed=_validate_complete(run,job,release,attempt/'result.json')
                row['handled']=True
                if observed=='SUCCEEDED':
                    row.update(state='SUCCEEDED',result=(attempt/'result.json').relative_to(run).as_posix())
                    atomic_json(run/'campaign_state.json',state)
                    continue
                if observed=='PAUSED':
                    row['state']='PLANNED';state['status']='PAUSED';break
                if observed!='RETRYABLE_FAILURE':
                    row['state']=observed;state['status']=observed
                    state['error']=read_json(attempt/'result.json').get('error');break
            if not live and read_json(run/'control.json')['action']=='pause':
                state['status']='PAUSED';break
            if live is None:
                # A coordinator can die between spawn and the owner handshake.
                # Wait out that interval before retrying the durable intent.
                if row.get('state')=='STARTING' and time.time()-row.get('started',0)<35:
                    time.sleep(.2);continue
                if row['attempts']>=3:
                    state['status']='PARTIAL_FAILURE';break
                try: telemetry=_resource_state(root,run,release,config['workers'])
                except ArtifactError as exc:
                    state['status']='BLOCKED_RESOURCES';state['error']=str(exc);break
                number=row['attempts']+1
                directory=run/'jobs'/job['id']/('attempt_'+str(number).zfill(3))
                directory.mkdir(parents=True,exist_ok=True)
                attempt_id=directory.relative_to(run).as_posix()
                row.update(state='STARTING',attempts=number,owner=None,attempt_dir=attempt_id,
                           started=time.time(),handled=False)
                state['status']='RUNNING';state['telemetry']=telemetry
                atomic_json(run/'campaign_state.json',state)
                command=[sys.executable,'-u',str(runtime/'bootstrap.py'),'capstone_lab.synthesis.formal',
                         'worker','--project-root',str(root),'--token',token,'--job',job['id'],'--attempt',attempt_id]
                with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
                    child=subprocess.Popen(command,cwd=runtime,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,
                        PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                deadline=time.monotonic()+25
                while not (directory/'owner.json').exists():
                    if child.poll() is not None or time.monotonic()>deadline: break
                    time.sleep(.1)
                if not (directory/'owner.json').exists():
                    row['state']='PLANNED';atomic_json(run/'campaign_state.json',state);continue
                row.update(state='RUNNING',owner=read_json(directory/'owner.json'))
                atomic_json(run/'campaign_state.json',state)
                atomic_json(directory/'dispatch.json',{'owner':row['owner']},immutable=True)
                live=psutil.Process(child.pid)
                handle=child
            while True:
                try:
                    if not live.is_running() or live.status()==psutil.STATUS_ZOMBIE: break
                    atomic_json(run/'heartbeat.json',{'status':'RUNNING','job':job['id'],'worker_pid':live.pid,
                                'supervisor_pid':os.getpid(),'in_job':False,'updated':time.time(),
                                'telemetry':memory()})
                    progress=run/'datasets'/job['subset']/'progress.json'
                    last=max(progress.stat().st_mtime if progress.exists() else 0,row['started'])
                    if time.time()-last>550:
                        parent=_owned(row['owner'])
                        if parent:
                            children=[(p.pid,p.create_time()) for p in parent.children(recursive=True)]
                            parent.terminate()
                            for pid,created in children:
                                try:
                                    child_process=psutil.Process(pid)
                                    if abs(child_process.create_time()-created)<.001: child_process.terminate()
                                except psutil.NoSuchProcess: pass
                        atomic_json(inside(run,row['attempt_dir'])/'timeout.json',{'reason':'550seconds without progress'})
                        break
                    time.sleep(2)
                except psutil.NoSuchProcess:
                    break
            row['state']='PLANNED'
            cleaned=cleanup_pool(run/'datasets'/job['subset'],row['owner'])
            atomic_json(inside(run,row['attempt_dir'])/'exit.json',{
                'owner':row['owner'],'observed':time.time(),
                'exit_code':handle.poll() if handle is not None else 'UNKNOWN_ADOPTED_PROCESS_EXIT',
                'pool_children_cleaned':cleaned},immutable=True)
            atomic_json(run/'campaign_state.json',state)
        state['updated']=time.time();atomic_json(run/'campaign_state.json',state);return state


def watch(root, token):
    config,release,runtime,run=_campaign(root)
    _require(Path(__file__).resolve().is_relative_to(runtime.resolve()), 'watchdog is not frozen runtime')
    with RunLock(run/'watchdog.lock'):
        atomic_json(run/'watchdog_owner.json',_record(os.getpid(),token,'capstone_lab.synthesis.formal'))
        restarts=read_json(run/'watchdog_status.json').get('restarts',0) if (run/'watchdog_status.json').exists() else 0
        starting=None
        while True:
            state=_load_state(run,config)
            if state['status'] in TERMINAL:
                atomic_json(run/'watchdog_status.json',{'status':state['status'],'restarts':restarts,'updated':time.time()});return state
            owner=read_json(run/'supervisor_owner.json') if (run/'supervisor_owner.json').exists() else None
            if (owner is None or _owned(owner) is None) and (starting is None or starting.poll() is not None):
                if restarts>=8:
                    state['status']='PARTIAL_FAILURE';state['error']='coordinator restart bound exhausted'
                    atomic_json(run/'campaign_state.json',state);return state
                command=[sys.executable,'-u',str(runtime/'bootstrap.py'),'capstone_lab.synthesis.formal',
                         'supervise','--project-root',str(root),'--token',token]
                logs=run/'coordinator_launches'/uuid.uuid4().hex;logs.mkdir(parents=True)
                with (logs/'stdout.log').open('xb') as out,(logs/'stderr.log').open('xb') as err:
                    starting=subprocess.Popen(command,cwd=runtime,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1'))
                restarts+=1
            atomic_json(run/'watchdog_status.json',{'status':'MONITORING','restarts':restarts,'updated':time.time()})
            time.sleep(2)


def launch(root):
    config,release,runtime,run=_campaign(root)
    run.mkdir(parents=True,exist_ok=True)
    with RunLock(run/'launch.lock'):
        if (run/'watchdog_owner.json').exists():
            record=read_json(run/'watchdog_owner.json')
            if _owned(record): return {'status':'ALREADY_RUNNING','owner':record}
        token_path=run/'token.json'
        if not token_path.exists(): atomic_json(token_path,{'token':uuid.uuid4().hex},immutable=True)
        token=read_json(token_path)['token']
        atomic_json(run/'control.json',{'action':'run','updated':time.time()})
        command=[sys.executable,'-u',str(runtime/'bootstrap.py'),'capstone_lab.synthesis.formal',
                 'watch','--project-root',str(root),'--token',token]
        logs=run/'watchdog_launches'/uuid.uuid4().hex;logs.mkdir(parents=True)
        with (logs/'stdout.log').open('xb') as out,(logs/'stderr.log').open('xb') as err:
            child=spawn(command,cwd=runtime,stdout=out,stderr=err,
                        env=dict(os.environ,PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1'))
        created=psutil.Process(child.pid).create_time()
        for _ in range(150):
            if child.poll() is not None: raise StateError('formal watchdog exited before handshake')
            if (run/'watchdog_owner.json').exists():
                owner=read_json(run/'watchdog_owner.json')
                if owner['pid']==child.pid and abs(owner['created']-created)<.001 and _owned(owner):
                    _require(not process_in_job(child.pid),'formal watchdog remained in session job')
                    return {'status':'STARTED','owner':owner,'logs':str(logs),'release':release['release_sha256']}
            time.sleep(.1)
        raise StateError('formal watchdog handshake timeout')


def status(root):
    config,release,runtime,run=_campaign(root)
    state=_load_state(run,config)
    state['release_sha256']=release['release_sha256']
    state['ledger']=read_json(run/'active_time_ledger.json') if (run/'active_time_ledger.json').exists() else None
    state['watchdog_alive']=_owned(read_json(run/'watchdog_owner.json')) is not None if (run/'watchdog_owner.json').exists() else False
    state['supervisor_alive']=_owned(read_json(run/'supervisor_owner.json')) is not None if (run/'supervisor_owner.json').exists() else False
    state['progress']={}
    for job in config['jobs']:
        path=run/'datasets'/job['subset']/'progress.json'
        if path.exists(): state['progress'][job['id']]=read_json(path)
    return state


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze-release');f.add_argument('--output',type=Path,default=RELEASE.parent)
    sub.add_parser('launch');sub.add_parser('status');sub.add_parser('pause');sub.add_parser('resume')
    for name in ('watch','supervise'):
        p=sub.add_parser(name);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--token',required=True)
    w=sub.add_parser('worker');w.add_argument('--project-root',type=Path,required=True);w.add_argument('--token',required=True);w.add_argument('--job',required=True);w.add_argument('--attempt',required=True)
    args=parser.parse_args();root=getattr(args,'project_root',Path.cwd()).resolve()
    if args.command=='freeze-release': result=freeze_release(root,(root/args.output).resolve())
    elif args.command=='launch': result=launch(root)
    elif args.command=='status': result=status(root)
    elif args.command in ('pause','resume'):
        config,_,_,run=_campaign(root);run.mkdir(parents=True,exist_ok=True)
        atomic_json(run/'control.json',{'action':'pause' if args.command=='pause' else 'run','updated':time.time()})
        result={'status':args.command.upper()}
        if args.command=='resume':
            state=_load_state(run,config)
            if state['status'] in {'PAUSED','BLOCKED_RESOURCES'}:
                state['status']='PLANNED';atomic_json(run/'campaign_state.json',state)
            result=launch(root)
    elif args.command=='watch': result=watch(root,args.token)
    elif args.command=='supervise': result=supervise(root,args.token)
    else: raise SystemExit(worker(root,args.token,args.job,args.attempt))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
