"""Bounded actual FULL acceptance through the same frozen supervisor and worker."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import psutil
from capstone_lab.config import sha256_file
from capstone_lab.campaign.contracts import read_json, digest
from capstone_lab.campaign.io import atomic_json
from . import formal

INDEX='artifacts/s8_heuristic_runtime_acceptance/latest.json'

def code_hashes(root):
    return {p.relative_to(root).as_posix():sha256_file(p) for p in formal._runtime_files(root)}

def wait_for(check, description, timeout=900):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=check()
        if value: return value
        time.sleep(.5)
    raise AssertionError('acceptance timeout: '+description)

def run(root):
    start=time.time()
    hashes=code_hashes(root)
    folder=root/'artifacts/s8_heuristic_preflight'/('runtime_'+uuid.uuid4().hex[:12])
    runtime=folder/'runtime';runtime.mkdir(parents=True)
    for relative,expected in hashes.items():
        destination=runtime/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,destination)
        assert sha256_file(destination)==expected
    (runtime/'bootstrap.py').write_text(formal._bootstrap_text(),encoding='utf-8')
    files={**hashes,'bootstrap.py':sha256_file(runtime/'bootstrap.py')}
    atomic_json(runtime/'runtime_manifest.json',{'files':files},immutable=True)
    cfg={'mode':'BOUNDED_RUNTIME_ACCEPTANCE','output':(folder/'run').relative_to(root).as_posix(),
         'workers':2,'runtime_code_sha256':digest(files),'input_hashes':formal._input_hashes(root),
         'prior_real_active_seconds':read_json(root/formal.REAL_STATUS)['active_seconds'],
         'jobs':[{'id':'train_probe','subset':'train748','count':2},
                 {'id':'low_probe','subset':'low187','count':1}]}
    path=folder/'acceptance_config.json';atomic_json(path,cfg,immutable=True)
    env=dict(os.environ,CAPSTONE_SYNTHESIS_ACCEPTANCE=str(path),PYTHONUNBUFFERED='1')
    output=folder/'run'
    def command(action):
        result=subprocess.run([sys.executable,'-m','capstone_lab.synthesis.formal',action],
            cwd=root,env=env,capture_output=True,text=True,timeout=40,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode: raise AssertionError(result.stdout+result.stderr)
        return json.loads(result.stdout)
    def state():
        return read_json(output/'campaign_state.json') if (output/'campaign_state.json').exists() else {'jobs':{}}
    def progress():
        p=output/'datasets/train748/progress.json'
        return read_json(p) if p.exists() else {}
    proof={'code':hashes,'config_path':path.relative_to(root).as_posix(),
           'config_sha256':sha256_file(path),'faults':[]}
    try:
        started=command('launch')
        assert started['status']=='STARTED' and not started['owner']['in_job']
        assert command('launch')['status']=='ALREADY_RUNNING'
        row=wait_for(lambda: state()['jobs'].get('train_probe') if
                     progress().get('feature_tasks_completed',0)>30 else None,'first worker features')
        owner=row['owner'];p=formal._owned(owner);assert p
        descendants=[(c.pid,c.create_time()) for c in p.children(recursive=True)]
        p.terminate()
        # Kill only the coordinator of the CPU pool: the supervisor must clean
        # its registered orphan children before replacing it.
        proof['faults'].append({'kind':'worker_only','owner':owner,'children':descendants,'time':time.time()})
        row=wait_for(lambda: state()['jobs'].get('train_probe') if
            state()['jobs'].get('train_probe',{}).get('attempts')==2 and
            state()['jobs'].get('train_probe',{}).get('state')=='RUNNING' else None,'worker retry2')
        second_owner=row['owner']
        for pid,created in descendants:
            try: assert abs(psutil.Process(pid).create_time()-created)>.001,'orphan pool child survived retry'
            except psutil.NoSuchProcess: pass
        supervisor=read_json(output/'supervisor_owner.json');p=formal._owned(supervisor);assert p;p.terminate()
        proof['faults'].append({'kind':'supervisor_only','owner':supervisor,'time':time.time()})
        wait_for(lambda: read_json(output/'supervisor_owner.json')['pid']!=supervisor['pid'],'supervisor restart')
        assert state()['jobs']['train_probe']['attempts']==2
        assert state()['jobs']['train_probe']['owner']==second_owner
        first=output/'datasets/train748/samples/train748_000001/PAIR.json'
        wait_for(first.exists,'first atomic pair')
        preserved={p.relative_to(output).as_posix():[sha256_file(p),p.stat().st_mtime_ns]
                   for p in first.parent.rglob('*') if p.is_file()}
        command('pause')
        wait_for(lambda: state().get('status')=='PAUSED','pause boundary')
        wait_for(lambda: formal._owned(read_json(output/'watchdog_owner.json')) is None,'watchdog paused exit',40)
        assert state()['jobs']['low_probe']['attempts']==0
        command('resume')
        terminal=wait_for(lambda: state() if state().get('status') in formal.TERMINAL else None,'resumed completion')
        assert terminal['status']=='SUCCEEDED',terminal
        assert terminal['jobs']['train_probe']['attempts']==3
        assert terminal['jobs']['low_probe']['attempts']==1
        for relative,value in preserved.items():
            p=output/relative;assert [sha256_file(p),p.stat().st_mtime_ns]==value
        reports={}
        for subset,counts in [('train748',{c:2 for c in ('A0','A1','A2','A3','A4','A5')}),('low187',{'A1':1})]:
            report=read_json(output/'datasets'/subset/'report.json');assert report['counts']==counts
            reports[subset]={'path':(output/'datasets'/subset/'report.json').relative_to(root).as_posix(),
                             'sha256':sha256_file(output/'datasets'/subset/'report.json')}
        ledger=read_json(output/'active_time_ledger.json')
        assert ledger['formal_active_seconds']>=time.time()-start-60
        assert ledger['prior_real_active_seconds']==cfg['prior_real_active_seconds']
        assert hashes==code_hashes(root),'source code changed during acceptance'
        proof.update(status='VERIFIED_RUNTIME_ACCEPTANCE',reports=reports,ledger=ledger,
                     seconds=time.time()-start,checks=['worker_retry_same_job','supervisor_adopts_same_worker',
                     'duplicate_launch_rejected','pause_before_low','resume_preserves_commits',
                     'two_datasets_sequential','reserved_time_survives_worker_crash','in_job_false'])
        atomic_json(folder/'summary.json',proof,immutable=True)
        atomic_json(root/INDEX,{'path':(folder/'summary.json').relative_to(root).as_posix(),
                    'sha256':sha256_file(folder/'summary.json')})
        print(json.dumps({'status':proof['status'],'folder':str(folder),'seconds':proof['seconds']}),flush=True)
    except BaseException as exc:
        # Stop this bounded acceptance at the next checkpoint. Never launch formal here.
        if output.exists(): atomic_json(output/'control.json',{'action':'pause','reason':'acceptance failure'})
        proof.update(status='FAILED_RUNTIME_ACCEPTANCE',error=repr(exc),seconds=time.time()-start)
        atomic_json(folder/'failed.json',proof)
        raise

def verify_acceptance(root):
    index=read_json(root/INDEX)
    path=root/index['path'];assert sha256_file(path)==index['sha256']
    proof=read_json(path)
    assert proof['status']=='VERIFIED_RUNTIME_ACCEPTANCE'
    assert proof['code']==code_hashes(root),'runtime acceptance code no longer current'
    for row in proof['reports'].values(): assert sha256_file(root/row['path'])==row['sha256']
    return index

if __name__=='__main__':
    run(Path.cwd().resolve())
