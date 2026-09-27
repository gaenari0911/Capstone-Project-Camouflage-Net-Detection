"""Detached, bounded preflight supervisor. Never dispatches formal synthesis."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import traceback
import psutil

from capstone_lab.campaign.detached import spawn, job_info, process_in_job
from capstone_lab.campaign.io import RunLock, atomic_json
from capstone_lab.campaign.contracts import read_json
from .full import memory


def owner(pid, token):
    return {'pid':pid,'created':psutil.Process(pid).create_time(),'token':token}


def alive(record):
    try:
        p=psutil.Process(record['pid'])
        return p.create_time()==record['created'] and record['token'] in p.cmdline() and p.is_running()
    except (psutil.Error,KeyError):
        return False


def monitor(root, output, subset, workers, count, token):
    if job_info()['in_job']:
        raise RuntimeError('preflight supervisor remained in session job')
    with RunLock(output/'supervisor.lock'):
        atomic_json(output/'owner.json',owner(os.getpid(),token))
        initial=time.monotonic()
        child=None
        for attempt in range(1,4):
            if (output/'pause.request').exists(): break
            directory=output/'attempts'/str(attempt)/uuid.uuid4().hex
            directory.mkdir(parents=True)
            command=[sys.executable,'-u','-m','capstone_lab.synthesis.full','--subset',subset,
                     '--output',str(output),'--workers',str(workers),'--count',str(count)]
            with (directory/'stdout.log').open('wb') as stdout, (directory/'stderr.log').open('wb') as stderr:
                child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,PYTHONFAULTHANDLER='1',
                    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                child_created=psutil.Process(child.pid).create_time()
                atomic_json(directory/'worker.json',{'pid':child.pid,'created':child_created,'command':command,
                    'in_job':process_in_job(child.pid)})
                while child.poll() is None:
                    atomic_json(output/'heartbeat.json',{'pid':os.getpid(),'in_job':False,'worker_pid':child.pid,
                        'attempt':attempt,'updated':time.time(),'telemetry':memory(),
                        'active_seconds_this_launch':time.monotonic()-initial})
                    # A live supervisor owns the child handle; never kill processes by name.
                    progress=output/'progress.json'
                    last=progress.stat().st_mtime if progress.exists() else child_created
                    if time.time()-max(last,child_created)>600:
                        p=psutil.Process(child.pid)
                        if p.create_time()!=child_created: raise RuntimeError('worker ownership changed')
                        descendants=p.children(recursive=True)
                        child.terminate()
                        for d in descendants:
                            try: d.terminate()
                            except psutil.NoSuchProcess: pass
                        child.wait(timeout=20)
                        atomic_json(directory/'timeout.json',{'reason':'no feature progress600seconds'})
                    time.sleep(2)
                code=child.returncode
            atomic_json(directory/'exit.json',{'exit_code':code})
            if code==0:
                atomic_json(output/'managed_result.json',{'status':'FINISHED_PREFLIGHT','attempts':attempt,
                    'in_job':False,'active_seconds':time.monotonic()-initial})
                return
            # Integrity/config/resource/geometry failures must not be hidden by blind restart.
            errors=(directory/'stderr.log').read_text(encoding='utf-8',errors='replace')
            if any(s in errors for s in ('ValueError','SAFE_WAIT','BUDGET_STOP','FREE_STOP','contract')):
                break
        atomic_json(output/'managed_result.json',{'status':'STOPPED_REVIEW_REQUIRED',
            'active_seconds':time.monotonic()-initial})


def launch(root,output,subset,workers,count):
    if not output.is_relative_to(root/'artifacts/s8_heuristic_preflight') or count not in range(1,13):
        raise ValueError('only bounded preflight outputs allowed')
    output.mkdir(parents=True,exist_ok=True)
    with RunLock(output/'launch.lock'):
        p=output/'owner.json'
        if p.exists() and alive(read_json(p)):
            return {'status':'ALREADY_RUNNING',**read_json(p)}
        # If a coordinator died while its worker survived, refuse duplicate dispatch.
        for path in (output/'attempts').glob('*/*/worker.json'):
            record=read_json(path)
            try:
                process=psutil.Process(record['pid'])
                if process.create_time()==record['created'] and process.is_running():
                    raise ValueError('surviving worker: wait for completion before coordinator relaunch')
            except psutil.NoSuchProcess: pass
        token=uuid.uuid4().hex
        command=[sys.executable,'-u','-m','capstone_lab.synthesis.managed','--output',str(output),
            '--subset',subset,'--workers',str(workers),'--count',str(count),'--monitor',token]
        with (output/'supervisor.stdout.log').open('ab') as stdout,(output/'supervisor.stderr.log').open('ab') as stderr:
            process=spawn(command,cwd=root,stdout=stdout,stderr=stderr,
                          env=dict(os.environ,PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1'))
        for _ in range(100):
            if p.exists():
                r=read_json(p)
                if r.get('token')==token and alive(r):
                    if process_in_job(r['pid']): raise RuntimeError('breakaway verification failed')
                    return {'status':'STARTED_PREFLIGHT',**r,'in_job':False}
            if process.poll() is not None: raise RuntimeError('supervisor exited before handshake')
            time.sleep(.1)
        raise RuntimeError('supervisor handshake timed out; inspect logs before retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subset',choices=['train748','low187'],required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--count',type=int,default=1)
    parser.add_argument('--monitor')
    a=parser.parse_args()
    root=Path.cwd().resolve()
    if a.monitor:
        monitor(root,a.output.resolve(),a.subset,a.workers,a.count,a.monitor)
    else:
        print(json.dumps(launch(root,a.output.resolve(),a.subset,a.workers,a.count),indent=2))
