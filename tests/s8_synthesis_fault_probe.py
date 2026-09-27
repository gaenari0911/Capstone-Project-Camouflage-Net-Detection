"""Explicit owned-process fault injection for the new preflight only."""
import argparse
import json
from pathlib import Path
import time
import psutil
from capstone_lab.campaign.contracts import read_json
from capstone_lab.campaign.io import atomic_json


def terminate_owned_worker(root,output):
    if not output.is_relative_to(root/'artifacts/s8_heuristic_preflight'):
        raise ValueError('fault injection outside preflight')
    candidates=[]
    for path in (output/'attempts').glob('*/*/worker.json'):
        r=read_json(path)
        try:
            p=psutil.Process(r['pid'])
            if p.create_time()==r['created'] and p.is_running():
                if str(output) not in p.cmdline() or 'capstone_lab.synthesis.full' not in p.cmdline():
                    raise ValueError('worker command mismatch')
                candidates.append((p,r))
        except psutil.NoSuchProcess: pass
    if len(candidates)!=1:
        raise ValueError('expected exactly one owned live worker')
    p,r=candidates[0]
    descendants=p.children(recursive=True)
    p.terminate()
    for child in descendants:
        try: child.terminate()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs([p]+descendants,timeout=10)
    evidence={'injected':'worker_tree_termination','owner':r,'children':[d.pid for d in descendants],
              'time':time.time()}
    atomic_json(output/'fault_probe.json',evidence,immutable=True)
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coordinator',action='store_true')
    a=p.parse_args()
    if a.coordinator:
        from capstone_lab.synthesis.managed import alive
        output=a.output.resolve()
        if not output.is_relative_to(Path.cwd().resolve()/'artifacts/s8_heuristic_preflight'):
            raise ValueError('outside preflight')
        record=read_json(output/'owner.json')
        if not alive(record):
            raise ValueError('coordinator ownership missing')
        psutil.Process(record['pid']).terminate()
        atomic_json(output/'coordinator_fault_probe.json',{'owner':record,'time':time.time()},immutable=True)
        print('terminated owned preflight coordinator only',record['pid'])
    else:
        print(json.dumps(terminate_owned_worker(Path.cwd().resolve(),a.output.resolve()),indent=2))
