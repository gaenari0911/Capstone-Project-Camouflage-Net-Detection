"""Collect reproducible evidence; never upgrades partial preflight to formal release."""
import json
from pathlib import Path
from capstone_lab.campaign.contracts import read_json
from capstone_lab.campaign.io import atomic_json
from capstone_lab.config import sha256_file


def collect(root):
    out=root/'artifacts/s8_heuristic_verification/run_v2'
    pairs={}
    for subset,prefix,count in [('train748','full',1),('low187','low',2)]:
        runs=[root/('artifacts/s8_heuristic_preflight/'+prefix+'_w'+str(w)+'_v3') for w in (1,2)]
        reports=[read_json(run/'report.json') for run in runs]
        if reports[0]['output_hashes']!=reports[1]['output_hashes']:
            raise ValueError('worker count changed output')
        for run,r in zip(runs,reports):
            for key,hashes in r['output_hashes'].items():
                for name,expected in hashes.items():
                    filename={'image':'image.png','mask':'mask.png','polygon':'polygon.txt','metadata':'metadata.json'}[name]
                    if sha256_file(run/'samples'/key/filename)!=expected:
                        raise ValueError('output modified: '+key)
            if r['status']!='VERIFIED_SMALL_RENDER' or any(n!=count for n in r['counts'].values()):
                raise ValueError('incomplete small render')
        pairs[subset]={'worker_output_hashes_equal':True,
            'runs':[r.relative_to(root).as_posix() for r in runs],
            'report_hashes':[sha256_file(r/'report.json') for r in runs],
            'contract_hashes':[r['contract_sha256'] for r in reports],
            'seconds':[r['seconds'] for r in reports],
            'counts':reports[0]['counts'],
            'quota':reports[0]['plan_quotas']}
    paths=['configs/campaigns/s8_heuristic_policy_v1.json',
           'configs/approvals/s8_foreground_amendment_v1.json',
           'configs/approvals/s8_low_diversity_v1.json',
           'artifacts/s8_source_audit/split_pools_v1/split_source_summary.json',
           'artifacts/s8_full_backgrounds/v1/manifest.json']
    paths += [p.relative_to(root).as_posix() for p in (root/'capstone_lab/synthesis').glob('*.py')]
    paths += ['capstone_lab/campaign/foreground_policy.py','capstone_lab/campaign/contracts.py']
    report={'status':'PARTIALLY_VERIFIED_NOT_FORMAL_RELEASE','formal_generation_started':False,
        'training_or_inference_started':False,'pairs':pairs,
        'input_code_hashes':{p:sha256_file(root/p) for p in sorted(paths)},
        'verified':['2496 FULL background decodes and hashes','source budget isolation',
                    '21000 target schedule/capacity','L3000 balanced matching simulation',
                    'FULL small geometry/render/output hash determinism',
                    'owned worker termination/retry and coordinator death/duplicate refusal/reuse on v2',
                    'legacy component atomic partial commit/resume and Windows breakaway regressions'],
        'remaining':['production release gate and immutable runtime launcher for21000',
                     'one campaign-wide CPU/IO admission across datasets; current managed wrapper is per-run',
                     'durable synthesis active-time ledger across interruptions; current preflight accounting is insufficient for production',
                     'FULL integration injection at partial paired commit, pause/resource transitions, and full-config tamper scenarios',
                     'revalidate production supervisor and frozen runtime before unattended generation'],
        'post_render_change':'contracts.read_json transient PermissionError retry; covered by final regression, does not change rendering math',
        'ssh_disconnect_test':'NOT_PERFORMED','power_cut_test':'NOT_PERFORMED',
        'mixed_training_loader':'NOT_IMPLEMENTED_OR_VERIFIED',
        'visual_review':'local image-view tool failed sandbox runner; do not claim manual quality inspection'}
    atomic_json(out/'summary.json',report,immutable=True)
    print(json.dumps({'status':report['status'],'pairs':pairs,'summary':str(out/'summary.json')},indent=2))


if __name__=='__main__':
    collect(Path.cwd().resolve())
