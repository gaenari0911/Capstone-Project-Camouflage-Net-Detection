"""Verify exported aggregate results with Python's standard library only."""
import argparse
import csv
import hashlib
import json
import math
import io
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
KEYS=['map','map50','precision','recall','f1','box_map','box_map50','fp_per_image']

def rows(name):
    with (ROOT/'results/tables'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--git-index',action='store_true');args=parser.parse_args()
    source=json.loads((ROOT/'results/source_manifest.json').read_text(encoding='utf-8'))
    for rel,record in source['files'].items():
        p=ROOT/rel
        assert p.is_file(),f'Missing export: {rel}'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],f'Changed export: {rel}'
    if args.git_index:
        paths=list(source['files'])
        payload=('\n'.join(':'+p for p in paths)+'\n').encode('utf-8')
        batch=subprocess.run(['git','cat-file','--batch'],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT,check=True)
        stream=io.BytesIO(batch.stdout)
        for path in paths:
            header=stream.readline().split();assert len(header)==3 and header[1]==b'blob',path
            blob=stream.read(int(header[2]));assert stream.read(1)==b'\n'
            assert hashlib.sha256(blob).hexdigest()==source['files'][path]['sha256'],f'Git changed export bytes: {path}'
    metrics=rows('all_checkpoint_metrics.csv');epochs=rows('all_epoch_metrics.csv');summary=rows('all_summary_mean_sd.csv')
    assert len(metrics)==74 and len(epochs)==5340
    assert len({(r['job'],r['split']) for r in metrics})==74
    val=[r for r in metrics if r['split']=='Validation'];test=[r for r in metrics if r['split']=='Test']
    assert len(val)==40 and len(test)==34
    assert len({r['job'] for r in metrics})==40
    assert all(int(r['images'])==100 for r in val) and all(int(r['images'])==200 for r in test)
    histories=defaultdict(list);groups=defaultdict(list)
    for r in epochs:histories[r['job']].append(r)
    for r in val:
        expected=40 if r['experiment'].startswith('A') else 150
        assert int(r['epochs'])==expected
        history=histories[r['job']]
        assert sorted(int(e['epoch']) for e in history)==list(range(1,expected+1)),r['job']
        assert all(math.isfinite(float(e[k])) for e in history for k in ['loss','map','global_step']),r['job']
        assert 1<=int(r['best_epoch'])<=expected
    for r in metrics:
        assert r['job']==r['experiment']+'_seed'+r['seed']
        assert all(math.isfinite(float(r[k])) for k in KEYS)
        for key in KEYS:groups[(r['split'],r['experiment'],key)].append(float(r[key]))
    assert len(summary)==len(groups)
    for row in summary:
        values=groups[(row['split'],row['experiment'],row['metric'])]
        assert int(row['n'])==len(values)
        assert close(row['mean'],statistics.mean(values)),row
        assert close(row['minimum'],min(values)) and close(row['maximum'],max(values))
        if len(values)>1:assert close(row['sample_sd'],statistics.stdev(values)),row
        else:assert row['sample_sd'] in ('','0','0.0','nan'),row
    print(json.dumps({'status':'VERIFIED_AGGREGATES_ONLY','verified_export_files':len(source['files']),
        'training_jobs':40,'epoch_rows':5340,'validation_models':40,'test_models':34,
        'summary_rows':len(summary),'git_index_verified':args.git_index,'new_model_training_or_inference':False},ensure_ascii=False))

if __name__=='__main__':main()
