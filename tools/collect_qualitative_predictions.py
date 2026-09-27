"""Replay fixed seed0 models on all Test200; verify recorded aggregate metrics."""
import gc
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from qualitative_core import ROOT,BASE,JOBS,FrozenEngine,catalog,bitmask,union,iou,read,preprocess_rgb
from capstone_lab.config import sha256_file
from capstone_lab.campaign.io import RunLock,atomic_json
from capstone_lab.campaign.data import ReadOnlySegDataset,make_loader,effective_args

@torch.inference_mode()
def run():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    BASE.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (ROOT/'manifests/real_split_observed_v1.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    rows=[r for r in rows if r['original_split']=='test'];assert len(rows)==200
    for r in rows:
        for k in ['image','label']:assert sha256_file(ROOT/r[k])==r[k+'_sha256']
    cat=catalog();assert set(cat)==set(JOBS)
    atomic_json(BASE/'plan.json',{'jobs':JOBS,'seed_selection':'seed0 fixed before case selection, not best Test seed',
        'purpose':'post-hoc explanatory examples; not new model selection or independent evaluation',
        'images':200,'confidence':.25,'mask_iou':.5,'AP_min_conf':.001,'nms_iou':.7,'batch':16,
        'model_input':640,'mask_grid':160,'case_score':'union-mask IoU and fixed-threshold instance TP/FP/FN',
        'checkpoints':{j:{k:cat[j][k] for k in ['checkpoint','sha256']} for j in JOBS}},immutable=True)
    image_dir=BASE/'images';image_dir.mkdir(exist_ok=True)
    dataset=ReadOnlySegDataset(ROOT,rows,effective_args(ROOT),augment=False)
    index_by_path={str((ROOT/r['image']).resolve()):i for i,r in enumerate(rows)}
    gt_records={};verified=[]
    with RunLock(ROOT/'artifacts/s8_gpu.lock'),RunLock(BASE/'collector.lock'):
        for job in JOBS:
            folder=BASE/job;folder.mkdir(exist_ok=True);done=folder/'result.json'
            if done.exists():
                saved=read(done)
                assert saved['checkpoint_sha256']==cat[job]['sha256']
                assert saved['status']=='VERIFIED' and len(saved['records'])==200
                verified.append(job);continue
            atomic_json(BASE/'status.json',{'status':'RUNNING','job':job,'completed':len(verified),'total':len(JOBS),'updated':time.time()})
            engine=FrozenEngine(job,folder);loader=make_loader(dataset,seed=0,workers=2)
            validator=engine.new_validator(folder,loader)
            tp=fp=fn=0;records=[];timings=[];boundary=[]
            hook=engine.model.model[-1].cv5.register_forward_hook(lambda *_:boundary.append(True))
            try:
                for bi,raw in enumerate(loader):
                    batch=validator.preprocess(raw)
                    torch.cuda.synchronize();started=time.perf_counter()
                    preds=validator.postprocess(engine.model(batch['img']))
                    torch.cuda.synchronize();timings.append((time.perf_counter()-started)*1000)
                    validator.update_metrics(preds,batch)
                    for index,p in enumerate(preds):
                        target=validator._prepare_batch(index,batch)
                        image_index=index_by_path[str(Path(target['im_file']).resolve())]
                        fixed={k:v[p['conf']>=.25] for k,v in p.items()}
                        matched=validator._process_batch(fixed,target)['tp_m'][:,0]
                        correct=int(matched.sum());false=len(matched)-correct;miss=len(target['cls'])-correct
                        tp+=correct;fp+=false;fn+=miss
                        predicted=fixed['masks'].cpu().numpy().astype(bool);gt=target['masks'].cpu().numpy().astype(bool)
                        pred_union=union(predicted);gt_union=union(gt)
                        if job==JOBS[0]:
                            rgb=np.rint(batch['img'][index].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                            Image.fromarray(rgb).save(image_dir/f'{image_index:03d}.jpg',quality=95)
                            if image_index in [0,49,99,149,199]:
                                source=cv2.imdecode(np.fromfile(ROOT/rows[image_index]['image'],dtype=np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]
                                assert np.array_equal(preprocess_rgb(source),rgb),'Live preprocessing mismatch'
                            gt_records[image_index]={'index':image_index,'source':rows[image_index]['image'],
                                'image_sha256':rows[image_index]['image_sha256'],'gt_bits':bitmask(gt_union),
                                'shape':list(gt_union.shape),'gt_count':len(gt),'image':f'images/{image_index:03d}.jpg'}
                        confidence=fixed['conf'].cpu().tolist()
                        record=dict(index=image_index,job=job,source=rows[image_index]['image'],
                            tp=correct,fp=false,fn=miss,instance_f1=2*correct/(2*correct+false+miss) if 2*correct+false+miss else 0,
                            union_iou=iou(pred_union,gt_union),mask_bits=bitmask(pred_union),
                            boxes=fixed['bboxes'].cpu().tolist(),confidences=confidence,n_pred=len(confidence))
                        records.append(record)
                        # Instance masks are retained for reproducible matching/visual analysis.
                        np.savez_compressed(folder/f'{image_index:03d}.npz',pred_masks=predicted,gt_masks=gt,
                            confidence=np.asarray(confidence),boxes=fixed['bboxes'].cpu().numpy())
                    atomic_json(BASE/'status.json',{'status':'RUNNING','job':job,'batch':bi+1,'batches':len(loader),'completed':len(verified),'total':len(JOBS),'updated':time.time()})
                library={k:float(v) for k,v in validator.get_stats().items()}
            finally:hook.remove()
            expected=cat[job]['recorded_metrics'];differences={k:library[k]-v for k,v in expected['library_metrics'].items()}
            count_match=(tp,fp,fn)==(expected['mask_tp'],expected['mask_fp'],expected['mask_fn'])
            passed=count_match and not boundary and all(abs(v)<1e-10 for v in differences.values())
            saved={'status':'VERIFIED' if passed else 'MISMATCH','job':job,'checkpoint_sha256':cat[job]['sha256'],
                'records':sorted(records,key=lambda r:r['index']),'library_metrics':library,'library_differences':differences,
                'counts_match':count_match,'mask_tp':tp,'mask_fp':fp,'mask_fn':fn,'boundary_calls':len(boundary),
                'batch_inference_ms':timings,'timing_scope':'synchronized forward+postprocess, batch16 except final8; includes cold first batch',
                'updated':time.time()}
            atomic_json(done,saved,immutable=True)
            if not passed:raise RuntimeError(f'{job}: prediction replay differs from frozen metrics')
            if job==JOBS[0]:atomic_json(BASE/'images.json',[gt_records[i] for i in range(200)],immutable=True)
            verified.append(job);print(json.dumps({'job':job,'verified':True,'completed':len(verified),'tp':tp,'fp':fp,'fn':fn}),flush=True)
            del engine,validator,loader,preds,batch;gc.collect();torch.cuda.empty_cache()
        atomic_json(BASE/'status.json',{'status':'SUCCEEDED','completed':len(verified),'total':len(JOBS),'updated':time.time()})
        assert len(read(BASE/'images.json'))==200

if __name__=='__main__':run()
