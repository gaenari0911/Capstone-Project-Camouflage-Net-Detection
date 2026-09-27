"""Frozen-model inference helpers; no changes to training or evaluation contracts."""
import base64
import copy
import io
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from capstone_lab.config import sha256_file
from capstone_lab.campaign.models import create_model
from capstone_lab.campaign.data import effective_args
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.data.augment import LetterBox

JOBS=[f'{e}_seed0' for e in ['M0','M1','M2','M3','BG0','BG1','L0','L1','L2','L3']]
BASE=ROOT/'artifacts/s10_qualitative/run_v1'
OUTPUT=ROOT/'artifacts/final_presentation/end_to_end_v5_demo'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def catalog():
    old=ROOT/'artifacts/s8_anydoor_campaign/run_v1';new=ROOT/'artifacts/s9_background/run_v1'
    frozen=read(old/'final_freeze.json')['test_checkpoints']+read(new/'final_freeze.json')['checkpoints']
    metrics={r['job']:r for b in [old,new] for r in read(b/'test/summary.json')['records']}
    return {r['job']:{**r,'recorded_metrics':metrics[r['job']]['metrics']} for r in frozen if r['job'] in JOBS}

def bitmask(mask):
    return base64.b64encode(np.packbits(mask.astype(np.uint8).reshape(-1)).tobytes()).decode('ascii')

def unmask(bits,shape=(160,160)):
    return np.unpackbits(np.frombuffer(base64.b64decode(bits),dtype=np.uint8))[:math.prod(shape)].reshape(shape).astype(bool)

def union(masks):
    return np.any(masks,axis=0) if len(masks) else np.zeros(masks.shape[1:],bool)

def iou(a,b):
    den=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/den) if den else 1.

def preprocess_rgb(rgb):
    # Matches BaseDataset.load_image + non-augmenting LetterBox + Format RGB.
    bgr=np.ascontiguousarray(rgb[:,:,::-1]);h,w=bgr.shape[:2];r=640/max(h,w)
    if r!=1:bgr=cv2.resize(bgr,(min(math.ceil(w*r),640),min(math.ceil(h*r),640)),interpolation=cv2.INTER_LINEAR)
    padded=LetterBox(new_shape=(640,640),scaleup=False)(image=bgr)
    return np.ascontiguousarray(padded[:,:,::-1])

def jpeg_data(rgb):
    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=93)
    return 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')

class FrozenEngine:
    def __init__(self,job,folder):
        if job not in JOBS:raise ValueError('Unknown frozen model')
        entry=catalog()[job];checkpoint=ROOT/entry['checkpoint']
        if sha256_file(checkpoint)!=entry['sha256']:raise ValueError('Checkpoint hash changed')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        self.job=job;self.entry=entry;self.args=effective_args(ROOT)
        self.model,_=create_model(ROOT,folder,0,'H2')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model.load_state_dict(state['model'],strict=True);del state
        self.model.cuda().eval()
        self.validator=self.new_validator(folder)

    def new_validator(self,folder,loader=None):
        v=SegmentationValidator(dataloader=loader,save_dir=folder,args=vars(copy.deepcopy(self.args)))
        v.device=torch.device('cuda:0');v.training=True
        v.data={'val':'frozen-test200','names':{0:'camouflage'}};v.init_metrics(self.model)
        return v

    @torch.inference_mode()
    def predict(self,rgb):
        started=time.perf_counter();grid=preprocess_rgb(rgb)
        tensor=torch.from_numpy(np.ascontiguousarray(grid.transpose(2,0,1))).unsqueeze(0).cuda().float()/255
        torch.cuda.synchronize();forward_start=time.perf_counter()
        p=self.validator.postprocess(self.model(tensor))[0]
        torch.cuda.synchronize();inference_ms=(time.perf_counter()-forward_start)*1000
        p={k:v[p['conf']>=.25] for k,v in p.items()}
        masks=p['masks'].cpu().numpy().astype(bool);conf=p['conf'].cpu().tolist()
        return dict(job=self.job,grid=grid,mask_bits=bitmask(union(masks)),mask_shape=list(masks.shape[1:]),
            boxes=p['bboxes'].cpu().tolist(),confidences=conf,n_pred=len(conf),inference_ms=inference_ms,
            preprocessing_inference_ms=(time.perf_counter()-started)*1000)
