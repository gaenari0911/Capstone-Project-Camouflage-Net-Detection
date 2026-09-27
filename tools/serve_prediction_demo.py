"""Local-only upload demo backed by frozen seed0 checkpoints."""
import argparse
import base64
import gc
import io
import json
import time
from http.server import HTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from qualitative_core import ROOT,BASE,OUTPUT,JOBS,FrozenEngine,jpeg_data
from capstone_lab.campaign.io import RunLock

class Handler(SimpleHTTPRequestHandler):
    engine=None
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(OUTPUT),**kwargs)

    def send_json(self,status,value):
        data=json.dumps(value,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

    def do_GET(self):
        if self.path=='/api/health':return self.send_json(200,{'status':'ready','mode':'local GPU inference','models':JOBS})
        return super().do_GET()

    def do_POST(self):
        if self.path!='/api/predict':return self.send_json(404,{'error':'Unknown endpoint'})
        origin=self.headers.get('Origin')
        if origin and urlparse(origin).netloc!=self.headers.get('Host'):
            return self.send_json(403,{'error':'Use the demo page served by this local server'})
        try:
            count=int(self.headers.get('Content-Length','0'))
            if not 0<count<=12*1024*1024:raise ValueError('Image request must be at most 12 MiB')
            request=json.loads(self.rfile.read(count));job=request.get('job')
            if job not in JOBS:raise ValueError('Unknown model')
            encoded=request['image'];encoded=encoded.split(',',1)[1] if encoded.startswith('data:') else encoded
            data=base64.b64decode(encoded,validate=True)
            with Image.open(io.BytesIO(data)) as source:
                if source.width*source.height>25_000_000:raise ValueError('Image exceeds 25 megapixels')
                rgb=np.asarray(source.convert('RGB'))
            started=time.perf_counter()
            with RunLock(ROOT/'artifacts/s8_gpu.lock'):
                cold=Handler.engine is None or Handler.engine.job!=job
                if cold:
                    Handler.engine=None;gc.collect();torch.cuda.empty_cache()
                    Handler.engine=FrozenEngine(job,BASE/'live'/job)
                    Handler.engine.predict(np.zeros((640,640,3),dtype=np.uint8))
                result=Handler.engine.predict(rgb)
            result['image']=jpeg_data(result.pop('grid'))
            result.update(cold_model_load=cold,server_ms=(time.perf_counter()-started)*1000,
                confidence=.25,nms_iou=.7,ground_truth_available=False)
            self.send_json(200,result)
        except (ValueError,KeyError,TypeError,OSError) as exc:self.send_json(400,{'error':str(exc)})
        except Exception as exc:
            self.send_json(503,{'error':str(exc),'hint':'Check CUDA availability or another active GPU job; no training was changed.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    if not (OUTPUT/'demo.html').exists():raise RuntimeError('Build the qualitative package first')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    server=HTTPServer(('127.0.0.1',a.port),Handler)
    print(f'Open http://127.0.0.1:{a.port}/demo.html (Ctrl+C to stop)',flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':main()
