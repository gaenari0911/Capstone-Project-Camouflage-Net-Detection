"""Exercise real upload inference and headless Chromium without new dependencies."""
import base64
import hashlib
import json
import os
import socket
import statistics as st
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from qualitative_core import ROOT,BASE,OUTPUT,read,unmask,iou
from capstone_lab.campaign.io import atomic_json

URL='http://127.0.0.1:8765'

def request(path,payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(URL+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)

class CDP:
    def __init__(self,url):
        u=urlparse(url);self.socket=socket.create_connection((u.hostname,u.port),timeout=30);self.stream=self.socket.makefile('rb');self.id=0;self.errors=[]
        key=base64.b64encode(os.urandom(16)).decode()
        self.socket.sendall((f'GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        status=self.stream.readline();assert b'101' in status,status
        while self.stream.readline()!=b'\r\n':pass

    def send(self,payload,opcode=1):
        mask=os.urandom(4);size=len(payload);header=bytes([128|opcode])
        header+=bytes([128|size]) if size<126 else bytes([128|126])+struct.pack('!H',size) if size<65536 else bytes([128|127])+struct.pack('!Q',size)
        self.socket.sendall(header+mask+bytes(x^mask[i%4] for i,x in enumerate(payload)))

    def receive(self):
        fragments=bytearray()
        while True:
            h=self.stream.read(2)
            if len(h)!=2:raise RuntimeError('Browser connection closed')
            opcode=h[0]&15;size=h[1]&127
            if size==126:size=struct.unpack('!H',self.stream.read(2))[0]
            elif size==127:size=struct.unpack('!Q',self.stream.read(8))[0]
            mask=self.stream.read(4) if h[1]&128 else None
            payload=self.stream.read(size)
            if mask:payload=bytes(x^mask[i%4] for i,x in enumerate(payload))
            if opcode==9:self.send(payload,10);continue
            if opcode==8:raise RuntimeError('Browser websocket closed')
            fragments.extend(payload)
            if h[0]&128:return json.loads(fragments)

    def call(self,method,params=None):
        self.id+=1;self.send(json.dumps({'id':self.id,'method':method,'params':params or {}}).encode())
        while True:
            r=self.receive()
            if r.get('method')=='Runtime.exceptionThrown':self.errors.append(r)
            if r.get('id')==self.id:
                if 'error' in r:raise RuntimeError(r['error'])
                return r.get('result',{})

    def evaluate(self,expression):
        r=self.call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
        if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
        return r['result'].get('value')

def main():
    check=BASE/'demo_verification';check.mkdir(exist_ok=True)
    assert request('/api/health')['status']=='ready'
    cases=read(OUTPUT/'case_selection.json')['cases'];images=read(BASE/'images.json');source=ROOT/images[cases[0]['index']]['source']
    payload={'job':'M2_seed0','image':base64.b64encode(source.read_bytes()).decode()}
    cold=request('/api/predict',payload);assert cold['n_pred']>0 and cold['mask_shape']==[160,160]
    cache=read(BASE/'M2_seed0/result.json')['records'][cases[0]['index']]
    cached_agreement=iou(unmask(cache['mask_bits']),unmask(cold['mask_bits']))
    assert cached_agreement>.98,cached_agreement
    timings=[]
    for n in range(15):
        src=ROOT/images[cases[n%3]['index']]['source']
        r=request('/api/predict',{'job':'M2_seed0','image':base64.b64encode(src.read_bytes()).decode()})
        assert r['cold_model_load'] is False and r['ground_truth_available'] is False
        timings.append(r['inference_ms'])
    rejected=[]
    for bad in [{'job':'not-a-model','image':payload['image']},{'job':'M2_seed0','image':'not base64!'}]:
        try:request('/api/predict',bad)
        except urllib.error.HTTPError as e:assert e.code==400;rejected.append(e.code)
        else:raise AssertionError('Bad upload accepted')
    atomic_json(check/'api.json',{'status':'VERIFIED','requests':16,'invalid_requests_rejected':rejected,
        'batch1_vs_recorded_batch16_mask_iou':cached_agreement,'model':'M2_seed0','warm_repetitions':15,
        'warm_forward_postprocess_median_ms':st.median(timings),'warm_forward_postprocess_p95_ms':sorted(timings)[14],
        'warm_forward_postprocess_samples_ms':timings,'scope':'FP32 CUDA batch1; 3 images x5; excludes model load, network, browser, drawing; p95 nearest rank',
        'cold_server_ms':cold['server_ms']})
    print('API and actual GPU inference VERIFIED',flush=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    profile=check/f'edge_profile_{int(time.time())}'
    browser=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    log=(check/'browser.log').open('ab')
    child=subprocess.Popen([str(browser),'--headless=new','--disable-gpu','--no-first-run',f'--remote-debugging-port={port}',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    cdp=None
    try:
        for _ in range(50):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json',timeout=2) as r:targets=json.load(r)
                pages=[t for t in targets if t['type']=='page']
                if pages:break
            except (OSError,ValueError):pass
            time.sleep(.2)
        else:raise RuntimeError('Headless browser failed to start')
        cdp=CDP(pages[0]['webSocketDebuggerUrl']);cdp.call('Runtime.enable');cdp.call('Page.enable')
        cdp.call('Emulation.setDeviceMetricsOverride',{'width':1400,'height':1200,'deviceScaleFactor':1,'mobile':False})
        cdp.call('Page.navigate',{'url':URL+'/demo.html'})
        for _ in range(80):
            if cdp.evaluate("Boolean(document.getElementById('leftStats')?.textContent.includes('Union IoU'))"):break
            time.sleep(.2)
        else:raise RuntimeError('Demo did not render')
        initial=cdp.evaluate("({cases:$('caseSelect').options.length,images:$('imageSelect').options.length,models:$('leftModel').options.length,left:$('leftStats').textContent,right:$('rightStats').textContent})")
        assert (initial['cases'],initial['images'],initial['models'])==(15,200,10),initial
        cdp.evaluate("selectCase(5);$('zoom').value='target';render();")
        time.sleep(.4)
        switched=cdp.evaluate("({index:$('imageSelect').value,left:$('leftModel').value,right:$('rightModel').value,text:$('reason').textContent})")
        assert switched['left']=='M1_seed0' and switched['right']=='BG1_seed0'
        cdp.evaluate("$('zoom').value='full';$('caseSelect').value='0';selectCase(0);")
        time.sleep(.3)
        shot=cdp.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
        (check/'demo_browser.png').write_bytes(base64.b64decode(shot['data']))
        root=cdp.call('DOM.getDocument')['root']['nodeId'];node=cdp.call('DOM.querySelector',{'nodeId':root,'selector':'#upload'})['nodeId']
        cdp.call('DOM.setFileInputFiles',{'nodeId':node,'files':[str(source)]})
        cdp.evaluate("$('runLive').click();")
        for _ in range(100):
            status=cdp.evaluate("$('liveStatus').textContent")
            if 'forward+' in status:break
            if '실패' in status:raise RuntimeError(status)
            time.sleep(.2)
        else:raise RuntimeError('Browser upload did not finish')
        cdp.call('Page.navigate',{'url':(OUTPUT/'demo.html').as_uri()})
        for _ in range(60):
            if cdp.evaluate("Boolean(document.getElementById('leftStats')?.textContent.includes('Union IoU'))"):break
            time.sleep(.2)
        else:raise RuntimeError('Offline file demo did not render')
        offline=cdp.evaluate("$('liveStatus').textContent")
        assert '오프라인' in offline and not cdp.errors,cdp.errors
        atomic_json(check/'browser.json',{'status':'VERIFIED','initial':initial,'case_switch':switched,'upload_status':status,
            'offline_status':offline,'javascript_exceptions':cdp.errors,'checks':['200-image/10-model controls','case selection','zoom render','actual browser upload','offline file rendering']})
        print('Browser interactions, upload and offline mode VERIFIED',flush=True)
    finally:
        if cdp:
            try:cdp.call('Browser.close')
            except Exception:pass
        try:child.wait(timeout=8)
        except subprocess.TimeoutExpired:child.terminate()
        log.close()

if __name__=='__main__':main()
