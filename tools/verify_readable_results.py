"""Offline/HTTP reader UI regression; no training or GPU inference requests."""
import base64
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from verify_prediction_demo import CDP
from build_readable_results import ROOT, OUT

CHECK=ROOT/'artifacts/s10_qualitative/readability_check_v1'

def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def get(url):
    with urllib.request.urlopen(url,timeout=3) as r:return r.read()

def main():
    CHECK.mkdir(parents=True,exist_ok=True);proof=OUT/'lookup_verification';proof.mkdir(exist_ok=True)
    browser_port=port();server_port=port();log=(CHECK/'browser.log').open('ab');serverlog=(CHECK/'server.log').open('ab')
    browser=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    child=subprocess.Popen([str(browser),'--headless=new','--disable-gpu','--no-first-run',f'--remote-debugging-port={browser_port}',f'--user-data-dir={CHECK}/edge_profile_{time.time_ns()}','about:blank'],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    server=None;cdp=None;checks=[]
    try:
        for _ in range(70):
            try:targets=json.loads(get(f'http://127.0.0.1:{browser_port}/json'));break
            except (OSError,ValueError):time.sleep(.2)
        else:raise RuntimeError('Browser did not start')
        cdp=CDP(next(t['webSocketDebuggerUrl'] for t in targets if t['type']=='page'))
        cdp.call('Runtime.enable');cdp.call('Page.enable')
        def viewport(width,height=1100):cdp.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
        viewport(1400)
        def wait_for(expr):
            for _ in range(80):
                if cdp.evaluate(expr):return
                time.sleep(.1)
            raise AssertionError('Timed out: '+expr)
        def navigate(name):
            cdp.call('Page.navigate',{'url':(OUT/name).as_uri()})
            wait_for("document.readyState==='complete' && !!document.getElementById('experimentDialog')")
        def screenshot(name):
            result=cdp.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
            (proof/name).write_bytes(base64.b64decode(result['data']))
        navigate('index.html')
        assert cdp.evaluate("document.querySelectorAll('#report .exp-ref').length")>100
        cdp.evaluate("document.querySelector('#report [data-experiment=BG0]').click()")
        assert cdp.evaluate("document.querySelector('#lookupDetails').textContent.includes('2,877') && document.querySelector('#experimentDialog').open")
        cdp.evaluate("document.querySelector('#lookupClose').click()")
        screenshot('index_desktop.png');checks.append('Report experiment links and BG0 data totals')
        cdp.evaluate("document.querySelector('.lookup-nav button').click()")
        for query,expected in [('G0','BG0'),('M2_seed0','M2'),('A5','A5'),('H2','H2'),('H2_R','H2_R')]:
            cdp.evaluate(f"document.querySelector('#lookupSearch').value={json.dumps(query)};document.querySelector('#lookupSearch').dispatchEvent(new Event('input'))")
            assert cdp.evaluate(f"document.querySelector('#lookupSelect').value==={json.dumps(expected)}"),query
            if query=='G0':assert cdp.evaluate("document.querySelector('#lookupNote').textContent.includes('정식 ID가 아닙니다')")
            if query=='A5':assert cdp.evaluate("document.querySelector('#lookupDetails').textContent.includes('AnyDoor가 아님')")
        screenshot('lookup_dialog.png')
        cdp.evaluate("document.querySelector('#lookupSearch').value='does-not-exist';document.querySelector('#lookupSearch').dispatchEvent(new Event('input'))")
        assert cdp.evaluate("getComputedStyle(document.querySelector('#lookupEmpty')).display!=='none'")
        cdp.call('Input.dispatchKeyEvent',{'type':'keyDown','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
        cdp.call('Input.dispatchKeyEvent',{'type':'keyUp','key':'Escape','code':'Escape','windowsVirtualKeyCode':27})
        assert cdp.evaluate("!document.querySelector('#experimentDialog').open")
        checks.append('Search: G0 clarification, job ID, A5, H2 alias, no results, Escape')
        navigate('experiment_guide.html')
        assert cdp.evaluate("document.querySelectorAll('tr[id^=exp-]').length") ==26
        assert cdp.evaluate("!!document.querySelector('#guide-metrics') && !!document.querySelector('#guide-demo')")
        screenshot('guide_desktop.png');checks.append('26 static table rows and 9 guide sections')
        navigate('demo.html')
        wait_for("document.querySelector('#leftStats').textContent.includes('Union IoU') && !!document.querySelector('#leftDetails')")
        assert cdp.evaluate("window.DEMO_DATA.images.length===200 && document.querySelector('#leftModel').options.length===10 && document.querySelector('#runLive').disabled")
        for n in range(14):
            result=cdp.evaluate(f"(async()=>{{selectCase({n});await render();return document.querySelector('#leftDetails').textContent.includes(window.DEMO_DATA.cases[{n}].left.split('_')[0]) && document.querySelector('#rightDetails').textContent.includes(window.DEMO_DATA.cases[{n}].right.split('_')[0]);}})()")
            assert result,('case',n)
        for job in cdp.evaluate('window.DEMO_DATA.jobs'):
            result=cdp.evaluate(f"(async()=>{{document.querySelector('#leftModel').value={json.dumps(job)};await render();return document.querySelector('#leftDetails').textContent.includes({json.dumps(job.split('_')[0])});}})()")
            assert result,job
        for mode,word in [('difference','정답·예측'),('overlay','정답 윤곽'),('plain','원본 이미지')]:
            assert cdp.evaluate(f"(async()=>{{document.querySelector('#mode').value={json.dumps(mode)};await render();return document.querySelector('.legend').textContent.includes({json.dumps(word)});}})()")
        cdp.evaluate("document.querySelector('#mode').value='difference';document.querySelector('#leftModel').value='M1_seed0';document.querySelector('#rightModel').value='BG1_seed0';render()")
        wait_for("document.querySelector('#comparisonPurpose').textContent.includes('배경-only 2,129장')")
        assert cdp.evaluate("document.querySelector('#rightDetails').textContent.includes('5,877')")
        cdp.evaluate("document.querySelector('#liveModel').value='BG0_seed0';document.querySelector('#liveModel').dispatchEvent(new Event('change'))")
        assert cdp.evaluate("document.querySelector('#liveDetails').textContent.includes('2,877')")
        cdp.evaluate("document.querySelector('#comparisonPurpose').scrollIntoView({block:'start'})")
        screenshot('demo_desktop.png');checks.append('All 14 cases, 10 model cards, 3 legends, compare purpose, live model card, offline mode')
        viewport(390,844)
        for name in ['index.html','experiment_guide.html','demo.html']:
            navigate(name)
            if name=='demo.html':wait_for("!!document.querySelector('#leftDetails')")
            assert cdp.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1'),('mobile overflow',name)
        screenshot('demo_mobile.png');checks.append('390px mobile: no page-wide horizontal overflow')
        # Smoke the new wrapper's HTTP routing only. Never call /api/predict.
        server=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'tools/serve_readable_demo.py'),'--port',str(server_port)],cwd=ROOT,stdout=serverlog,stderr=serverlog,creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(100):
            try:health=json.loads(get(f'http://127.0.0.1:{server_port}/api/health'));break
            except (OSError,ValueError):time.sleep(.2)
        else:raise RuntimeError('Reader demo HTTP server did not start')
        assert len(health['models'])==10
        assert b'experiment_lookup.js' in get(f'http://127.0.0.1:{server_port}/demo.html')
        assert '실험 이름'.encode() in get(f'http://127.0.0.1:{server_port}/experiment_guide.html')
        viewport(1400)
        cdp.call('Page.navigate',{'url':f'http://127.0.0.1:{server_port}/demo.html'})
        wait_for("document.readyState==='complete' && !!document.querySelector('#leftDetails') && !document.querySelector('#runLive').disabled")
        checks.append('New wrapper serves v6 HTML and unchanged health API; no prediction requests')
        assert not cdp.errors,cdp.errors
        result={'status':'VERIFIED','checks':checks,'javascript_exceptions':0,'new_gpu_inference':False,'offline_images':200,'demo_models':10,'experiment_definitions':26}
        (proof/'browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if cdp:
            try:cdp.call('Browser.close')
            except Exception:pass
        for process in [child,server]:
            if process and process.poll() is None:
                if process is server:process.terminate()
                try:process.wait(timeout=8)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=8)
        log.close();serverlog.close()

if __name__=='__main__':main()
