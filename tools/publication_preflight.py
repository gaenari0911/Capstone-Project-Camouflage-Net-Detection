"""Conservative source-publication audit. Never prints credential values.

This is a heuristic scan of the current proposed tree, not full Git history,
legal clearance, or a guarantee that no secret exists.
"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    'github_token':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})',
    'openai_key':r'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}',
    'aws_access_id':r'AKIA[0-9A-Z]{16}',
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'slack_token':r'xox[baprs]-[0-9A-Za-z-]{20,}',
    'google_api_key':r'AIza[0-9A-Za-z_-]{35}',
}
DENIED={'.pt','.pth','.ckpt','.safetensors','.zip','.7z','.sqlite','.db','.pem','.key','.jpg','.jpeg','.webp','.bmp','.gif','.npz','.npy','.mp4'}

def main():
    raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    names=sorted(set(n.decode('utf-8') for n in raw.split(b'\0') if n))
    findings=[];size=0;python_files=0;media=[]
    expected=json.loads((ROOT/'results/source_manifest.json').read_text(encoding='utf-8'))['files']
    for missing in sorted(set(expected)-set(names)):
        findings.append({'file':missing,'reason':'required export excluded from proposed Git tree'})
    for name in names:
        p=ROOT/name
        if p.is_symlink() or not p.is_file():findings.append({'file':name,'reason':'non-regular file'});continue
        count=p.stat().st_size;size+=count
        if count>50*1024*1024:findings.append({'file':name,'reason':'over 50 MiB'})
        if p.suffix.lower() in DENIED or p.name in ['demo_data.js','.env'] or p.name.startswith('.env.'):
            findings.append({'file':name,'reason':'forbidden asset/credential format'})
        if p.suffix.lower()=='.png':
            media.append(name)
            if name not in expected or not name.startswith('results/figures/'):
                findings.append({'file':name,'reason':'unreviewed PNG'})
            continue
        data=p.read_bytes()
        try:text=data.decode('utf-8-sig')
        except UnicodeDecodeError:
            findings.append({'file':name,'reason':'unreviewed binary'});continue
        for kind,pattern in PATTERNS.items():
            if re.search(pattern,text):findings.append({'file':name,'reason':'potential '+kind})
        if p.suffix=='.py':
            python_files+=1
            try:ast.parse(text,filename=name)
            except SyntaxError:findings.append({'file':name,'reason':'Python syntax error'})
    docs=[ROOT/'README.md',ROOT/'THIRD_PARTY_NOTICES.md',*sorted((ROOT/'docs').glob('*.md'))]
    for p in docs:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if not link.startswith(('http:','https:','#')) and not (p.parent/link.split('#')[0]).exists():
                findings.append({'file':p.relative_to(ROOT).as_posix(),'reason':'broken relative Markdown link: '+link})
    result={'status':'REVIEW_REQUIRED' if findings else 'PASS_SCOPED_PREFLIGHT','files':len(names),'bytes':size,
        'python_files_parsed':python_files,'reviewed_numerical_charts':len(media),'findings':findings,
        'limits':'Current proposed files only; no full-history secret audit or redistribution-rights clearance.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(1 if findings else 0)

if __name__=='__main__':main()
