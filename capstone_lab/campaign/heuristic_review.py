"""Build a local review gallery without changing generated images."""
import html
import json
from pathlib import Path
from .contracts import read_json


def build(root, output):
    rows = read_json(output/'samples.json')
    cards = []
    for row in rows:
        image = root/row['image']
        mask = root/row['mask']
        polygons = []
        for line in (root/row['label']).read_text().splitlines():
            points = list(map(float,line.split()[1:]))
            polygons.append('<polygon points="'+ ' '.join(f'{points[i]*1000},{points[i+1]*1000}' for i in range(0,len(points),2))+'"/>')
        title = html.escape(row['tag']+' '+row['sample_id']+' '+json.dumps(row['plan']))
        cards.append(f'<article><h3>{title}</h3><div class="row"><a href="{image.as_uri()}"><img src="{image.as_uri()}"></a><a href="{mask.as_uri()}"><img src="{mask.as_uri()}"></a><div class="overlay"><img src="{image.as_uri()}"><svg viewBox="0 0 1000 1000" preserveAspectRatio="none">'+''.join(polygons)+'</svg></div></div><p>'+html.escape(str(image))+'</p></article>')
    document = '<!doctype html><meta charset="utf-8"><title>Heuristic sample review</title><style>body{font:14px sans-serif;background:#222;color:#eee}article{margin:24px 0}.row{display:flex;gap:8px}.row>a,.overlay{width:32%;position:relative}img{width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%}polygon{fill:none;stroke:#f22;stroke-width:2}h3,p{overflow-wrap:anywhere}</style><h1>Image / mask / polygon overlay</h1><p>User visual approval: PENDING. Open original links for full resolution.</p>'+''.join(cards)
    (output/'review.html').write_text(document,encoding='utf-8')
    print(json.dumps({'samples':len(rows),'gallery':str(output/'review.html')}))


if __name__ == '__main__':
    build(Path.cwd(),Path.cwd()/'artifacts/s8_heuristic_audit/run_v1')
