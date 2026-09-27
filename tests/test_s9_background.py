import json,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))

class BackgroundRegression(unittest.TestCase):
    def test_negative_and_mixed_collate(self):
        import torch
        from s9_background_data import BackgroundDataset
        def item(n):
            masks=torch.zeros((n,16,16));masks[:,4:12,4:12]=1
            return dict(img=torch.zeros((3,16,16)),masks=masks,cls=torch.zeros((n,1)),bboxes=torch.zeros((n,4)),batch_idx=torch.zeros(n))
        for inputs,total in [([item(0),item(0)],0),([item(1),item(0)],1)]:
            batch=BackgroundDataset.collate_fn(inputs)
            self.assertEqual(tuple(batch['masks'].shape),(total,16,16))
            self.assertEqual(batch['boundaries'].shape,batch['masks'].shape)
            self.assertEqual(batch['distance_maps'].shape,batch['masks'].shape)

    def test_report_fixture_isolated(self):
        from s9_background_report import generate
        with tempfile.TemporaryDirectory(prefix='s9_report_fixture_') as tmp:
            root=Path(tmp);base=root/'artifacts/s9_background/run_v1'
            old=root/'artifacts/s8_anydoor_campaign/run_v1/test';old.mkdir(parents=True)
            (base/'test').mkdir(parents=True)
            original=root/'artifacts/final_presentation/end_to_end_v2';original.mkdir(parents=True)
            for d in ['figures','tables']:(original/d).mkdir()
            for name in ['END_TO_END_REPORT_KO.md','FIGURE_INDEX.md','CHATGPT_HANDOFF_KO.md']:(original/name).write_text('TEST FIXTURE ONLY',encoding='utf-8')
            (original/'index.html').write_text('<html>TEST FIXTURE</html>',encoding='utf-8')
            def record(exp,seed):
                return dict(job=f'{exp}_seed{seed}',experiment=exp,seed=seed,metrics=dict(library_metrics={'metrics/mAP50-95(M)':.2+seed*.01,'metrics/mAP50(M)':.4},mask_precision=.5,mask_recall=.4,mask_f1=4/9,mask_fp=12,mask_fn=6,mask_tp=4,images=200))
            for path,exps in [(old/'summary.json',['M0','M1']),(base/'test/summary.json',['BG0','BG1'])]:
                path.write_text(json.dumps({'records':[record(e,s) for e in exps for s in range(3)]}),encoding='utf-8')
            outputs=generate(root,base)
            self.assertTrue(all(p.exists() for p in outputs))
            report=(base/'report/BACKGROUND_REPORT_KO.md').read_text(encoding='utf-8')
            self.assertIn('평균 mAP 개선은 관찰되지 않았다',report)
            with zipfile.ZipFile(outputs[-1]) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('figures/17_background_supplement.png',z.namelist())
            self.assertEqual((original/'END_TO_END_REPORT_KO.md').read_text(encoding='utf-8'),'TEST FIXTURE ONLY')

if __name__=='__main__':unittest.main()
