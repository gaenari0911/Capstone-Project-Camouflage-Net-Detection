import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import tempfile
from capstone_lab.campaign.contracts import read_json, readiness_plan
from capstone_lab.campaign.io import atomic_json
from capstone_lab.synthesis.full import plans, load_sources, resource_guard, execute, POLICY
from capstone_lab.synthesis.contract import CandidateFeature, CandidatePlan
from capstone_lab.synthesis.scoring import UsageState, select_candidate

ROOT=Path(__file__).resolve().parents[1]


class FullSynthesisTests(unittest.TestCase):
    def test_reader_retries_only_transient_access_errors(self):
        import json
        from capstone_lab.campaign.contracts import read_json
        with patch.object(Path,'read_text',side_effect=[PermissionError('sharing'),PermissionError('sharing'),'{"ok":true}']) as reader:
            self.assertEqual(read_json(Path('fixture.json')),{'ok':True})
            self.assertEqual(reader.call_count,3)
        with patch.object(Path,'read_text',return_value='{broken') as reader:
            with self.assertRaises(json.JSONDecodeError):
                read_json(Path('fixture.json'))
            self.assertEqual(reader.call_count,1)

    def test_effective_dag_split(self):
        p=readiness_plan(ROOT)
        self.assertEqual(p['formal_images'],33000)
        self.assertEqual(p['heuristic_images'],21000)
        nodes={n['node_id']:n for n in p['nodes']}
        for seed in range(3):
            self.assertIn('validate_L_A1_3000',nodes['L1_seed'+str(seed)]['dependencies'])
            self.assertNotIn('validate_A1_3000',nodes['L1_seed'+str(seed)]['dependencies'])
            self.assertIn('select_generative_random_3000_L',nodes['L2_seed'+str(seed)]['dependencies'])
        self.assertNotIn('generate_generative_pool_6000',nodes)

    def test_complete_plan_capacity_and_determinism(self):
        policy=read_json(ROOT/POLICY)
        bg={r['background_id']:r for r in read_json(ROOT/'artifacts/s8_full_backgrounds/v1/manifest.json')['backgrounds']}
        for subset,expected in [('train748',748),('low187',187)]:
            proof=read_json(ROOT/('artifacts/s8_source_audit/split_pools_v1/'+subset+'_object_provenance.json'))
            fg={Path(r['object_image']).stem:r for r in proof['records']}
            self.assertEqual(len(fg),expected)
            a,q=plans(ROOT,subset,fg,bg,policy)
            self.assertEqual((a,q),plans(ROOT,subset,fg,bg,policy))
            self.assertEqual(len(a),3000)
            self.assertEqual(len({r['sample_id'] for r in a}),3000)
            self.assertEqual(sum(r['mode']=='natural' for r in a),2700)
            self.assertEqual(sum(r['difficulty']=='hard' for r in a),1050)
            self.assertTrue(all(r['capacity']>=r['target'] for r in q.values()))
            if subset=='low187':
                self.assertEqual(q['snow']['ceiling'],30)
                self.assertEqual(q['non_snow']['ceiling'],15)

    def test_formal_launch_blocked(self):
        with self.assertRaisesRegex(ValueError,'preflight only'):
            execute(ROOT,ROOT/'artifacts/s8_heuristic_preflight/not_created','low187',1,3000)

    def test_low_full_schedule_balanced_assignment_capacity(self):
        from collections import Counter
        import hashlib
        policy=read_json(ROOT/POLICY)
        bg={r['background_id']:r for r in read_json(ROOT/'artifacts/s8_full_backgrounds/v1/manifest.json')['backgrounds']}
        fg={Path(r['object_image']).stem:r for r in read_json(ROOT/'artifacts/s8_source_audit/split_pools_v1/low187_object_provenance.json')['records']}
        schedule,quota=plans(ROOT,'low187',fg,bg,policy)
        used=Counter()
        seen=set()
        for row in schedule:
            env='snow' if row['domain']=='snow' else 'non_snow'
            ids=[k for k,r in fg.items() if r['environment']==env]
            chosen=row['assigned_foreground']
            self.assertIn(chosen,ids)
            self.assertNotIn((chosen,row['background_id']),seen)
            used[chosen]+=1
            seen.add((chosen,row['background_id']))
        self.assertEqual(sum(used.values()),3000)
        for env in ('snow','non_snow'):
            values=[used[k] for k,r in fg.items() if r['environment']==env]
            self.assertLessEqual(max(values)-min(values),1)
            self.assertLessEqual(max(values),quota[env]['ceiling'])

    def test_low_dynamic_choices_do_not_greedily_deadend(self):
        from capstone_lab.synthesis.full import balanced_options
        policy=read_json(ROOT/POLICY)
        bg={r['background_id']:r for r in read_json(ROOT/'artifacts/s8_full_backgrounds/v1/manifest.json')['backgrounds']}
        fg={Path(r['object_image']).stem:r for r in read_json(ROOT/'artifacts/s8_source_audit/split_pools_v1/low187_object_provenance.json')['records']}
        schedule,quota=plans(ROOT,'low187',fg,bg,policy)
        state=UsageState(tuple(sorted(fg)),max_usage=30)
        for row in schedule:
            choices=balanced_options(schedule,fg,state,row,row['sample_id'],limit=1)
            self.assertTrue(choices,row['sample_id'])
            state.commit(choices[0],row['domain'],row['background_id'])
        self.assertEqual(sum(state.used_total.values()),3000)
        for env in ('snow','non_snow'):
            values=[state.used_total[k] for k,r in fg.items() if r['environment']==env]
            self.assertLessEqual(max(values)-min(values),1)

    def test_wrong_pool_binding_rejected(self):
        from capstone_lab.synthesis import full
        original=full.read_json
        def swapped(path):
            value=original(path)
            if str(path).endswith('split_source_summary.json'):
                value['pools']['low187']=value['pools']['train748']
            return value
        with patch.object(full,'read_json',side_effect=swapped):
            with self.assertRaisesRegex(ValueError,'wrong source budget'):
                load_sources(ROOT,'low187')

    def test_commit_pressure_stops_before_dispatch(self):
        with patch('capstone_lab.synthesis.full.memory',return_value={
                'available_bytes':16*2**30,'commit_limit':36*2**30,
                'commit_used':35*2**30,'commit_fraction':35/36}):
            with self.assertRaisesRegex(ValueError,'COMMIT_SAFE_WAIT'):
                resource_guard(ROOT,read_json(ROOT/POLICY),1,disk=False)

    def test_A0_does_not_call_score(self):
        p=CandidatePlan('x','b','natural','snow','medium','05to10',.08,('f',),('p',),'1','2',0)
        f=CandidateFeature('x','f@p','f','p',.5,.5,.5,.5,.3,True,())
        with patch('capstone_lab.synthesis.scoring.score_candidate',side_effect=AssertionError):
            selected,meta=select_candidate('A0',p,[f],UsageState(('f',),max_usage=0),7)
        self.assertIsNone(meta['final_score'])
        self.assertEqual(selected,f)


if __name__=='__main__':
    unittest.main()
