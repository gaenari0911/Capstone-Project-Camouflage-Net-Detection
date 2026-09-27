"""Production-release guards plus one preserved, actual FULL fault acceptance run."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import unittest

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError
from capstone_lab.campaign.contracts import read_json
from capstone_lab.campaign.io import atomic_json
from capstone_lab.synthesis import full
from capstone_lab.synthesis.formal import (
    _initial_state, _ledger_tick, _owned, _verify_baseline, _resource_state,
)


ROOT=Path(__file__).resolve().parents[1]
from capstone_lab.campaign.contracts import digest
ACCEPTANCE=ROOT/'artifacts/s8_heuristic_preflight'/('formal_acceptance_'+digest(
    {p.name:sha256_file(p) for p in (ROOT/'capstone_lab/synthesis').glob('*.py')})[:12])


class FormalGuardTests(unittest.TestCase):
    def test_formal_config_mutation_is_rejected(self):
        from capstone_lab.synthesis import formal
        base={'status':'APPROVED_FORMAL_HEURISTIC21000','release':'release.json',
              'output':formal.FORMAL_ROOT.as_posix(),'workers':2,'cpu_heavy_max':6,'io_max':2,
              'jobs':[{'id':'full','subset':'train748','count':3000},
                      {'id':'low','subset':'low187','count':3000}]}
        release={'release_sha256':'fixture','campaign_contract':base}
        import copy
        for key in ('workers','jobs','output'):
            changed=copy.deepcopy(base);changed['release_sha256']='fixture'
            if key=='workers': changed[key]=6
            elif key=='jobs': changed[key][1]['subset']='train748'
            else: changed[key]='artifacts/s8_heuristic_formal/other'
            with patch.dict('os.environ',{},clear=True), patch.object(formal,'read_json',return_value=changed), patch.object(
                    formal,'verify_release',return_value=(release,ROOT)):
                with self.assertRaisesRegex(ArtifactError,'campaign config changed'):
                    formal._campaign(ROOT)

    def test_runtime_bootstrap_has_spawn_guard(self):
        from capstone_lab.synthesis.formal import _bootstrap_text
        import ast
        tree=ast.parse(_bootstrap_text())
        self.assertIsInstance(tree.body[1],ast.If)

    def test_original_dataset_fingerprint_is_unchanged(self):
        from capstone_lab.manifests import tree_fingerprint
        value=tree_fingerprint(ROOT/'Dataset',workers=1)
        self.assertEqual(value['sha256'],'8ff188ae4788961b00221e0452a5e38da0aa345d2ea1ffed740cfa001902b197')
        self.assertEqual(value['file_count'],6009)

    def test_lease_charges_crash_window_and_refunds_only_normal_exit(self):
        import tempfile
        from capstone_lab.synthesis.formal import _Ticker
        base=ROOT/'artifacts/s8_formal_tests';base.mkdir(parents=True,exist_ok=True)
        run=Path(tempfile.mkdtemp(dir=base))
        release={'release_sha256':'lease-fixture','prior_real_active_seconds':100.0}
        first=_Ticker(ROOT,run,release)
        first.__enter__()
        charged=read_json(run/'active_time_ledger.json')['formal_active_seconds']
        self.assertEqual(charged,600)
        # Simulate a dead owner: no __exit__. A replacement must retain its charge.
        with _Ticker(ROOT,run,release):
            self.assertGreaterEqual(read_json(run/'active_time_ledger.json')['formal_active_seconds'],1200)
        final=read_json(run/'active_time_ledger.json')
        self.assertGreaterEqual(final['formal_active_seconds'],600)
        self.assertFalse(final['active'])

    def test_claimed_formal_status_is_not_a_release(self):
        with patch('capstone_lab.synthesis.formal.verify_release',side_effect=ArtifactError('missing evidence')):
            with self.assertRaisesRegex(ArtifactError,'missing evidence'):
                full.execute(ROOT,ROOT/'artifacts/s8_heuristic_formal/forged','train748',2,3000,
                    formal_release={'status':'VERIFIED_FORMAL_GENERATOR'},code_root=ROOT)

    def test_baseline_hashes_except_reviewed_formal_extension(self):
        result=_verify_baseline(ROOT)
        self.assertTrue(result['pairs']['train748']['worker_output_hashes_equal'])
        self.assertTrue(result['pairs']['low187']['worker_output_hashes_equal'])

    def test_preflight_limit_is_not_a_formal_bypass(self):
        with self.assertRaisesRegex(ValueError,'preflight only'):
            full.execute(ROOT,ROOT/'artifacts/s8_heuristic_preflight/refused_formal',
                         'train748',2,3000)
        with self.assertRaisesRegex(ValueError,'exactly3000'):
            full.execute(ROOT,ROOT/'artifacts/s8_heuristic_formal/refused_small',
                         'train748',2,1,formal_release={'status':'VERIFIED_FORMAL_GENERATOR'},
                         code_root=ROOT)

    def test_campaign_state_is_exactly_two_ordered_jobs(self):
        config={'jobs':[{'id':'train748_A0_A5_3000'},{'id':'low187_A1_3000'}]}
        state=_initial_state(config)
        self.assertEqual(list(state['jobs']),['train748_A0_A5_3000','low187_A1_3000'])
        self.assertTrue(all(v['state']=='PLANNED' for v in state['jobs'].values()))

    def test_durable_ledger_accumulates_and_binds_release(self):
        run=ROOT/'artifacts/s8_formal_tests/ledger_v1'
        run.mkdir(parents=True,exist_ok=True)
        release={'release_sha256':'ledger-fixture','prior_real_active_seconds':29.5}
        # A unique fixture path is intentionally reusable; establish its immutable identity.
        if (run/'active_time_ledger.json').exists():
            ledger=read_json(run/'active_time_ledger.json')
            self.assertEqual(ledger['release_sha256'],'ledger-fixture')
        before=read_json(run/'active_time_ledger.json')['formal_active_seconds'] if (run/'active_time_ledger.json').exists() else 0
        first=_ledger_tick(ROOT,run,release,1.25,True)
        second=_ledger_tick(ROOT,run,release,.75,False)
        self.assertAlmostEqual(second['formal_active_seconds']-before,2.0)
        self.assertAlmostEqual(second['total_active_seconds'],29.5+second['formal_active_seconds'])
        self.assertGreater(second['ticks'],first['ticks'])
        with self.assertRaisesRegex(ArtifactError,'ledger release mismatch'):
            _ledger_tick(ROOT,run,{'release_sha256':'different','prior_real_active_seconds':0})

    def test_campaign_resource_gate_fails_closed(self):
        run=ROOT/'artifacts/s8_formal_tests/resource_v1';run.mkdir(parents=True,exist_ok=True)
        release={'release_sha256':'resource-fixture','prior_real_active_seconds':604800.0}
        with self.assertRaisesRegex(ArtifactError,'TIME_BUDGET_STOP'):
            _resource_state(ROOT,run,release,2)

    def test_foreign_pid_is_not_owned(self):
        import os,psutil
        record={'pid':os.getpid(),'created':psutil.Process().create_time(),
                'token':'not-on-command','module':'capstone_lab.synthesis.formal'}
        self.assertIsNone(_owned(record))

    def test_actual_v2_worker_and_coordinator_fault_evidence_is_intact(self):
        run=ROOT/'artifacts/s8_heuristic_preflight/full_w2_v2'
        worker=read_json(run/'fault_probe.json')
        coordinator=read_json(run/'coordinator_fault_probe.json')
        report=read_json(run/'report.json')
        self.assertEqual(worker['injected'],'worker_tree_termination')
        self.assertIn('owner',coordinator)
        self.assertEqual(report['status'],'VERIFIED_SMALL_RENDER')
        self.assertEqual(report['counts'],{c:2 for c in ('A0','A1','A2','A3','A4','A5')})


class ActualFullAcceptance(unittest.TestCase):
    def test_partial_pause_tamper_and_completed_reuse(self):
        """Preserve this expensive actual FULL evidence and verify it on later runs."""
        output=ACCEPTANCE
        output.mkdir(parents=True,exist_ok=True)
        final=output/'samples/train748_000001'
        evidence_path=output/'formal_acceptance_evidence.json'
        control=output/'acceptance_control.json'
        if not (final/'PAIR.json').exists():
            atomic_json(control,{'action':'pause'})
            paused=full.execute(ROOT,output,'train748',2,1,pause_path=control)
            self.assertEqual(paused['status'],'PAUSED')
            atomic_json(control,{'action':'run'})
            with self.assertRaisesRegex(InterruptedError,'paired staging'):
                full.execute(ROOT,output,'train748',2,1,
                             pause_path=control,interrupt_stage='after_first_condition')
            self.assertFalse((final/'PAIR.json').exists())
            pending=list((output/'.pending').glob('*/samples/train748_000001/*/COMMITTED.json'))
            self.assertEqual(len(pending),1)
            report=full.execute(ROOT,output,'train748',2,1,pause_path=control)
            self.assertEqual(report['counts'],{c:1 for c in ('A0','A1','A2','A3','A4','A5')})
            hashes={str(p.relative_to(output)):sha256_file(p)
                    for p in final.rglob('*') if p.is_file()}
            mtimes={str(p.relative_to(output)):p.stat().st_mtime_ns
                    for p in final.rglob('*') if p.is_file()}
            atomic_json(evidence_path,{'hashes':hashes,'mtimes':mtimes,
                        'partial_committed_bundles':len(pending)},immutable=True)
        evidence=read_json(evidence_path)
        before={p:(sha256_file(output/p),(output/p).stat().st_mtime_ns)
                for p in evidence['hashes']}
        reused=full.execute(ROOT,output,'train748',2,1,pause_path=control)
        self.assertEqual(reused['status'],'VERIFIED_SMALL_RENDER')
        for relative,(file_hash,mtime) in before.items():
            self.assertEqual((sha256_file(output/relative),(output/relative).stat().st_mtime_ns),
                             (file_hash,mtime))
        original=full.read_json
        def changed_policy(path):
            value=original(path)
            if str(path).endswith('s8_heuristic_policy_v1.json'):
                value=dict(value);value['seed']=8
            return value
        with patch.object(full,'read_json',side_effect=changed_policy):
            with self.assertRaisesRegex(Exception,'immutable artifact differs'):
                full.execute(ROOT,output,'train748',2,1,pause_path=control)
        real_hash=full.sha256_file
        def changed_input(path):
            return '0'*64 if str(path).endswith('s8_heuristic_policy_v1.json') else real_hash(path)
        with patch.object(full,'sha256_file',side_effect=changed_input):
            with self.assertRaisesRegex(Exception,'immutable artifact differs'):
                full.execute(ROOT,output,'train748',2,1,pause_path=control)


if __name__=='__main__': unittest.main()
