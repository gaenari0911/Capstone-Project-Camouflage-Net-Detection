import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import s8_anydoor_pipeline as pipeline
from s8_anydoor_stages import polygon


class PipelineTests(unittest.TestCase):
    def test_order_and_training_scope(self):
        stages=pipeline.stages()
        self.assertEqual(len(stages),23)
        jobs=[x for x in stages if x.startswith(('train_M','train_L'))]
        self.assertEqual(len(jobs),12)
        self.assertLess(stages.index('audit'),stages.index('selection'))
        self.assertLess(stages.index('train_preflight'),stages.index('train_M2_seed0'))
        self.assertLess(stages.index('freeze'),stages.index('test'))
        self.assertEqual(stages[-1],'report')
        self.assertFalse(any('sam' in x.lower() for x in stages))

    def test_polygon_empty_and_roundtrip(self):
        with self.assertRaises(ValueError):polygon(np.zeros((32,32),np.uint8))
        mask=np.zeros((32,32),np.uint8);mask[4:20,3:18]=1
        label,score=polygon(mask)
        self.assertEqual(score,1.)
        values=[float(x) for x in label.split()]
        self.assertEqual(values[0],0)
        self.assertTrue(all(0<=v<=1 for v in values[1:]))

    def test_completion_detects_changed_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'campaign';base.mkdir()
            file=root/'result.json';file.write_text('{}')
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'BASE',base),patch.object(pipeline,'verify_binding',return_value='binding'):
                self.assertFalse(pipeline.is_done('pilot'))
                pipeline.mark_done('pilot',[file]);self.assertTrue(pipeline.is_done('pilot'))
                file.write_text('{"changed":true}')
                with self.assertRaises(ValueError):pipeline.is_done('pilot')

    def test_missing_dependency_blocks_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(pipeline,'BASE',Path(temp)),patch.object(pipeline,'verify_binding',return_value='binding'),patch.object(pipeline,'is_done',return_value=False):
                with self.assertRaises(ValueError):pipeline.work('test')

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):pipeline.work('unapproved_test')

    def test_start_requires_verified_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'implementation_release.json').write_text(json.dumps({'status':'PENDING'}))
            with patch.object(pipeline,'BASE',root),patch.object(pipeline,'verify_binding',return_value='binding'):
                with self.assertRaises(ValueError):pipeline.start()


if __name__=='__main__':unittest.main()
