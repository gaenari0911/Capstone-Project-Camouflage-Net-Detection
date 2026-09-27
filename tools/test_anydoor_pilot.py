"""CPU safety regressions; separate from semantic pilot quality approval."""
import json
from pathlib import Path
import tempfile
import unittest

import anydoor_pilot as pilot
from anydoor_runtime import sha256


class PilotSafetyTests(unittest.TestCase):
    def test_immutable_json_refuses_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            pilot.atomic_json(path, {'seed': 1}, immutable=True)
            pilot.atomic_json(path, {'seed': 1}, immutable=True)
            with self.assertRaises(Exception):
                pilot.atomic_json(path, {'seed': 2}, immutable=True)
            self.assertEqual(pilot.read(path), {'seed': 1})

    def test_lock_rejects_concurrent_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            with pilot.RunLock(Path(temp) / 'gpu.lock'):
                with self.assertRaises(Exception):
                    with pilot.RunLock(Path(temp) / 'gpu.lock'):
                        pass

    def test_resume_hash_checks_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory / 'output.json'
            pilot.atomic_json(output, {'sample': 1})
            before = output.stat().st_mtime_ns
            pilot.atomic_json(directory / 'COMMITTED.json', {'binding': 'a', 'files': {'output.json': sha256(output)}})
            self.assertTrue(pilot.committed(directory, 'a'))
            self.assertEqual(output.stat().st_mtime_ns, before)
            with self.assertRaises(ValueError):
                pilot.committed(directory, 'b')
            pilot.atomic_json(output, {'sample': 2})
            with self.assertRaises(ValueError):
                pilot.committed(directory, 'a')

    def test_partial_directory_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(pilot.committed(Path(temp), 'a'))

    def test_real_pilot_provenance_and_inputs(self):
        import numpy as np
        manifest = pilot.read(pilot.ROOT / 'artifacts/s8_anydoor_pilot/run_v1/manifest.json')
        records = manifest['records']
        self.assertEqual(len(records), 100)
        self.assertEqual(len({r['id'] for r in records}), 100)
        for subset in ('low187', 'train748'):
            rows = [r for r in records if r['subset'] == subset]
            self.assertEqual(len(rows), 50)
            self.assertEqual(len({r['domain'] for r in rows}), 6)
            allowed = {r['object_image'] for r in pilot.read(pilot.ROOT / rows[0]['provenance_path'])['records']}
            self.assertTrue(all(r['reference'] in allowed for r in rows))
        for row in records[::10]:
            reference, mask, background, target = pilot.inputs(row)
            self.assertEqual(reference.shape[:2], mask.shape)
            self.assertEqual(background.shape[:2], target.shape)
            self.assertEqual(max(background.shape[:2]), 1280)
            self.assertTrue(np.any(target))
            self.assertTrue(set(np.unique(target)).issubset({0, 1}))
        self.assertEqual(manifest['annotation_status'], 'INPUT_PLACEMENT_ONLY_NOT_VERIFIED_GT')
        self.assertIsNone(manifest['time_ceiling_hours'])

    def test_square_region_probe_is_square_and_bounded(self):
        import numpy as np
        manifest = pilot.read(pilot.ROOT / 'artifacts/s8_anydoor_square_probe/run_v1/manifest.json')
        self.assertEqual(len(manifest['records']), 12)
        for row in manifest['records']:
            _, _, background, target = pilot.inputs(row, 'square_region')
            y, x = np.where(target)
            self.assertEqual(y.max() - y.min(), x.max() - x.min())
            self.assertEqual(int(target.sum()), int((y.max() - y.min() + 1) ** 2))
            self.assertEqual(target.shape, background.shape[:2])


if __name__ == '__main__':
    unittest.main()
