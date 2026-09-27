"""Shape-channel repair regressions; never substitutes for semantic review."""
import unittest
import numpy as np
from anydoor_runtime import apply_shape_control, activate, load_pair_functions
import anydoor_pilot as pilot


class ShapeTests(unittest.TestCase):
    def test_only_fourth_channel_changes(self):
        mask = np.zeros((9, 9), np.uint8)
        mask[1:7, 4] = 1
        hint = np.ones((9, 9, 4), np.float32)
        item = {'hint': hint, 'extra_sizes': [9, 9, 9, 9], 'tar_box_yyxx_crop': [0, 9, 0, 9]}
        actual = apply_shape_control(item, mask)
        np.testing.assert_array_equal(actual['hint'][:, :, :3], hint[:, :, :3])
        np.testing.assert_array_equal(actual['hint'][:, :, 3], mask)
        np.testing.assert_array_equal(item['hint'], np.ones((9, 9, 4)))

    def test_padding_unknown_and_nearest(self):
        mask = np.ones((3, 5), np.uint8)
        item = {'hint': np.zeros((5, 5, 4)), 'extra_sizes': [3, 5, 5, 5], 'tar_box_yyxx_crop': [0, 3, 0, 5]}
        result = apply_shape_control(item, mask)['hint'][:, :, 3]
        np.testing.assert_array_equal(result[1:4], mask)
        self.assertTrue((result[[0, 4]] == -1).all())

    def test_invalid_masks_rejected(self):
        for mask in (np.zeros((3, 3)), np.ones((3, 3)) * 255, np.ones((3, 3, 3))):
            with self.assertRaises(ValueError):
                apply_shape_control({}, mask)

    def test_real_pairs_condition_only_change(self):
        activate(pilot.SOURCE)
        process, _ = load_pair_functions(pilot.SOURCE)
        records = pilot.read(pilot.ROOT / 'artifacts/s8_anydoor_shape_probe/run_v1/manifest.json')['records']
        for row in records:
            args = pilot.inputs(row)
            np.random.seed(row['seed'])
            item = process(*args)
            fixed = apply_shape_control(item, args[3])
            np.testing.assert_array_equal(item['hint'][:, :, :3], fixed['hint'][:, :, :3])
            self.assertGreater(np.count_nonzero(item['hint'][:, :, 3] != fixed['hint'][:, :, 3]), 0)
            self.assertTrue(set(np.unique(fixed['hint'][:, :, 3])).issubset({-1, 0, 1}))


if __name__ == '__main__':
    unittest.main()
