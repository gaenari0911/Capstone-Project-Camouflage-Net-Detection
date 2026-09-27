from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.data.dataset import generate_boundary_and_distance_map
from ultralytics.models.yolo.segment.val import SegmentationValidator

from capstone_lab.campaign.data import effective_args, manifest_records, model_batch
from capstone_lab.errors import ArtifactError, ConfigError

ROOT = Path(__file__).resolve().parents[1]


class S8RealDataContracts(unittest.TestCase):
    def test_test_split_rejected_and_frozen_args(self):
        with self.assertRaises(ConfigError):
            manifest_records(ROOT, "test")
        args = effective_args(ROOT)
        self.assertEqual((args.batch, args.imgsz, args.mask_ratio, args.overlap_mask), (16,640,1,False))
        self.assertEqual((args.mosaic, args.mixup, args.copy_paste, args.cutmix), (0,0,0,0))
        self.assertEqual(args.nbs, 16)

    def test_multi_instance_empty_masks_and_H1_distance_removal(self):
        masks = torch.zeros(3, 64, 64)
        masks[0, 8:24, 8:24] = 1
        masks[1, 30:50, 30:50] = 1
        boundaries, distance = generate_boundary_and_distance_map(masks)
        batch = {"img": torch.zeros(2,3,64,64,dtype=torch.uint8), "masks": masks,
                 "boundaries": boundaries, "distance_maps": distance,
                 "batch_idx": torch.tensor([0,0,1])}
        h2 = model_batch(batch, "H2", "cpu")
        h1 = model_batch(batch, "H1", "cpu")
        self.assertTrue(torch.equal(h1["boundaries"], h2["boundaries"]))
        self.assertTrue(torch.equal(h1["distance_maps"], torch.ones_like(distance)))
        self.assertGreater(float(h2["distance_maps"][0].max()), 1)
        self.assertEqual(int(boundaries[2].sum()), 0)
        missing = dict(batch)
        del missing["boundaries"]
        with self.assertRaises(ArtifactError):
            model_batch(missing, "H2", "cpu")

    def test_full_mask_matching_duplicate_prediction_and_empty_cases(self):
        base = ROOT / "artifacts/s8_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            validator = SegmentationValidator(save_dir=Path(directory), args={"task":"segment","plots":False})
            validator.device = torch.device("cpu")
            validator.training = True
            validator.data = {"val": "fixture-val"}
            validator.init_metrics(SimpleNamespace(names={0: "camouflage"}, end2end=False))
            target = {"cls":torch.zeros(1), "bboxes":torch.tensor([[0.,0.,4.,4.]]), "masks":torch.ones(1,4,4)}
            pred = {"cls":torch.zeros(2), "bboxes":torch.tensor([[0.,0.,4.,4.],[0.,0.,4.,4.]]),
                    "masks":torch.ones(2,4,4), "conf":torch.tensor([.9,.8])}
            stats = validator._process_batch(pred, target)
            self.assertEqual(stats["tp_m"].shape, (2,10))
            self.assertTrue(np.all(stats["tp_m"].sum(axis=0) == 1))
            empty = {k:v[:0] for k,v in pred.items()}
            self.assertEqual(validator._process_batch(empty, target)["tp_m"].shape, (0,10))
            no_gt = {k:v[:0] for k,v in target.items()}
            self.assertFalse(validator._process_batch(pred, no_gt)["tp_m"].any())


if __name__ == "__main__":
    unittest.main()
