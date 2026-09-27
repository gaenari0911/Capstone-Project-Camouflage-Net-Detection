import unittest
from pathlib import Path
from unittest.mock import patch

from capstone_lab.campaign.sources import audit_split_sources
from capstone_lab.campaign.contracts import read_json
from capstone_lab.errors import ArtifactError

ROOT = Path(__file__).resolve().parents[1]


class SourcePolicyTests(unittest.TestCase):
    def test_approved_mapping_and_counts(self):
        a = read_json(ROOT / 'configs/approvals/s8_foreground_amendment_v1.json')
        self.assertEqual(a['status'], 'APPROVED_BY_USER')
        self.assertEqual(a['foreground_subsets'], dict(A='train748', M='train748', H='train748', L='low187'))
        self.assertEqual(sum(a['heuristic'].values()) + sum(a['generative_candidates'].values()), 33000)
        self.assertEqual(a['training_jobs'], 34)
        self.assertFalse(a['val_test_foreground_allowed'])

    def test_source_output_guard(self):
        for output in (ROOT / 'Dataset/test', ROOT / 'artifacts/s8_source_audit'):
            with self.assertRaises(ArtifactError):
                audit_split_sources(ROOT, output)

    def test_wrong_policy_rejected_before_audit(self):
        with patch('capstone_lab.campaign.contracts.read_json', return_value={'foreground_subsets': {'L': 'train748'}}):
            with self.assertRaises(ArtifactError):
                audit_split_sources(ROOT, ROOT / 'artifacts/s8_source_audit/unused_fixture')

    def test_diversity_not_false_completion(self):
        a = read_json(ROOT / 'configs/approvals/s8_low_diversity_v1.json')
        self.assertEqual(a['foreground_subset'], 'low187')
        self.assertEqual(a['low_heuristic_images'], 3000)
        self.assertEqual(a['implementation_status'], 'PENDING_IMPLEMENTATION_AND_VALIDATION')
        self.assertFalse(a['generation_authorized_by_this_record'])


if __name__ == '__main__':
    unittest.main()
