import unittest
from anydoor_annotation_probe import annotation_is_releasable


class AnnotationGateTests(unittest.TestCase):
    def test_proposals_never_auto_release(self):
        for status in ('PROPOSAL', 'PROPOSALS_COMPLETE_NOT_GT', 'VERIFIED_ANNOTATION'):
            self.assertFalse(annotation_is_releasable({'status': status, 'confidence': 1.0}, 'i', 'm'))

    def test_verified_decision_binds_pixels_and_semantics(self):
        good = dict(status='VERIFIED_ANNOTATION', image_sha256='i', mask_sha256='m',
            identity_review='PASS', extra_object_review='PASS', mask_review='PASS',
            reviewer_role='human', reviewer='fixture')
        self.assertTrue(annotation_is_releasable(good, 'i', 'm'))
        for key, value in [('image_sha256', 'changed'), ('mask_sha256', 'changed'),
                           ('identity_review', 'FAIL'), ('extra_object_review', 'FAIL'),
                           ('mask_review', 'PENDING'), ('reviewer_role', 'model')]:
            self.assertFalse(annotation_is_releasable({**good, key: value}, 'i', 'm'))


if __name__ == '__main__':
    unittest.main()
