import unittest
from s8_anydoor_selection import task_score, select


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(id=str(i), valid=True, mask_iou=i/10, confidence=i/10, score=task_score(i/10,i/10),
            domain='snow' if i%2 else 'mixed', scale_bucket='small', reference=str(i), prediction_count=i>0) for i in range(6)]

    def test_score_and_no_prediction(self):
        self.assertEqual(task_score(0, 0), 2)
        self.assertEqual(task_score(1, 1), 0)
        for value in (float('nan'), float('inf'), -1, 2):
            with self.assertRaises(ValueError): task_score(value, .5)

    def test_deterministic_common_pool(self):
        a=select(self.rows(),3); b=select(list(reversed(self.rows())),3)
        self.assertEqual(a,b)
        self.assertEqual(a['task'],['0','1','2'])
        self.assertEqual(len(a['random']),3)

    def test_invalid_duplicate_and_score_tampering(self):
        for rows in (self.rows()+[self.rows()[0]], [{**r,'valid':False} for r in self.rows()],
                     [{**r,'score':-1} for r in self.rows()]):
            with self.assertRaises(ValueError): select(rows,3)


if __name__=='__main__': unittest.main()
