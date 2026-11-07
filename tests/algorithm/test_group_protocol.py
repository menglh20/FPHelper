import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fphelper.data import LOW
from fphelper.group_protocol import patient_groups, nested_group_protocol


class PatientProtocolTests(unittest.TestCase):
    def test_nested_subjects_never_cross_and_outer_test_stays_out_of_inner(self):
        groups = np.repeat(np.arange(20), 3)
        labels = np.tile(LOW, (len(groups), 1))
        seen = []
        for fold in nested_group_protocol(labels, groups):
            train, test = fold['train_indices'], fold['test_indices']
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen.extend(test)
            for split in fold['inner']:
                a, b = split['train_indices'], split['test_indices']
                self.assertFalse(set(groups[a]) & set(groups[b]))
                self.assertTrue(set(a + b) <= set(train))
        self.assertEqual(sorted(seen), list(range(len(groups))))

    def test_mapping_rejects_missing_duplicate_extra_and_conflicting_hash(self):
        manifest = [{'video_id':'v1','video_sha256':'same'}, {'video_id':'v2','video_sha256':'same'}]
        cases = [[('v1','p1'),('v2','')], [('v1','p1'),('v1','p1')],
                 [('v1','p1'),('v2','p1'),('v3','p1')], [('v1','p1'),('v2','p2')]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mapping.csv'
            for rows in cases:
                with path.open('w',newline='') as f:
                    w=csv.writer(f); w.writerow(['video_id','patient_group_id']); w.writerows(rows)
                with self.assertRaises(ValueError): patient_groups(manifest,path)

    def test_insufficient_subjects_does_not_fall_back_to_videos(self):
        with self.assertRaises(ValueError):
            nested_group_protocol(np.tile(LOW,(30,1)),np.repeat(np.arange(3),10))


if __name__ == '__main__': unittest.main()
