import unittest
import json
import tempfile
from pathlib import Path
from fphelper.data import ITEMS, HIGH, LOW
from fphelper.score_contract import score_result, PHASES, read_action_evidence, reviewed_result


class CompletenessTests(unittest.TestCase):
    def test_complete_normal_score_and_partial_action(self):
        labels=dict(zip(ITEMS, [0,0,0]+[5]*5+[0]*5))
        self.assertEqual(score_result(labels,PHASES)['total'],100)
        partial=score_result(labels,PHASES[:-1])
        self.assertIsNone(partial['total'])
        self.assertNotIn('vol_lip_pucker',partial['subscores'])
        self.assertNotIn('syn_lip_pucker',partial['subscores'])
        self.assertEqual(len(partial['subscores']),11)

    def test_missing_label_is_not_zero_or_normal(self):
        labels=dict(zip(ITEMS,LOW))
        labels['syn_smile']=None
        result=score_result(labels,PHASES)
        self.assertIsNone(result['raw_total'])
        self.assertIn('syn_smile',result['missing_subscores'])
        del labels['rest_eye']
        self.assertIsNone(score_result(labels,PHASES)['total'])

    def test_unverified_windows_cannot_yield_total_and_invalid_scores_rejected(self):
        labels=dict(zip(ITEMS,HIGH))
        self.assertIsNone(score_result(labels,[])['total'])
        labels['rest_eye']=True
        with self.assertRaises(ValueError): score_result(labels,PHASES)

    def test_reviewed_output_removes_unverified_legacy_totals(self):
        legacy={'status':'needs_review','subscores':dict(zip(ITEMS,map(int,HIGH))),
                'total':77,'raw_total':77,'rest_penalty':10,'voluntary_score':100,'synkinesis_penalty':13}
        result=reviewed_result(legacy,[])
        self.assertIsNone(result['total'])
        self.assertIsNone(result['raw_total'])
        self.assertNotIn('voluntary_score',result)
        self.assertEqual(result['subscores'],{})
        self.assertEqual(len(result['candidate_subscores']),13)
        self.assertEqual(legacy['total'],77)
        failed=reviewed_result({'status':'ungradable','subscores':None},PHASES)
        self.assertEqual(failed['status'],'ungradable')
        self.assertIsNone(failed['total'])

    def test_evidence_cannot_be_reused_for_another_video_or_protocol(self):
        self.assertEqual(read_action_evidence(None,'a','p'),([],None))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'evidence.json'
            evidence={'schema_version':1,'video_sha256':'a','feature_protocol':'p',
                      'verification_method':'manual review of assigned windows','verified_phases':list(PHASES)}
            path.write_text(json.dumps(evidence))
            self.assertEqual(read_action_evidence(path,'a','p')[0],list(PHASES))
            for video,protocol in [('b','p'),('a','q')]:
                with self.assertRaises(ValueError): read_action_evidence(path,video,protocol)


if __name__ == '__main__': unittest.main()
