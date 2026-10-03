import unittest
from moderation03 import apply_policy
from moderation_policy import decisions,PROFILES
from categories import CATEGORIES
import app

class FakeModel:
    def scores(self,text):
        return {'raw_scores':{c:(.45 if c=='harassment' else 0.) for c in CATEGORIES},
                'input_tokens':4,'processed_tokens':4,'truncated':False}

class ReleaseTests(unittest.TestCase):
    def test_threshold_boundary_and_disabled(self):
        scores=dict.fromkeys(CATEGORIES,0.);scores['harassment']=.5
        t=dict.fromkeys(CATEGORIES,None);t['harassment']=.5
        self.assertTrue(apply_policy(scores,t)['block'])
        t['harassment']=.501
        self.assertFalse(apply_policy(scores,t)['block'])
        t['harassment']=None
        self.assertFalse(apply_policy(scores,t)['block'])
    def test_presets_match_evaluated_policy(self):
        for value in [0,.01,.1,.5,1]:
            scores=dict.fromkeys(CATEGORIES,value)
            expected=decisions(scores,True,app.CALIBRATION['thresholds'])
            for p in PROFILES:
                actual=apply_policy(scores,app.CALIBRATION['thresholds'][p],block_profanity=p in ['high','corporate'],text='fuck',language='en')
                self.assertEqual(actual['block'],expected[p]['block'])
                self.assertEqual(actual['reasons'],expected[p]['reasons'])
    def test_ui_modes_preserve_raw_scores(self):
        app.MODEL=FakeModel();controls=app.settings('medium')[1:]
        raw=app.analyze('test','en','raw','medium',False,*controls)[3]
        preset=app.analyze('test','en','preset','medium',False,*controls)[3]
        custom_controls=[v for c in CATEGORIES for v in [False,.99]]
        custom=app.analyze('test','en','custom','medium',False,*custom_controls)[3]
        self.assertEqual(raw['raw_scores'],preset['raw_scores'])
        self.assertEqual(raw['raw_scores'],custom['raw_scores'])
        self.assertIsNone(raw['policy'])
        self.assertTrue(preset['policy']['block'])
        self.assertFalse(custom['policy']['block'])
    def test_ui_builds(self):
        demo=app.build_app()
        self.assertTrue(demo.config['components'])

if __name__=='__main__':unittest.main()
