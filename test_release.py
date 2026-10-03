import unittest
from moderation03 import apply_policy
from moderation_policy import decisions,PROFILES
from categories import CATEGORIES
import app

class FakeModel:
    calls=0
    def scores(self,text):
        self.calls+=1
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
    def test_live_thresholds_preserve_raw_scores_without_inference(self):
        app.MODEL=FakeModel();controls=app.settings('medium')[1:]
        outputs=app.analyze('test','en','medium',False,*controls)
        preset=outputs[3];cached=outputs[4]
        custom_controls=[v for c in CATEGORIES for v in [False,.99]]
        custom=app.render(cached,'en','medium',False,*custom_controls)[3]
        self.assertEqual(preset['raw_scores'],custom['raw_scores'])
        self.assertTrue(preset['policy']['block'])
        self.assertFalse(custom['policy']['block'])
        self.assertTrue(custom['customized'])
        custom_controls[0]=True;custom_controls[1]=.4
        self.assertTrue(app.render(cached,'en','medium',False,*custom_controls)[3]['policy']['block'])
        custom_controls[1]=.5
        self.assertFalse(app.render(cached,'en','medium',False,*custom_controls)[3]['policy']['block'])
        self.assertEqual(app.MODEL.calls,1)
        self.assertIsNone(app.clear_result()[-1])
    def test_ui_builds(self):
        demo=app.build_app()
        self.assertTrue(demo.config['components'])
        self.assertFalse(any(c['type']=='radio' for c in demo.config['components']))
        for c in demo.config['components']:
            if c['type'] in ['checkbox','slider']:self.assertTrue(c['props']['interactive'])
            label=c['props'].get('label','')
            self.assertFalse(any('\u0400'<=letter<='\u04ff' for letter in label))
    def test_plain_asgi_routes(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import gradio as gr
        mounted=gr.mount_gradio_app(FastAPI(),app.build_app(),path='/',ssr_mode=False)
        with TestClient(mounted) as client:
            self.assertEqual(client.get('/').status_code,200)
            config=client.get('/config').json()
            self.assertFalse(config.get('enable_ssr',False))
            self.assertIn('/moderate',client.get('/gradio_api/info').json()['named_endpoints'])

if __name__=='__main__':unittest.main()
