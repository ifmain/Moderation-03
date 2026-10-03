"""English-only UI with always-active editable thresholds and session-local raw scores."""
import json
import os
from pathlib import Path
from threading import Lock
import time
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
os.environ['GRADIO_SSR_MODE']='False'
import gradio as gr
from moderation03 import Moderation03,apply_policy,MODEL_ID
from moderation_policy import LANGS,PROFILES
from categories import CATEGORIES

ROOT=Path(__file__).resolve().parent
CALIBRATION=json.loads((ROOT/'calibration/thresholds.json').read_text(encoding='utf-8'))
MODEL=None
LOCK=Lock()
LABELS={'light':'Light','medium':'Medium','high':'High','corporate':'Corporate'}

def settings(preset):
    values=[]
    for c in CATEGORIES:
        t=CALIBRATION['thresholds'][preset][c]
        values.extend([t is not None,round(t,6) if t is not None else .5])
    return [preset in ['high','corporate'],*values]

def load_model():
    global MODEL
    if MODEL is None:
        MODEL=Moderation03(os.getenv('MODEL_ID',MODEL_ID),device=os.getenv('DEVICE') or None,
                           backbone_path=os.getenv('BACKBONE_PATH') or None)
    return MODEL

def render(cached,language,preset,lexical,*controls):
    if not cached:return 'Enter text and click Analyze.',[],[],{}
    thresholds={c:float(controls[2*i+1]) if controls[2*i] else None for i,c in enumerate(CATEGORIES)}
    raw=cached['prediction']['raw_scores'];text=cached['text']
    policy=apply_policy(raw,thresholds,block_profanity=bool(lexical),text=text,language=language)
    reference=settings(preset)
    customized=bool(lexical)!=reference[0] or any(bool(controls[2*i])!=reference[1+2*i] or
        (controls[2*i] and abs(float(controls[2*i+1])-reference[2+2*i])>1e-8) for i in range(len(CATEGORIES)))
    result={**cached['prediction'],'base_preset':preset,'customized':customized,'language':language,
            'policy':policy,'inference_seconds':cached['seconds'],
            'all_presets':{p:apply_policy(raw,CALIBRATION['thresholds'][p],
                block_profanity=p in ['high','corporate'],text=text,language=language) for p in PROFILES}}
    summary=('## BLOCK' if policy['block'] else '## ALLOW')
    summary+=f'\n\n{LABELS[preset]}'+(' · modified thresholds' if customized else ' preset')
    summary+='\n\nTriggered: '+(', '.join(policy['reasons']) or 'none')
    summary+=f'\n\nModel inference: {cached["seconds"]:.2f}s. Changing thresholds reuses these scores.'
    if result['truncated']:summary+='\n\n⚠ Only the first 512 tokens were analyzed.'
    table=[[c,raw[c],thresholds[c],'Disabled' if thresholds[c] is None else
            ('Triggered' if policy['tags'][c] else 'Not triggered')] for c in CATEGORIES]
    comparison=[[LABELS[p],result['all_presets'][p]['block'],', '.join(result['all_presets'][p]['reasons']) or 'None'] for p in PROFILES]
    return summary,table,comparison,result

def analyze(text,language,preset,lexical,*controls):
    if not text or not text.strip():raise gr.Error('Enter non-empty text.')
    if len(text)>50000:raise gr.Error('Maximum input length: 50,000 characters.')
    with LOCK:
        model=load_model();start=time.perf_counter();prediction=model.scores(text)
        cached={'text':text,'prediction':prediction,'seconds':time.perf_counter()-start}
    return (*render(cached,language,preset,lexical,*controls),cached)

def clear_result():
    return 'Text changed. Click Analyze to update the scores.',[],[],{},None

def build_app():
    with gr.Blocks(title='Moderation 03',analytics_enabled=False) as demo:
        cache=gr.State(value=None,time_to_live=3600)
        gr.Markdown('# Moderation 03\n17 languages · 11 category scores · editable thresholds')
        with gr.Row():
            with gr.Column():
                text=gr.Textbox(label='Text',lines=7,max_lines=14,interactive=True)
                language=gr.Dropdown(LANGS,value='en',label='Text language',info='Used by the optional profanity rule.',interactive=True)
                preset=gr.Dropdown([(LABELS[p],p) for p in PROFILES],value='medium',label='Preset',
                    info='Loads thresholds below. You can edit them directly.',interactive=True)
                button=gr.Button('Analyze',variant='primary')
            with gr.Column():
                summary=gr.Markdown('Enter text and click Analyze.')
                table=gr.Dataframe(headers=['Category','Raw score','Threshold','Tag'],datatype=['str','number','number','str'],interactive=False)
        with gr.Accordion('Category thresholds',open=True):
            gr.Markdown('Every enabled category can block the text. Disable a category with its checkbox. Changes immediately update the decision for the last analyzed text.')
            lexical=gr.Checkbox(value=False,label='Block profanity using the word list',interactive=True)
            controls=[];defaults=settings('medium')[1:]
            for i,c in enumerate(CATEGORIES):
                with gr.Row():
                    enabled=gr.Checkbox(value=defaults[2*i],label=c,interactive=True)
                    threshold=gr.Slider(0,1,value=defaults[2*i+1],step=.001,label='Threshold',interactive=True)
                controls.extend([enabled,threshold])
        comparison=gr.Dataframe(headers=['Preset','Blocked','Reasons'],datatype=['str','bool','str'],interactive=False,label='Unmodified presets for comparison')
        details=gr.JSON(label='Raw scores and decisions')
        outputs=[summary,table,comparison,details]
        refresh_inputs=[cache,language,preset,lexical,*controls]
        preset.change(settings,inputs=preset,outputs=[lexical,*controls],queue=False).then(
            render,inputs=refresh_inputs,outputs=outputs,queue=False,api_name=False)
        for control in [language,lexical,*controls]:
            control.input(render,inputs=refresh_inputs,outputs=outputs,queue=False,api_name=False)
        text.input(clear_result,outputs=[*outputs,cache],queue=False,api_name=False)
        button.click(analyze,inputs=[text,language,preset,lexical,*controls],outputs=[*outputs,cache],
                     concurrency_limit=1,api_name='moderate')
        gr.Markdown('Raw scores are not calibrated probabilities. Presets classify content; they do not determine whether a topic belongs in a workplace. Text stays in session memory for interactive threshold changes and is not saved to a dataset.')
        gr.Markdown('[Model](https://huggingface.co/ifmain/Moderation-03) · [Evaluation](https://github.com/ifmain/Moderation-03/tree/main/evaluation) · [Source](https://github.com/ifmain/Moderation-03)')
    return demo.queue(max_size=8)

demo=build_app()

if __name__=='__main__':
    from fastapi import FastAPI
    import uvicorn
    print('Loading model before accepting requests...',flush=True)
    load_model()
    web=gr.mount_gradio_app(FastAPI(),demo,path='/',ssr_mode=False)
    uvicorn.run(web,host=os.getenv('GRADIO_SERVER_NAME','0.0.0.0' if os.getenv('SPACE_ID') else '127.0.0.1'),
                port=int(os.getenv('PORT','7860')),reload=False)
