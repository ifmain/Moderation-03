"""Gradio demo: raw outputs, frozen presets, and independent custom category thresholds."""
import json
import os
from pathlib import Path
from threading import Lock
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
import gradio as gr
from moderation03 import Moderation03,apply_policy,MODEL_ID
from moderation_policy import LANGS,PROFILES
from categories import CATEGORIES

ROOT=Path(__file__).resolve().parent
CALIBRATION=json.loads((ROOT/'calibration/thresholds.json').read_text(encoding='utf-8'))
MODEL=None
LOCK=Lock()
LABELS={'light':'Light / Лёгкий','medium':'Medium / Средний','high':'High / Высокий','corporate':'Corporate / Корпоративный'}

def settings(preset):
    values=[]
    for c in CATEGORIES:
        t=CALIBRATION['thresholds'][preset][c]
        values.extend([t is not None,round(t,6) if t is not None else .5])
    return [preset in ['high','corporate'],*values]

def analyze(text,language,mode,preset,lexical,*controls):
    global MODEL
    if not text or not text.strip():raise gr.Error('Enter text / Введите текст')
    if len(text)>50000:raise gr.Error('Maximum 50,000 characters / Максимум 50 000 символов')
    with LOCK:
        if MODEL is None:
            MODEL=Moderation03(os.getenv('MODEL_ID',MODEL_ID),device=os.getenv('DEVICE') or None,
                               backbone_path=os.getenv('BACKBONE_PATH') or None)
        result=MODEL.scores(text)
    thresholds=None;block_profanity=False
    if mode=='preset':
        thresholds=CALIBRATION['thresholds'][preset];block_profanity=preset in ['high','corporate']
    elif mode=='custom':
        thresholds={c:float(controls[2*i+1]) if controls[2*i] else None for i,c in enumerate(CATEGORIES)}
        block_profanity=bool(lexical)
    policy=None if thresholds is None else apply_policy(result['raw_scores'],thresholds,
        block_profanity=block_profanity,text=text,language=language)
    result.update(mode=mode,preset=preset if mode=='preset' else None,language=language,policy=policy,
        all_presets={p:apply_policy(result['raw_scores'],CALIBRATION['thresholds'][p],
            block_profanity=p in ['high','corporate'],text=text,language=language) for p in PROFILES})
    summary='**Raw scores / Сырые оценки**' if policy is None else (
        '**BLOCK / Блокировать**' if policy['block'] else '**ALLOW / Пропустить**')
    if policy and policy['reasons']:summary+='\n\nTriggered / Сработали: '+', '.join(policy['reasons'])
    if result['truncated']:summary+='\n\n⚠ Input truncated to 512 tokens / Текст обрезан до 512 токенов.'
    table=[[c,result['raw_scores'][c],None if thresholds is None else thresholds[c],
            '—' if policy is None else ('BLOCK' if policy['tags'][c] else '—')] for c in CATEGORIES]
    comparison=[[LABELS[p],result['all_presets'][p]['block'],', '.join(result['all_presets'][p]['reasons'])] for p in PROFILES]
    return summary,table,comparison,result

def build_app():
    with gr.Blocks(title='Moderation 03',analytics_enabled=False) as demo:
        gr.Markdown('# Moderation 03\n17 languages · 11 raw scores · 4 presets · custom thresholds')
        with gr.Row():
            with gr.Column():
                text=gr.Textbox(label='Text / Текст',lines=7,max_lines=14)
                language=gr.Dropdown(LANGS,value='ru',label='Language / Язык (for the profanity rule)')
                mode=gr.Radio([('Raw scores / Сырые оценки','raw'),('Preset / Пресет','preset'),
                               ('Custom thresholds / Свои пороги','custom')],value='preset',label='Mode / Режим')
                preset=gr.Dropdown([(LABELS[p],p) for p in PROFILES],value='medium',label='Preset / Пресет')
                button=gr.Button('Analyze / Проверить',variant='primary')
            with gr.Column():
                summary=gr.Markdown('Enter text and choose a mode / Введите текст и выберите режим.')
                table=gr.Dataframe(headers=['Category','Raw score','Threshold','Tag'],datatype=['str','number','number','str'],interactive=False)
        comparison=gr.Dataframe(headers=['Preset','Blocked','Reasons'],datatype=['str','bool','str'],interactive=False,label='All presets / Все пресеты')
        with gr.Accordion('Custom thresholds / Индивидуальные пороги',open=False):
            gr.Markdown('Editing these controls switches to custom mode. Disable a category with its checkbox. / Изменение включает режим своих порогов. Снимите галочку, чтобы отключить категорию.')
            lexical=gr.Checkbox(value=False,label='Block finite profanity lexicon / Блокировать мат по словарю')
            controls=[];defaults=settings('medium')[1:]
            for i,c in enumerate(CATEGORIES):
                with gr.Row():
                    enabled=gr.Checkbox(value=defaults[2*i],label=c)
                    threshold=gr.Slider(0,1,value=defaults[2*i+1],step=.001,label='Threshold / Порог')
                controls.extend([enabled,threshold])
            for control in [lexical,*controls]:control.input(lambda:'custom',outputs=mode,queue=False)
        preset.change(settings,inputs=preset,outputs=[lexical,*controls],queue=False)
        details=gr.JSON(label='Raw scores and decisions / Оценки и решения JSON')
        button.click(analyze,inputs=[text,language,mode,preset,lexical,*controls],outputs=[summary,table,comparison,details],
                     concurrency_limit=1,api_name='moderate')
        gr.Markdown('Scores are not calibrated probabilities. Presets are content thresholds, not a work-topic classifier. Corporate does not reliably determine whether a topic belongs in a particular workplace. / Оценки не являются калиброванными вероятностями. Пресеты не определяют, относится ли тема к работе.')
        gr.Markdown('[Model](https://huggingface.co/ifmain/Moderation-03) · [Evaluation report](https://huggingface.co/spaces/ifmain/Moderation-03-report) · [GitHub](https://github.com/ifmain/Moderation-03)\n\nEvaluation found false positives and missed fraud requests; the reported researcher vulnerability is not established as fixed. The app does not save submitted text to a dataset.')
    return demo.queue(max_size=8)

if __name__=='__main__':
    build_app().launch(server_name=os.getenv('GRADIO_SERVER_NAME','0.0.0.0' if os.getenv('SPACE_ID') else '127.0.0.1'),
                       server_port=int(os.getenv('PORT','7860')),share=False)
