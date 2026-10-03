"""Portable inference for the exact FP32 epoch-5 head used in the published evaluation."""
import gc
import json
import os
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from moderation_model import ModerationHead
from categories import CATEGORIES

MODEL_ID='ifmain/Moderation-03'
PROFILES=['light','medium','high','corporate']

def apply_policy(scores, thresholds):
    if set(scores)!=set(CATEGORIES) or set(thresholds)!=set(CATEGORIES):
        raise ValueError('Exactly 11 named categories are required')
    for value in thresholds.values():
        if value is not None and not 0<=value<=1:raise ValueError('Thresholds must be in [0,1] or null (disabled)')
    tags={c:thresholds[c] is not None and scores[c]>=thresholds[c] for c in CATEGORIES}
    reasons=[c for c,flagged in tags.items() if flagged]
    return {'block':bool(reasons),'tags':tags,'reasons':reasons,'thresholds':thresholds}

class Moderation03:
    def __init__(self, model_id=MODEL_ID, device=None, backbone_path=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # CPU BF16 is often emulated on basic hosted machines; use native FP32 there.
        self.dtype=torch.bfloat16 if self.device.startswith('cuda') else torch.float32
        torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS','4')))
        torch.backends.mha.set_fastpath_enabled(False)
        path=Path(model_id)
        self.root=path if path.is_dir() else Path(snapshot_download(model_id,allow_patterns=[
            'config.json','model.safetensors','calibration/thresholds.json']))
        self.config=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        self.calibration=json.loads((self.root/'calibration/thresholds.json').read_text(encoding='utf-8'))
        backbone=backbone_path or self.config['backbone']
        local=Path(backbone).is_dir()
        kwargs={'local_files_only':True} if local else {'revision':self.config['backbone_revision']}
        self.tokenizer=AutoTokenizer.from_pretrained(backbone,**kwargs)
        self.tokenizer.padding_side='right'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        full=Qwen3_5ForConditionalGeneration.from_pretrained(backbone,dtype=self.dtype,
                attn_implementation='sdpa',**kwargs)
        self.backbone=full.model.language_model
        del full;gc.collect()
        self.backbone.requires_grad_(False).eval().to(self.device)
        self.head=ModerationHead(self.config)
        self.head.load_state_dict(load_file(str(self.root/'model.safetensors'),device='cpu'))
        self.head.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def scores(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError('Enter non-empty text')
        if len(text)>50000:raise ValueError('Maximum input length is 50,000 characters')
        ids=self.tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
        total=len(ids);ids=ids[:self.config['max_length']] or [self.tokenizer.eos_token_id]
        encoded={'input_ids':torch.tensor([ids],device=self.device),
                 'attention_mask':torch.ones((1,len(ids)),dtype=torch.long,device=self.device)}
        hidden=self.backbone(**encoded,use_cache=False,return_dict=True).last_hidden_state
        with torch.autocast(self.device.split(':')[0],dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
            logits=self.head(hidden.float(),encoded['attention_mask'])
        if not bool(torch.isfinite(logits).all()):raise ValueError('Non-finite scores')
        values=logits.float().sigmoid()[0].cpu().tolist()
        return {'raw_scores':dict(zip(CATEGORIES,values)), 'input_tokens':total,
                'processed_tokens':len(ids),'truncated':total>self.config['max_length'],
                'checkpoint_epoch':5,'score_type':'uncalibrated sigmoid score',
                'device':self.device,'compute_dtype':str(self.dtype)}

    def predict(self,text,preset=None,thresholds=None):
        if preset is not None and preset not in PROFILES:raise ValueError('Unknown preset')
        if preset is not None and thresholds is not None:raise ValueError('Choose preset or custom thresholds')
        result=self.scores(text)
        if preset is not None:
            thresholds=self.calibration['thresholds'][preset]
        result['preset']=preset
        result['policy']=None if thresholds is None else apply_policy(result['raw_scores'],thresholds)
        return result
