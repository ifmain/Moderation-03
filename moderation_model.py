"""Trainable transformer head over the full sequence of frozen Qwen hidden states."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'models/moderation_head/config.json'


def load_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class ModerationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config['input_size'])
        self.projection = nn.Linear(config['input_size'], config['hidden_size'])
        self.positions = nn.Embedding(config['max_length'], config['hidden_size'])
        # Construct independently: every layer receives its own random initialization.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'], nhead=config['num_heads'],
                dim_feedforward=config['intermediate_size'], dropout=config['dropout'],
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(config['num_layers'])
        ])
        self.final_norm = nn.LayerNorm(config['hidden_size'])
        self.classifier = nn.Linear(config['hidden_size'], len(config['categories']))

    def forward(self, hidden_states, attention_mask):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config['input_size']:
            raise ValueError('Expected [batch, sequence, Qwen_hidden_size]')
        if hidden_states.shape[:2] != attention_mask.shape:
            raise ValueError('Mask shape mismatch')
        if hidden_states.shape[1] > self.config['max_length']:
            raise ValueError('Sequence exceeds configured maximum')
        mask = attention_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Every example requires at least one non-padding token')
        x = self.projection(self.input_norm(hidden_states))
        positions = (mask.long().cumsum(dim=1) - 1).clamp_min(0)
        x = x + self.positions(positions)
        padding = ~mask
        for layer in self.layers:
            if self.training and self.config.get('gradient_checkpointing', False):
                x = checkpoint(layer, x, src_key_padding_mask=padding, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=padding)
        x = self.final_norm(x)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.classifier(pooled)

    @torch.inference_mode()
    def predict_scores(self, hidden_states, attention_mask):
        if self.training:
            raise RuntimeError('Call eval() before inference')
        probabilities = self(hidden_states, attention_mask).float().sigmoid()
        return [dict(zip(self.config['categories'], row)) for row in probabilities.cpu().tolist()]


def apply_profile(scores, thresholds):
    """Future calibrated thresholds must be provided explicitly for all categories."""
    if thresholds is None or set(scores) != set(thresholds):
        raise ValueError('This profile has not been calibrated for all 11 categories')
    if any(not 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError('Thresholds must lie in [0, 1]')
    return {name: score >= thresholds[name] for name, score in scores.items()}
