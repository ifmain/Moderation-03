# Moderation 03

Frozen **Qwen3.5-2B-Base** text features → **254,564,363-parameter transformer head** → **11 raw category scores**.
Supports 17 languages: en, de, fr, es, it, sv, fi, pl, cs, lv, zh, ja, ko, ru, uk, be, kk. This is the selected epoch-5 checkpoint.

[Interactive demo](https://huggingface.co/spaces/ifmain/moderation-3) · [Full evaluation report](https://github.com/ifmain/Moderation-03/tree/main/evaluation) · [Model weights](https://huggingface.co/ifmain/Moderation-03) · [Source](https://github.com/ifmain/Moderation-03)

## Outputs and controls

The raw sigmoid scores are **not calibrated probabilities**. They remain unchanged when selecting a preset.
Categories: `harassment`, `harassment_threatening`, `hate`, `hate_threatening`, `self_harm`, `self_harm_instructions`, `self_harm_intent`, `sexual`, `sexual_minors`, `violence`, `violence_graphic`.

Raw scores are always displayed. Select a preset (`light`, `medium`, `high`, `corporate`) to populate editable per-category thresholds; every edit immediately affects filtering. There is no mode selector. A `null` threshold disables a category. Enabled categories trigger when `score >= threshold`; any trigger blocks the text. High and corporate presets also apply a finite multilingual profanity lexicon. Custom thresholds can change each category independently and optionally enable that lexicon. Changing presets or thresholds reuses session-local scores without running the model again. Editing the text clears the previous result. The Python API still supports raw-only output with `preset=None`.

Corporate is the strictest **content-threshold** preset. The project owner also intends to restrict non-work discussions in workplace deployments, but this release has **no workplace-topic relevance detector**. A benign statement about equal rights is not a hate ground-truth label merely because an organization chooses to restrict that topic. The published evaluation retains its original content-policy labels; it does not measure a separate work-topic policy.

## Run

Clone the GitHub repository, install `requirements.txt`, and run `python app.py`. Head weights and the pinned Qwen backbone download automatically. CUDA is used when available; CPU is supported but slower. The model loads before the application accepts requests. A plain ASGI server runs without hot reload or Node SSR. Local execution needs no hosting subscription. The hosted demo is available at https://huggingface.co/spaces/ifmain/moderation-3.

```python
from moderation03 import Moderation03

model = Moderation03("ifmain/Moderation-03")
raw = model.predict("Hello, thanks for your help.", language="en")
medium = model.predict("Hello, thanks for your help.", language="en", preset="medium")
custom = dict.fromkeys(model.config["categories"], None)
custom["harassment"] = 0.4
result = model.predict("Hello, thanks for your help.", language="en", thresholds=custom)
print(result["raw_scores"], result["policy"])
```

`MODEL_ID` may name a local model package or Hub repository. `BACKBONE_PATH` optionally points to a local Qwen snapshot for the app. `DEVICE=cpu` or `cuda` overrides automatic device selection. Local app defaults to `127.0.0.1:7860`; Spaces binds port 7860 publicly. The app does not write submitted text to a dataset. Hugging Face/Gradio hosting has its own service policies.

## Architecture and provenance

Backbone: `Qwen/Qwen3.5-2B-Base`, pinned revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
Head: input size 2048, width 1024, 20 transformer encoder blocks, 16 attention heads, feed-forward size 4096, masked mean pooling, 11 classifiers. Token limit: **512**, no chat template. Longer texts are truncated and the API returns `truncated=true`.
Published `model.safetensors` is the **FP32 head**, SHA256 `4218fb73825ff2fbbdd897a4346dd67141b15fc63d1811413929dddb88d408d3`; it is not the full Qwen model. CUDA uses the evaluated BF16 inference/autocast pipeline. CPU uses FP32 to avoid software-emulated BF16; small numerical differences are possible. Weights and frozen thresholds are unchanged. The rounded BF16 head export is intentionally not substituted for the evaluated weights.

Training: 100,000 examples, 70% clean / 30% character-level augmentation, 17 languages; 2,000 separate validation examples. Selected checkpoint: epoch 5, validation BCE 0.0967052458. Training data derives from [ifmain/text-moderation-02-multilingual](https://huggingface.co/datasets/ifmain/text-moderation-02-multilingual). Character substitutions and inserted separators do not establish robustness to semantic attacks. Source language assignment included uncertain groups.

## Calibration and evaluation

Internal threshold fitting: **828 rows / 638 source families**. Internal audit: **412 rows / 268 families**. All 11 categories have positive reference examples in all 17 languages in both partitions; translations are correlated, not independent examples. Most labels are inherited teacher estimates, not independently reviewed human ground truth. Thresholds minimize group-weighted balanced error on the fit partition, with nested preset constraints. This is threshold calibration, not probability calibration; no globally optimal thresholds are claimed.

Internal held-out content-policy results:

| Preset | TP | FP | TN | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Light | 109 | 35 | 244 | 24 | 75.7% | 82.0% |
| Medium | 189 | 51 | 146 | 26 | 78.8% | 87.9% |
| High | 245 | 24 | 121 | 22 | 91.1% | 91.8% |
| Corporate | 252 | 22 | 115 | 23 | 92.0% | 91.6% |

External evaluation: [mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation), revision `84e5cf3bcd6acb3dfc70b6760451645872218a3e`, **1,680 English examples**. Unknown labels are excluded, not treated as negatives. No thresholds were fitted on this dataset. Exact normalized overlaps with used groups, including all their translations: zero; arbitrary paraphrase/pretraining contamination is not excluded. 76 external texts exceed 512 tokens and were truncated.

External projections onto available benchmark categories (not full four-policy ground truth; profanity and three unsupported heads excluded):

| Preset | Evaluable rows | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|
| Light | 800 | 202 | 210 | 323 | 65 |
| Medium | 859 | 493 | 196 | 141 | 29 |
| High | 859 | 504 | 220 | 117 | 18 |
| Corporate | 859 | 505 | 222 | 115 | 17 |

## Known failures

An additional 265 synthetic checks cover adult sexual content, fraud assistance, disturbing imagery, hate-related requests and violence, plus benign controls, typography changes and 10 English/Russian paraphrases. They are not independently human-reviewed and do not reproduce the researchers' undisclosed benchmark. Corporate misses 29 of 36 fraud-request variants; under the evaluated content policy it also blocks 16 of 17 benign equal-rights messages. There is no dedicated illegal-activity head; broad disturbing content is not equivalent to graphic violence. Lower thresholds introduce false positives. The reported ModerationBERT-En transformed-prompt vulnerability is **not established as fixed** by this release.

The detailed report contains category/language metrics and the frozen thresholds. Public evaluation outputs contain row IDs and scores, **not source text**. Training data, source prompts, credentials, hidden-state caches and optimizer states are not distributed in this repository.

## License

Apache-2.0 for released code and head weights. The backbone is downloaded separately under its own Apache-2.0 license. External evaluation provenance and definitions: [OpenAI moderation-api-release](https://github.com/openai/moderation-api-release).
