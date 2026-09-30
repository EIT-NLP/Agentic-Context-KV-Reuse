# Baselines

These are Qwen3 ports sharing this repository's prompts and evaluation runtime.
They do not reproduce the upstream projects' optimized serving stacks.

| Method | Source | Release settings |
|---|---|---|
| APE | [Infini-AI-Lab/APE](https://github.com/Infini-AI-Lab/APE) | Temperature 0.9, scale 0.9; fused PyTorch attention; artifact-only cache, gaps computed online |
| EPIC | [DerekHJH/epic](https://github.com/DerekHJH/epic) | Recompute the first 4 tokens of each artifact |
| CacheBlend | [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) | Selection at zero-based layer 1, ratio 0.16, original-position causal mask |
| AgentKVShift | [Paper](https://arxiv.org/abs/2607.21604) | Paper reimplementation; no confirmed author repository; zero-based layer 1, ratio 0.1 |
| KVLink | [UCSB-NLP-Chang/KVLink](https://github.com/UCSB-NLP-Chang/KVLink) | Full-backbone/link-token training; response CE; learning rate 5e-6 |
| COMB | [shijuzhao/Comb](https://github.com/shijuzhao/Comb) | Frozen decoder, train encoder/projections/cross-attention; response CE; learning rate 5e-5 |

CacheBlend's upstream prototype also supports a bottom-right mask for sparse
queries; this release defaults to the original-position causal variant. APE's
`artifact_with_gaps` option reproduces the older cache layout; the public defaults
use `artifact_only` for both quality and timing. Keep these choices matched when
comparing methods. Do not combine results from different variants.

KVLink and COMB use the same 12,000 training questions and Full responses as
Attuner, with their own response-CE objectives, not Attuner's distribution KL.
Both use BF16, AdamW, weight decay 0.01, gradient clipping 1, one epoch and global
batch 16. These are our controlled comparison settings, not the complete training
schedules of the original papers. COMB accepts the CLI name `comb`; its internal
directory/identifier remains `encoder_pic`.

Train with `python -m baselines.train_mixed`; evaluate quality with
`python -m attuner.benchmark`; measure latency with
`python -m baselines.measure_efficiency`. Full commands are in the root README.
