"""APE with artifact-only caches and causal online separator tokens.

Artifacts retain APE's shared-prefix encoding. Every gap is an online query,
reading only preceding artifacts and online tokens. Execute online spans by
layer, then store prefix/artifacts/online KV contiguously for ordinary APE decode.
RoPE for each online span starts after the largest position visible so far.
"""
import time
import torch
from attuner.attention_ops import project, repeat_kv, rotary
from attuner.cache import move_cache
from attuner.runtime import option
from baselines.ape.qwen3 import attention


def online_spans(plan):
    prefix = plan.segments[0].start
    longest, next_position, count = 0, prefix, 0
    spans = []
    for i, segment in enumerate(plan.segments):
        longest = max(longest, segment.token_count)
        end = plan.segments[i + 1].start if i + 1 < len(plan.segments) else len(plan.token_ids)
        if end > segment.end:
            start_position = max(next_position, prefix + longest)
            width = end - segment.end
            spans.append((segment.end, end, i + 1, count, count + width, start_position))
            count += width
            next_position = start_position + width
    return spans, next_position


def prefill(runtime, plan):
    pieces, built = runtime._ape_caches(plan)
    runtime._sync()
    started = time.perf_counter()
    pieces = [move_cache(piece, runtime.device) for piece in pieces]
    runtime._sync()
    loaded = time.perf_counter() - started
    spans, next_position = online_spans(plan)
    if not spans or spans[-1][1] != len(plan.token_ids):
        raise ValueError('APE requires an online suffix ending at the final prompt token')
    positions = [i for start, end, *_ in spans for i in range(start, end)]
    rope_positions = [i for start, end, _, _, _, pos in spans for i in range(pos, pos + end - start)]
    base, cfg = runtime.model.model, runtime.model.config
    tokens = torch.tensor([[plan.token_ids[i] for i in positions]], device=runtime.device)
    hidden = base.embed_tokens(tokens)
    pos = torch.tensor([rope_positions], device=runtime.device)
    cos, sin = base.rotary_emb(hidden, pos)
    prefix = plan.segments[0].start
    temperature = option(runtime.config, 'ape_temperature', .9)
    scale = option(runtime.config, 'ape_scale', .9)
    kernel = option(runtime.config, 'ape_kernel', 'flash')
    result = []
    for layer_index, layer in enumerate(base.layers):
        projected = [project(layer.self_attn, layer.input_layernorm(hidden[:, a:b]), cfg)
                     for _, _, _, a, b, _ in spans]
        q, k, v = (torch.cat([part[j] for part in projected], dim=-2) for j in range(3))
        q, k = rotary(q, k, cos, sin)
        updated = []
        for _, _, artifact_count, a, b, _ in spans:
            # No future artifact or online KV is supplied to this span.
            past = [piece[layer_index] for piece in pieces[:artifact_count + 1] if piece]
            keys = torch.cat([pair[0] for pair in past] + [k[..., :b, :]], dim=-2)
            values = torch.cat([pair[1] for pair in past] + [v[..., :b, :]], dim=-2)
            context = sum(s.token_count for s in plan.segments[:artifact_count])
            out = attention(q[..., a:b, :], repeat_kv(keys, cfg.num_attention_heads),
                            repeat_kv(values, cfg.num_attention_heads), prefix, context,
                            temperature, scale, kernel)
            out = out.transpose(1, 2).reshape(hidden.shape[0], b - a, -1).contiguous()
            part = hidden[:, a:b] + layer.self_attn.o_proj(out)
            updated.append(part + layer.mlp(layer.post_attention_layernorm(part)))
        hidden = torch.cat(updated, dim=1)
        cached = [piece[layer_index] for piece in pieces if piece]
        result.append((torch.cat([pair[0] for pair in cached] + [k], dim=-2),
                       torch.cat([pair[1] for pair in cached] + [v], dim=-2)))
    runtime._ape_layout = (prefix, sum(s.token_count for s in plan.segments))
    logits = runtime.model.lm_head(base.norm(hidden[:, -1:]))
    runtime._sync()
    return logits, tuple(result), dict(cache_build_seconds=built, kv_load_seconds=loaded,
        online_started=started, online_prefill_seconds=time.perf_counter()-started,
        logical_next_position=next_position, reuse_tokens=prefix+sum(s.token_count for s in plan.segments),
        recompute_tokens=0, online_tokens=len(positions), baseline_source='qwen3_port',
        baseline_kernel=kernel, ape_cache_layout='artifact_only', prefill_schedule='layerwise_online')
