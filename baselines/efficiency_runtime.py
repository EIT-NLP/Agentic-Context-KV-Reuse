"""PIC prefill without serial model calls between cached blocks.

Only the execution schedule changes: every online position reads its original
causal history, and artifact KV/positions are identical to the quality runtime.
Fixed prefixes may be cached only when the driver has established that their
token sequence is shared across the dataset. Gaps remain causally contextualized.
"""
import contextlib
import time

import torch
import torch.nn.functional as F

from attuner.attention_ops import project, repeat_kv, rotary
from attuner.cache import move_cache
from baselines.qwen3_runtime import Qwen3BaselineRuntime


class EfficiencyRuntime(Qwen3BaselineRuntime):
    fixed_prefix_tokens = ()

    def _prefix(self, plan, method):
        tokens = self.fixed_prefix_tokens
        if not tokens or not plan.segments:
            return (), 0
        if (tuple(plan.token_ids[:len(tokens)]) != tokens or
                len(tokens) > plan.segments[0].start):
            raise ValueError('Fixed prefix overlaps an artifact or differs from the prompt')
        key = ('fixed_prefix', method == 'pic_attuner', tokens)
        if key not in self._resident_cache:
            if getattr(self, 'require_warm_cache', False):
                raise RuntimeError('Fixed-prefix cache missed during TTFT')
            # For PIC+adapter, reuse the prefix computed WITH its online adapter.
            # This preserves the original online-prefix computation exactly.
            with torch.inference_mode():
                _, cache = self._forward(tokens, position_start=0,
                    mask=None if method == 'pic_attuner' else False)
            self.remember_cache(key, move_cache(cache, 'cpu', detach=True))
        return self._resident_cache[key], len(tokens)

    def prepare_request(self, plan, method):
        super().prepare_request(plan, method)
        if method in {'pic', 'position_oracle', 'pic_attuner'}:
            self._prefix(plan, method)

    def prefill(self, plan, method, training=False):
        if method not in {'pic', 'position_oracle', 'pic_attuner'} or not plan.segments:
            return super().prefill(plan, method, training)
        if training:
            raise ValueError('Efficiency runtime is inference-only')
        self._ape_layout = None
        controller = self._controller()
        if method == 'pic_attuner' and controller is None:
            raise ValueError('PIC+adapter requires a matching adapter')
        gating = (controller.disabled() if controller and method != 'pic_attuner'
                  else contextlib.nullcontext())
        with torch.inference_mode(), gating:
            pieces, built = self._warm(plan, method)
            prefix, prefix_length = self._prefix(plan, method)
            self._sync()
            start = time.perf_counter()
            cfg, base = self.model.config, self.model.model
            length = len(plan.token_ids)
            # Transfer exactly the original cached states; never re-rotate PIC K.
            pieces = [move_cache(piece, self.device) for piece in pieces]
            prefix = move_cache(prefix, self.device)
            self._sync()
            loaded = time.perf_counter() - start
            online = [True] * length
            online[:prefix_length] = [False] * prefix_length
            for segment in plan.segments:
                online[segment.start:segment.end] = [False] * segment.token_count
            positions = torch.tensor([i for i, yes in enumerate(online) if yes],
                                     device=self.device, dtype=torch.long)
            if positions.numel() == 0 or positions[-1].item() != length - 1:
                raise ValueError('The final prompt token must be online')
            hidden = base.embed_tokens(torch.tensor([plan.token_ids], device=self.device)[:, positions])
            cos, sin = base.rotary_emb(hidden, positions[None])
            mask = torch.arange(length, device=self.device)[None] <= positions[:, None]
            adapter_mask = torch.tensor([plan.adapter_mask], device=self.device)[:, positions]
            # Preserve the original GEMM shapes for BF16. Combining a one-token
            # gap with a long suffix changes cuBLAS rounding enough to amplify
            # through later layers, even though the dependency graph is equal.
            indices = positions.tolist()
            cuts = [0] + [i for i in range(1,len(indices)) if indices[i] != indices[i-1]+1] + [len(indices)]
            groups = list(zip(cuts,cuts[1:]))
            result = []
            with contextlib.nullcontext():
                for index, layer in enumerate(base.layers):
                    projected = []
                    for a,b in groups:
                        token_gating = (controller.masked_tokens(adapter_mask[:,a:b])
                                        if method == 'pic_attuner' else contextlib.nullcontext())
                        with token_gating:
                            projected.append(project(layer.self_attn,layer.input_layernorm(hidden[:,a:b]),cfg))
                    q,k,v = (torch.cat([part[j] for part in projected],dim=-2) for j in range(3))
                    q, k = rotary(q, k, cos, sin)
                    shape = (1, cfg.num_key_value_heads, length, layer.self_attn.head_dim)
                    keys, values = k.new_empty(shape), v.new_empty(shape)
                    if prefix_length:
                        keys[..., :prefix_length, :] = prefix[index][0]
                        values[..., :prefix_length, :] = prefix[index][1]
                    for segment, piece in zip(plan.segments, pieces):
                        keys[..., segment.start:segment.end, :] = piece[index][0]
                        values[..., segment.start:segment.end, :] = piece[index][1]
                    keys[..., positions, :] = k
                    values[..., positions, :] = v
                    outputs = []
                    for a,b in groups:
                        visible = indices[b-1]+1
                        causal = indices[a] == 0 and b-a == visible and b-a > 1
                        local_mask = None if b-a == 1 or causal else mask[None,None,a:b,:visible]
                        outputs.append(F.scaled_dot_product_attention(q[...,a:b,:],
                            repeat_kv(keys[...,:visible,:], cfg.num_attention_heads),
                            repeat_kv(values[...,:visible,:], cfg.num_attention_heads),
                            attn_mask=local_mask, dropout_p=0., is_causal=causal,
                            scale=float(layer.self_attn.scaling)))
                    out = torch.cat(outputs,dim=-2)
                    attention = out.transpose(1, 2).reshape(*hidden.shape[:-1], -1).contiguous()
                    updated = []
                    for a,b in groups:
                        part = hidden[:,a:b] + layer.self_attn.o_proj(attention[:,a:b])
                        updated.append(part + layer.mlp(layer.post_attention_layernorm(part)))
                    hidden = torch.cat(updated,dim=1)
                    result.append((keys, values))
            logits = self.model.lm_head(base.norm(hidden[:, -1:]))
            self._sync()
            return logits, tuple(result), dict(cache_build_seconds=built,
                kv_load_seconds=loaded, online_started=start,
                online_prefill_seconds=time.perf_counter()-start,
                reuse_tokens=sum(s.token_count for s in plan.segments)+prefix_length,
                recompute_tokens=0, online_tokens=len(positions),
                fixed_prefix_tokens=prefix_length, prefill_schedule='layerwise_online',
                cache_semantics='original_artifact_kv_and_causal_online_states')
