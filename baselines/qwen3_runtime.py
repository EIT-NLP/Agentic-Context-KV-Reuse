"""Isolated Qwen3 ports. This module is not registered in the paper runtime.

Shared loading, tokenization and cache primitives come from attuner; method
algorithms live under baselines. The original baseline implementations are unused.
"""
import contextlib
import time
import torch
import torch.nn.functional as F
from attuner.runtime import Runtime, option
from attuner.attention_ops import cache_rope, project, repeat_kv, rotary
from attuner.cache import move_cache, concat_caches, slice_cache
from baselines.ape.qwen3 import patched_attention
from baselines.epic.qwen3 import selected_positions
from baselines.cacheblend.qwen3 import select_queries, attention_mask
from baselines.agentkvshift import qwen3 as shift


class Qwen3BaselineRuntime(Runtime):
    def __init__(self, config):
        super().__init__(config)
        if self.model.config.model_type != 'qwen3':
            raise ValueError('This port is for dense Qwen3 only')
        self._ape_layout = None

    def _forward(self, *args, **kwargs):
        context = contextlib.nullcontext()
        if self._ape_layout is not None:
            context = patched_attention(self.model, self._ape_layout,
                temperature=option(self.config, 'ape_temperature', .9),
                scale=option(self.config, 'ape_scale', .9),
                kernel=option(self.config, 'ape_kernel', 'flash'))
        with context:
            return super()._forward(*args, **kwargs)

    def _ape_caches(self, plan):
        prefix = plan.segments[0].start
        prefix_tokens = plan.token_ids[:prefix]
        pieces, prefix_gpu, built = [], (), 0.
        starts = [s.start for s in plan.segments]
        bounds = [(0, prefix)] + ([(s.start, s.end) for s in plan.segments]
            if option(self.config, 'ape_cache_layout', 'artifact_with_gaps') == 'artifact_only'
            else list(zip(starts, starts[1:] + [plan.segments[-1].end])))
        for index, (start, end) in enumerate(bounds):
            if start == end:
                pieces.append(())
                continue
            tokens = plan.token_ids[start:end]
            key = ('ape_qwen3', prefix_tokens, tokens, index == 0)
            if key in self._resident_cache:
                piece = self._resident_cache[key]
            else:
                if getattr(self, 'require_warm_cache', False):
                    raise RuntimeError('APE CPU RAM cache miss in measured request')
                self._sync(); begin = time.perf_counter()
                if index and not prefix_gpu:
                    prefix_gpu = move_cache(pieces[0], self.device)
                _, cache = self._forward(tokens, cache=() if index == 0 else prefix_gpu,
                    position_start=0 if index == 0 else prefix, mask=False)
                piece = move_cache(cache if index == 0 else slice_cache(cache, prefix), 'cpu', detach=True)
                self._sync(); built += time.perf_counter() - begin
                self.remember_cache(key, piece)
            pieces.append(piece)
        return pieces, built

    def prepare_request(self, plan, method):
        self._ape_layout = None
        if method == 'ape' and plan.segments:
            self._ape_caches(plan)
        elif method in {'epic', 'cacheblend', 'agentkvshift'}:
            self._warm(plan, 'raw_keys')
        else:
            super().prepare_request(plan, method)

    def prefill(self, plan, method, training=False):
        self._ape_layout = None
        if method not in {'ape', 'epic', 'cacheblend', 'agentkvshift'} or not plan.segments:
            return super().prefill(plan, 'full' if not plan.segments else method, training)
        if training:
            raise ValueError('Training-free baseline called with training=True')
        controller = self._controller()
        with torch.inference_mode(), controller.disabled() if controller else contextlib.nullcontext():
            return self._ape_prefill(plan) if method == 'ape' else self._sparse_prefill(plan, method)

    def _sparse_prefill(self, plan, method):
        pieces, built = self._warm(plan, 'raw_keys')
        self._sync(); started = time.perf_counter() - self._cache_read_seconds
        # All online slots will be refreshed before their layer's attention.
        # They need no preliminary PIC prefill and cannot supply future evidence.
        cfg = self.model.config
        shape = (1, cfg.num_key_value_heads, len(plan.token_ids), self.model.model.layers[0].self_attn.head_dim)
        cache = [[torch.zeros(shape, device=self.device, dtype=next(self.model.parameters()).dtype)
                  for _ in range(2)] for _ in self.model.model.layers]
        for segment, piece in zip(plan.segments, pieces):
            positioned = cache_rope(self.model, move_cache(piece, self.device), segment.start)
            for index, (k, v) in enumerate(positioned):
                cache[index][0][..., segment.start:segment.end, :] = k
                cache[index][1][..., segment.start:segment.end, :] = v
        self._sync(); loaded = time.perf_counter() - started
        logits, result, count = self._repair(plan, cache, method)
        self._sync()
        return logits, result, dict(cache_build_seconds=built, kv_load_seconds=loaded,
            online_started=started, online_prefill_seconds=time.perf_counter()-started,
            reuse_tokens=sum(s.token_count for s in plan.segments), recompute_tokens=count,
            baseline_source='paper_reimplementation' if method=='agentkvshift' else 'qwen3_port',
            baseline_kernel='sdpa',
            agentkvshift_check_layer=option(self.config,'agentkvshift_check_layer',None) if method=='agentkvshift' else None,
            cacheblend_mask=option(self.config, 'cacheblend_mask', 'upstream_bottom_right') if method=='cacheblend' else None)

    def _ape_prefill(self, plan):
        if option(self.config, 'ape_cache_layout', 'artifact_with_gaps') == 'artifact_only':
            from baselines.ape.online import prefill
            return prefill(self, plan)
        pieces, built = self._ape_caches(plan)
        self._sync(); started = time.perf_counter()
        cache = concat_caches([move_cache(piece, self.device) for piece in pieces])
        self._sync(); loaded = time.perf_counter() - started
        prefix, end = plan.segments[0].start, plan.segments[-1].end
        starts = [s.start for s in plan.segments]
        longest = max(b-a for a,b in zip(starts, starts[1:]+[end]))
        self._ape_layout = (prefix, end-prefix)
        logits, cache = self._forward(plan.token_ids[end:], cache=cache,
            position_start=prefix+longest, mask=False)
        self._sync()
        return logits, cache, dict(cache_build_seconds=built, kv_load_seconds=loaded,
            online_started=started, online_prefill_seconds=time.perf_counter()-started,
            logical_next_position=prefix+longest+len(plan.token_ids)-end,
            reuse_tokens=end, recompute_tokens=0, baseline_source='qwen3_port',
            baseline_kernel=option(self.config,'ape_kernel','flash'))

    def _repair(self, plan, cache, method):
        base, cfg = self.model.model, self.model.config
        positions = torch.tensor(selected_positions(plan, option(self.config,'epic_tokens',4))
            if method=='epic' else list(range(len(plan.token_ids))), device=self.device, dtype=torch.long)
        hidden = base.embed_tokens(torch.tensor([plan.token_ids],device=self.device))[:,positions]
        if method=='agentkvshift' and 'agentkvshift_check_layer' not in self.config:
            raise ValueError('AgentKVShift requires an explicit zero-based checkpoint layer; see its README')
        check = int(option(self.config,'agentkvshift_check_layer' if method=='agentkvshift' else 'cacheblend_check_layer',1))
        probes = None
        if not 0 <= check < len(base.layers):
            raise ValueError('Invalid sparse-repair check layer')
        for i, layer in enumerate(base.layers):
            q,k,v = project(layer.self_attn,layer.input_layernorm(hidden),cfg)
            cos,sin = base.rotary_emb(hidden,positions[None])
            q,k = rotary(q,k,cos,sin)
            if method=='agentkvshift':
                if i<check:
                    cache[i]=[k,v]
                else:
                    if i==check:
                        probes=shift.select(plan,k,v,*cache[i],option(self.config,'agentkvshift_ratio',.1))
                    cache[i]=[shift.correct(cache[i][0],k,positions,probes,'key'),
                              shift.correct(cache[i][1],v,positions,probes,'value')]
                    if i==check:
                        chosen=shift.selected_positions(plan,probes,self.device)
                        positions=positions[chosen];hidden=hidden[:,chosen];q=q[...,chosen,:]
            elif method=='cacheblend' and i<=check:
                if i==check:
                    chosen=select_queries(v,cache[i][1],plan.segments[-1].end,
                                          option(self.config,'cacheblend_ratio',.16))
                    positions=positions[chosen]; hidden=hidden[:,chosen]; q=q[...,chosen,:]
                cache[i]=[k,v]
            else:
                cache[i][0][...,positions,:]=k; cache[i][1][...,positions,:]=v
            mode=option(self.config,'cacheblend_mask','upstream_bottom_right') if method=='cacheblend' and i>=check else 'original_positions'
            mask=attention_mask(positions,len(plan.token_ids),mode)
            k,v=(repeat_kv(x,cfg.num_attention_heads) for x in cache[i])
            full_causal = len(positions) == len(plan.token_ids)
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=None if full_causal else mask[None,None],
                dropout_p=0.,is_causal=full_causal,scale=float(layer.self_attn.scaling))
            hidden=hidden+layer.self_attn.o_proj(out.transpose(1,2).reshape(*hidden.shape[:-1],-1).contiguous())
            hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
        if positions[-1].item()!=len(plan.token_ids)-1:
            raise ValueError('The final online token must be retained')
        return self.model.lm_head(base.norm(hidden[:,-1:])),tuple((k,v) for k,v in cache),len(positions)
