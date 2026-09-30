"""Merge query LoRA for online inference while retaining the base artifact encoder.

Only query projections need a second copy. Cache construction temporarily uses
the original Linear modules; online prefill/decode use ordinary merged Linears.
No adapter matrix products or token gating are executed by the merged modules.
"""
import contextlib

import torch
from torch import nn

from attuner.lora import LoRALinear
from baselines.efficiency_runtime import EfficiencyRuntime


class MergedQueries:
    def __init__(self, model, controller):
        if controller is None or controller.target != 'q_proj':
            raise ValueError('Merged PIC timing requires a query-only adapter')
        self.projections = []
        self.target = controller.target
        self.model_identifier = getattr(controller, 'model_identifier', None)
        with torch.no_grad():
            for key, wrapper in controller.modules.items():
                layer, name = key.split('.')
                if name != 'q_proj' or not isinstance(wrapper, LoRALinear):
                    raise ValueError('Expected an unmerged query adapter')
                parent = model.model.layers[int(layer)].self_attn
                base = wrapper.base
                merged = nn.Linear(base.in_features, base.out_features,
                                   bias=base.bias is not None,
                                   device=base.weight.device, dtype=base.weight.dtype)
                # Accumulate the low-rank weight update in FP32, then round once
                # to the inference dtype. BF16 need not be bitwise identical to
                # the unmerged sequence of three matrix products and an add.
                weight = base.weight.float() + wrapper.scale * (
                    wrapper.lora_b.float() @ wrapper.lora_a.float())
                if not torch.isfinite(weight).all():
                    raise ValueError(f'Non-finite merged query weight: {key}')
                merged.weight.copy_(weight.to(base.weight.dtype))
                if base.bias is not None:
                    merged.bias.copy_(base.bias)
                merged.requires_grad_(False).eval()
                self.projections.append((parent, base, merged))
                parent.q_proj = merged

    @contextlib.contextmanager
    def disabled(self):
        previous = [parent.q_proj for parent, _, _ in self.projections]
        try:
            for parent, base, _ in self.projections:
                parent.q_proj = base
            yield
        finally:
            for (parent, _, _), original in zip(self.projections, previous):
                parent.q_proj = original

    def masked_tokens(self, mask):
        # This runtime validates that every ONLINE position is adapted before
        # preparing the request. Artifact rows are never passed to this path.
        return contextlib.nullcontext()


def merge_runtime_queries(runtime):
    controller = MergedQueries(runtime.model, runtime._controller())
    runtime.controller = controller
    runtime.model._kv_order_lora_controller = controller
    # In particular, do not reuse a fixed prefix produced before the merge.
    runtime.clear_cache()
    return controller


class MergedAttunerRuntime(EfficiencyRuntime):
    def __init__(self, config):
        super().__init__(config)
        merge_runtime_queries(self)

    def prepare_request(self, plan, method):
        if method != 'pic_attuner' or not plan.segments:
            raise ValueError('Merged runtime supports PIC+adapter with artifact KV only')
        online = list(plan.adapter_mask)
        for segment in plan.segments:
            online[segment.start:segment.end] = [True] * segment.token_count
        if not all(online):
            raise ValueError('Merged query weights require adaptation at all online tokens')
        return super().prepare_request(plan, method)

    def prefill(self, plan, method, training=False):
        if method != 'pic_attuner' or training or not plan.segments:
            raise ValueError('Merged runtime supports PIC+adapter inference only')
        return super().prefill(plan, method, training)

    def _forward(self, tokens, *, cache=(), position_start=None, mask=None, training=False):
        if training or (mask is not None and mask is not False and not all(mask)):
            raise ValueError('Mixed base/adapted tokens cannot use one merged projection')
        return super()._forward(tokens, cache=cache, position_start=position_start,
                                mask=mask, training=training)
