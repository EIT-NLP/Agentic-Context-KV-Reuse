"""APE's two-group attention for Qwen3; upstream formula, model-specific port.

CUDA uses PyTorch's fused FlashAttention primitive (output + LSE), not the
upstream flash-attn Python package. Eager is only a numerical reference.
"""
import contextlib
import types
import torch
from attuner.attention_ops import project, repeat_kv, rotary


def _group(q, k, v, scale, causal, kernel):
    if kernel == 'flash':
        if not q.is_cuda:
            raise ValueError('Flash APE requires CUDA; use eager explicitly for reference tests')
        out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), 0.0, causal, False, scale=scale)
        return out, lse
    if kernel != 'eager':
        raise ValueError(kernel)
    scores = q.float() @ k.float().transpose(-1, -2) * scale
    if causal:
        mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, -torch.inf)
    return (scores.softmax(-1) @ v.float()).to(v.dtype), scores.logsumexp(-1)


def attention(q, k, v, prefix, context, temperature=.9, scale=.9, kernel='flash'):
    end = prefix + context
    if not 0 <= prefix < end <= k.shape[-2] - q.shape[-2]:
        raise ValueError('APE expects unpadded online queries after all cached contexts')
    if temperature <= 0 or scale <= 0:
        raise ValueError('APE temperature and scale must be positive')
    softmax_scale = q.shape[-1] ** -.5
    co, cl = _group(q, k[..., prefix:end, :], v[..., prefix:end, :],
                     softmax_scale / temperature, False, kernel)
    other_k = torch.cat([k[..., :prefix, :], k[..., end:, :]], -2)
    other_v = torch.cat([v[..., :prefix, :], v[..., end:, :]], -2)
    # PyTorch's non-square causal SDPA is top-left aligned; align online queries
    # explicitly to preserve upstream FA2 bottom-right visibility.
    past = other_k.shape[-2] - q.shape[-2]
    if q.shape[-2] == 1:
        oo, ol = _group(q, other_k, other_v, softmax_scale, False, kernel)
    else:
        # Pad only the query axis to emulate FA2's bottom-right causal mask.
        # This also avoids rounding two separate partial outputs before merging.
        padded = torch.cat([q.new_zeros(*q.shape[:-2], past, q.shape[-1]), q], -2)
        oo, ol = _group(padded, other_k, other_v, softmax_scale, True, kernel)
        oo, ol = oo[..., -q.shape[-2]:, :], ol[..., -q.shape[-2]:]
    # Keep the tiny group-normalization tensors in FP32. Upstream casts these
    # to BF16; that rounding can amplify small kernel differences across layers.
    # The temperature/scaling formula is unchanged; this precision adaptation
    # is recorded explicitly in the port's README.
    logits = torch.stack([cl.float() * (scale * temperature), ol.float()], -1)
    weights = logits.float().softmax(-1).to(q.dtype)
    return weights[..., :1] * co + weights[..., 1:] * oo


@contextlib.contextmanager
def patched_attention(model, layout, *, temperature=.9, scale=.9, kernel='flash'):
    saved = []
    def forward(module, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        if hidden_states.shape[0] != 1:
            raise ValueError('This APE port supports unpadded batch-one inference')
        q, k, v = project(module, hidden_states, model.config)
        q, k = rotary(q, k, *position_embeddings)
        cache = past_key_values if past_key_values is not None else kwargs.get('past_key_value')
        if cache is not None:
            k, v = cache.update(k, v, module.layer_idx)
        k, v = (repeat_kv(x, model.config.num_attention_heads) for x in (k, v))
        out = attention(q, k, v, layout[0], layout[1], temperature, scale, kernel)
        out = out.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return module.o_proj(out.contiguous()), None
    try:
        for layer in model.model.layers:
            module = layer.self_attn
            saved.append((module, module.forward))
            module.forward = types.MethodType(forward, module)
        yield
    finally:
        for module, original in saved:
            module.forward = original
