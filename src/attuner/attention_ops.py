"""Shared dense decoder operations for the documented training-free ports."""
from __future__ import annotations

import contextlib
import torch


def decoder(model):
    config = model.config
    if config.model_type not in {"qwen3", "qwen2", "llama"}:
        raise NotImplementedError(f"Training-free attention unsupported: {config.model_type}")
    base = model.model
    if len(base.layers) != config.num_hidden_layers:
        raise ValueError("Model layer count disagrees with config")
    if getattr(config, "use_sliding_window", False):
        raise NotImplementedError("Sliding-window attention is not supported by this port")
    layer_types = getattr(config, "layer_types", None)
    if layer_types and any(kind != "full_attention" for kind in layer_types):
        raise NotImplementedError("Training-free ports require full attention at every layer")
    for layer in base.layers:
        attention = layer.self_attn
        required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "scaling")
        if any(not hasattr(attention, name) for name in required):
            raise NotImplementedError("Decoder does not expose standard separate Q/K/V projections")
        if getattr(attention, "sliding_window", None):
            raise NotImplementedError("Sliding-window attention is not supported")
    return base


def rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@contextlib.contextmanager
def capture_raw_keys(model):
    """Capture normalized pre-RoPE K without inverse-rotating rounded caches."""
    keys, handles = {}, []
    try:
        for index, layer in enumerate(decoder(model).layers):
            attention = layer.self_attn
            def hook(module, inputs, output, index=index, attention=attention):
                key = output.view(*output.shape[:-1], model.config.num_key_value_heads, attention.head_dim)
                if hasattr(attention, "k_norm"):
                    key = attention.k_norm(key)
                keys[index] = key.transpose(1,2).detach()
            handles.append(attention.k_proj.register_forward_hook(hook))
        yield keys
    finally:
        for handle in handles:
            handle.remove()


def cache_rope(model, cache, start):
    """Store pre-RoPE keys offline; apply target coordinates online.

    Default Qwen3 RoPE only: position-dependent scaling needs a separately
    validated conversion and must never silently use this path.
    """
    cfg = model.config
    params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if params.get("rope_type", params.get("type", "default")) != "default":
        raise NotImplementedError("Raw-key cache conversion requires default RoPE")
    result = []
    for key, value in cache:
        positions = torch.arange(start, start + key.shape[-2], device=key.device)[None]
        cos, sin = model.model.rotary_emb(key, positions)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        result.append((key * cos + rotate_half(key) * sin, value))
    return tuple(result)


def rotary(query, key, cos, sin):
    if cos.shape[-1] != query.shape[-1]:
        raise NotImplementedError("Partial or nonstandard rotary dimensions are unsupported")
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def project(attention, hidden, config):
    batch, length, _ = hidden.shape
    dim = attention.head_dim
    query = attention.q_proj(hidden).view(batch, length, config.num_attention_heads, dim)
    key = attention.k_proj(hidden).view(batch, length, config.num_key_value_heads, dim)
    value = attention.v_proj(hidden).view(batch, length, config.num_key_value_heads, dim)
    if hasattr(attention, "q_norm"):
        query = attention.q_norm(query)
    if hasattr(attention, "k_norm"):
        key = attention.k_norm(key)
    return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)


def repeat_kv(tensor, query_heads):
    if query_heads % tensor.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    return tensor.repeat_interleave(query_heads // tensor.shape[1], dim=1)
