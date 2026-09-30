"""CacheBlend selection and explicit upstream-vs-causal mask semantics."""
import torch


def select_queries(fresh_v, cached_v, suffix_start, ratio=.16):
    if not 0 <= ratio <= 1:
        raise ValueError('Recompute ratio must be in [0, 1]')
    count = int(suffix_start * ratio)
    diff = ((fresh_v[..., :suffix_start, :] - cached_v[..., :suffix_start, :]) ** 2).sum((0, 1, 3))
    chosen = torch.topk(diff, count).indices.sort().values
    suffix = torch.arange(suffix_start, fresh_v.shape[-2], device=fresh_v.device)
    return torch.cat([chosen, suffix])


def attention_mask(positions, key_count, mode):
    if mode == 'upstream_bottom_right':
        bounds = torch.arange(len(positions), device=positions.device) + key_count - len(positions)
    elif mode == 'original_positions':
        bounds = positions
    else:
        raise ValueError('Specify upstream_bottom_right or original_positions explicitly')
    return torch.arange(key_count, device=positions.device)[None] <= bounds[:, None]
