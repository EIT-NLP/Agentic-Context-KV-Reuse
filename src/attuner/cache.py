"""Model-native forward and Hugging Face cache compatibility helpers."""
from __future__ import annotations

import contextlib
import fcntl
import json
from pathlib import Path

import inspect
import torch
from transformers.cache_utils import DynamicCache

CacheTensors = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def cache_to_tensors(cache) -> CacheTensors:
    if cache is None:
        return ()
    if isinstance(cache, (tuple, list)):
        return tuple((layer[0], layer[1]) for layer in cache)
    if callable(getattr(cache, "to_legacy_cache", None)):
        return tuple((layer[0], layer[1]) for layer in cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")


def cache_from_tensors(cache: CacheTensors, config):
    if not cache:
        return None
    # Construct new cache containers: model forwards may mutate their container.
    try:
        result = DynamicCache(config=config)
    except TypeError:
        result = DynamicCache()
    for index, (key, value) in enumerate(cache):
        result.update(key, value, index)
    return result


def cache_length(cache: CacheTensors) -> int:
    return int(cache[0][0].shape[-2]) if cache else 0


def concat_cache(left: CacheTensors, right: CacheTensors) -> CacheTensors:
    if not left:
        return right
    if not right:
        return left
    if len(left) != len(right):
        raise ValueError("Cache layer counts differ")
    return tuple((torch.cat((a[0], b[0]), dim=-2),
                  torch.cat((a[1], b[1]), dim=-2)) for a, b in zip(left, right))


def concat_caches(parts) -> CacheTensors:
    parts = [part for part in parts if part]
    if not parts:
        return ()
    if len(parts) == 1:
        return parts[0]
    if len({len(part) for part in parts}) != 1:
        raise ValueError('Cache layer counts differ')
    return tuple(tuple(torch.cat([part[layer][kind] for part in parts], dim=-2)
                       for kind in range(2)) for layer in range(len(parts[0])))


def move_cache(cache: CacheTensors, device, *, detach=False) -> CacheTensors:
    return tuple(tuple((tensor.detach() if detach else tensor).to(device)
                       for tensor in layer) for layer in cache)


def slice_cache(cache: CacheTensors, start: int, end=None) -> CacheTensors:
    return tuple((key[..., start:end, :], value[..., start:end, :])
                 for key, value in cache)


def forward_tokens(model, token_ids, *, device, cache: CacheTensors = (),
                   position_start: int | None = None, position_ids=None,
                   last_logits_only=False):
    values = list(token_ids)
    if not values:
        raise ValueError("Cannot forward empty token sequence")
    past = cache_length(cache)
    start = past if position_start is None else position_start
    inputs = torch.tensor([values], dtype=torch.long, device=device)
    positions = (torch.arange(start, start + len(values), device=device)[None, :]
                 if position_ids is None else position_ids.to(device))
    kwargs = dict(input_ids=inputs, position_ids=positions,
                  attention_mask=torch.ones((1, past + len(values)),
                                            dtype=torch.long, device=device),
                  past_key_values=cache_from_tensors(cache, model.config),
                  use_cache=True, return_dict=True)
    parameters = inspect.signature(model.forward).parameters
    if "cache_position" in parameters:
        kwargs["cache_position"] = torch.arange(past, past + len(values), device=device)
    if last_logits_only and "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    logits = output.logits[:, -1:, :] if last_logits_only else output.logits
    return logits, cache_to_tensors(output.past_key_values)


def teacher_forced_logits(model, prompt_logits, prompt_cache, response_ids,
                         *, device, prompt_length):
    if not response_ids:
        raise ValueError("Teacher response must contain at least one token")
    pieces = [prompt_logits[:, -1:, :]]
    if len(response_ids) > 1:
        continuation, _ = forward_tokens(model, response_ids[:-1], device=device,
                                         cache=prompt_cache, position_start=prompt_length)
        pieces.append(continuation)
    return torch.cat(pieces, dim=1)


@contextlib.contextmanager
def directory_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.provenance.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def bind_cache_directory(directory, identity):
    from .io import write_json
    directory = Path(directory)
    with directory_lock(directory):
        owner = directory / 'cache_identity.json'
        exists = owner.exists()
        if exists and json.loads(owner.read_text()) != identity:
            raise ValueError('Disk cache model/config differs; use a new cache directory')
        if not exists and any(directory.glob('*.pt')):
            raise ValueError('Legacy disk cache has no model reference; use a new cache directory')
        if not exists:
            write_json(owner, identity)
