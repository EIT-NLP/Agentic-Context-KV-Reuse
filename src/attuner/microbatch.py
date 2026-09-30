"""Padded native model forwards for unequal-length Full/PIC training requests.

Past KV is left-padded and new tokens are right-padded. RoPE uses each
request's real positions; cache padding is masked and removed after each call.
No requests share attention or artifact states.
"""
from __future__ import annotations

import contextlib
import inspect

import torch
import torch.nn.functional as F

from .cache import (cache_from_tensors, cache_length, cache_to_tensors,
                    concat_cache, move_cache)


def forward_batch(runtime, token_rows, *, caches=None, starts=None,
                  adapter_masks=None, training=False, last_only=False):
    """One native forward for all rows, returning unpadded logits and KV."""
    if not token_rows or any(not row for row in token_rows):
        raise ValueError("Micro-batch rows must be nonempty")
    batch = len(token_rows)
    caches = [()] * batch if caches is None else caches
    if len(caches) != batch:
        raise ValueError("Cache batch size differs")
    lengths = [len(row) for row in token_rows]
    pasts = [cache_length(c) for c in caches]
    starts = pasts if starts is None else starts
    if len(starts) != batch:
        raise ValueError("Position batch size differs")
    width, past_width = max(lengths), max(pasts)
    device = runtime.device
    pad = getattr(runtime.model.config, "pad_token_id", None)
    inputs = torch.full((batch, width), 0 if pad is None else pad, dtype=torch.long, device=device)
    positions = torch.zeros_like(inputs)
    attention = torch.zeros((batch, past_width + width), dtype=torch.long, device=device)
    token_mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
    if adapter_masks is not False and adapter_masks is not None and len(adapter_masks) != batch:
        raise ValueError("Adapter mask batch size differs")
    for i, (tokens, length, past, start) in enumerate(zip(token_rows, lengths, pasts, starts)):
        inputs[i, :length] = torch.tensor(tokens, dtype=torch.long, device=device)
        positions[i, :length] = torch.arange(start, start + length, device=device)
        attention[i, past_width - past:past_width + length] = 1
        if adapter_masks is None:
            token_mask[i, :length] = True
        elif adapter_masks is not False:
            if len(adapter_masks[i]) != length:
                raise ValueError("Adapter mask token length differs")
            token_mask[i, :length] = torch.tensor(adapter_masks[i], dtype=torch.bool, device=device)
    packed_cache = ()
    if past_width:
        template = next(c for c in caches if c)
        if any(c and len(c) != len(template) for c in caches):
            raise ValueError("Cache layer counts differ")
        packed_cache = tuple(tuple(torch.cat([
            F.pad(c[layer][kind], (0, 0, past_width - pasts[i], 0)) if c else
            template[layer][kind].new_zeros((1, template[layer][kind].shape[1],
                                            past_width, template[layer][kind].shape[-1]))
            for i, c in enumerate(caches)], dim=0) for kind in range(2))
            for layer in range(len(template)))
    kwargs = dict(input_ids=inputs, position_ids=positions, attention_mask=attention,
                  past_key_values=cache_from_tensors(packed_cache, runtime.model.config),
                  use_cache=True, return_dict=True)
    parameters = inspect.signature(runtime.model.forward).parameters
    if "cache_position" in parameters:
        kwargs["cache_position"] = torch.arange(past_width, past_width + width, device=device)
    keep = None
    if last_only and "logits_to_keep" in parameters:
        keep = sorted(set(n - 1 for n in lengths))
        kwargs["logits_to_keep"] = torch.tensor(keep, dtype=torch.long, device=device)
    controller = runtime._controller()
    gating = (contextlib.nullcontext() if controller is None else
              controller.disabled() if adapter_masks is False else controller.masked_tokens(token_mask))
    with torch.set_grad_enabled(training), gating:
        output = runtime.model(**kwargs)
    result_cache = cache_to_tensors(output.past_key_values)
    logits, unpacked = [], []
    for i, (length, past) in enumerate(zip(lengths, pasts)):
        if last_only:
            index = keep.index(length - 1) if keep is not None else length - 1
            logits.append(output.logits[i:i+1, index:index+1])
        else:
            logits.append(output.logits[i:i+1, :length])
        # The old left-padded portion and the new right-padded portion are
        # compacted independently. torch.cat preserves online-KV gradients.
        unpacked.append(tuple(tuple(torch.cat((
            t[i:i+1, :, past_width-past:past_width],
            t[i:i+1, :, past_width:past_width+length]), dim=-2)
            for t in layer) for layer in result_cache))
    return logits, unpacked


def prefill_batch(runtime, plans, method, *, training=False):
    if method == "full":
        return forward_batch(runtime, [p.token_ids for p in plans],
            adapter_masks=False, training=training, last_only=True)
    if method != "pic_attuner":
        raise ValueError("Training micro-batches support Full and PIC with adapters")
    # Frozen artifact building remains independent and never joins the graph.
    frozen = [runtime._warm(p, "pic")[0] for p in plans]
    caches, logits = [()] * len(plans), [None] * len(plans)
    cursors, segments = [0] * len(plans), [0] * len(plans)
    while True:
        active, tokens, masks = [], [], []
        for i, plan in enumerate(plans):
            while segments[i] < len(plan.segments) and plan.segments[segments[i]].start == cursors[i]:
                s = plan.segments[segments[i]]
                caches[i] = concat_cache(caches[i], move_cache(frozen[i][segments[i]], runtime.device))
                cursors[i] = s.end
                segments[i] += 1
            if cursors[i] == len(plan.token_ids):
                continue
            end = (plan.segments[segments[i]].start if segments[i] < len(plan.segments)
                   else len(plan.token_ids))
            if end <= cursors[i]:
                raise ValueError("Overlapping or unordered artifact segments")
            active.append(i)
            tokens.append(plan.token_ids[cursors[i]:end])
            masks.append(plan.adapter_mask[cursors[i]:end])
        if not active:
            break
        new_logits, new_caches = forward_batch(runtime, tokens,
            caches=[caches[i] for i in active], starts=[cursors[i] for i in active],
            adapter_masks=masks, training=training, last_only=True)
        for j, i in enumerate(active):
            logits[i], caches[i] = new_logits[j], new_caches[j]
            cursors[i] += len(tokens[j])
    if any(x is None for x in logits) or any(
            p.segments and p.segments[-1].end == len(p.token_ids) for p in plans):
        raise ValueError("Training prompts must end with an online generation suffix")
    return logits, caches


def response_logits_batch(runtime, prompt_logits, caches, responses, *, training=False,
                          adapter=False):
    if any(not r for r in responses):
        raise ValueError("Teacher response prefixes must be nonempty")
    active = [i for i, r in enumerate(responses) if len(r) > 1]
    result = list(prompt_logits)
    if active:
        continuation, _ = forward_batch(runtime, [responses[i][:-1] for i in active],
            caches=[caches[i] for i in active], adapter_masks=None if adapter else False,
            training=training)
        for j, i in enumerate(active):
            result[i] = torch.cat((prompt_logits[i], continuation[j]), dim=1)
    return result


def distillation_losses(runtime, teacher_plans, student_plans, responses):
    """Return one response-token-mean KL per example; never average padding."""
    with torch.no_grad():
        logits, caches = prefill_batch(runtime, teacher_plans, "full")
        logits = response_logits_batch(runtime, logits, caches, responses)
        teachers = [x.float().log_softmax(-1).detach() for x in logits]
        del logits, caches
    logits, caches = prefill_batch(runtime, student_plans, "pic_attuner", training=True)
    logits = response_logits_batch(runtime, logits, caches, responses, training=True, adapter=True)
    losses = []
    for teacher, student in zip(teachers, logits):
        log_student = student.float().log_softmax(-1)
        losses.append((teacher.exp() * (teacher - log_student)).sum(-1).mean())
    return torch.stack(losses)
