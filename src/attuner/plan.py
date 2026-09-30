"""Locate artifacts in the unchanged official chat template output."""
from __future__ import annotations

import json
import bisect
from dataclasses import dataclass
from collections.abc import Mapping

_BOUNDARIES = {}


def message_boundaries(tokenizer, messages, artifacts, rendered, enable_thinking):
    points = sorted({a[k] for a in artifacts for k in ('message_start', 'message_end')})
    if not points or points[0] < 0 or points[-1] > len(messages):
        raise ValueError('Invalid artifact message boundaries')
    key = (id(tokenizer), enable_thinking, tuple(points),
           json.dumps(messages[:points[-1]], sort_keys=True, ensure_ascii=False))
    if key not in _BOUNDARIES:
        bounds = {}
        # Qwen3 renders the final historical assistant differently when a prefix
        # is rendered alone. An appended sentinel user preserves historical
        # treatment; strip its exact template suffix and verify against the real
        # full prompt. The sentinel is never tokenized or sent to the model.
        sentinel = {'role':'user', 'content':'ATTUNER_BOUNDARY_SENTINEL_8b91b520'}
        sentinel_suffix = None
        for point in points:
            prefix = tokenizer.apply_chat_template(messages[:point], tokenize=False,
                add_generation_prompt=False, enable_thinking=enable_thinking) if point else ''
            if not rendered.startswith(prefix):
                if sentinel_suffix is None:
                    anchor = tokenizer.apply_chat_template(messages[:1], tokenize=False,
                        add_generation_prompt=False, enable_thinking=enable_thinking)
                    probe = tokenizer.apply_chat_template(messages[:1]+[sentinel], tokenize=False,
                        add_generation_prompt=False, enable_thinking=enable_thinking)
                    if not probe.startswith(anchor):
                        raise ValueError('Chat template does not expose stable message boundaries')
                    sentinel_suffix = probe[len(anchor):]
                probe = tokenizer.apply_chat_template(messages[:point]+[sentinel], tokenize=False,
                    add_generation_prompt=False, enable_thinking=enable_thinking)
                if not sentinel_suffix or not probe.endswith(sentinel_suffix):
                    raise ValueError('Chat template sentinel suffix is context-dependent')
                prefix = probe[:-len(sentinel_suffix)]
                if not rendered.startswith(prefix):
                    raise ValueError('Chat template changes earlier messages: cannot derive exact artifact boundary')
            bounds[point] = len(prefix)
        if len(_BOUNDARIES) >= 256:
            _BOUNDARIES.pop(next(iter(_BOUNDARIES)))
        _BOUNDARIES[key] = bounds
    return _BOUNDARIES[key]


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    token_ids: tuple[int, ...]
    document_id: str

    @property
    def token_count(self):
        return self.end - self.start

@dataclass(frozen=True)
class PromptPlan:
    token_ids: tuple[int, ...]
    segments: tuple[Segment, ...]
    adapter_mask: tuple[bool, ...]

    @property
    def prompt_token_ids(self):
        return self.token_ids


def build_plan(tokenizer, row, *, enable_thinking=False) -> PromptPlan:
    messages = row["messages"]
    artifacts = row.get("artifacts", row.get("skills", []))
    kwargs = dict(add_generation_prompt=True, enable_thinking=enable_thinking)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if row.get('artifact_tokenization') == 'independent':
        return build_concatenated_plan(tokenizer, rendered, artifacts)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        if len(tokens) != 1:
            raise ValueError("Only one prompt per plan is supported")
        tokens = tokens[0]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(tokens):
        raise RuntimeError("Official chat-template token identity mismatch")
    offsets = encoded["offset_mapping"]
    explicit = bool(artifacts and 'message_start' in artifacts[0])
    bounds = message_boundaries(tokenizer, messages, artifacts, rendered, enable_thinking) if explicit else None
    starts, ends = [a for a,b in offsets], [b for a,b in offsets]
    segments, char_cursor, token_cursor = [], 0, 0
    for index, artifact in enumerate(artifacts):
        if explicit:
            char_start, char_end = bounds[artifact['message_start']], bounds[artifact['message_end']]
        else:
            content = artifact.get("content", artifact.get("text"))
            if not isinstance(content, str) or not content:
                raise ValueError(f"Artifact {index} has no nonempty verbatim content")
            char_start = rendered.find(content, char_cursor)
            if char_start < 0:
                raise ValueError(f"Artifact {index} absent from official prompt")
            char_end = char_start + len(content)
        start, end = bisect.bisect_right(ends, char_start), bisect.bisect_left(starts, char_end)
        if start >= end or start < token_cursor:
            raise ValueError("Empty or overlapping artifact token span")
        if explicit and (offsets[start][0] != char_start or offsets[end-1][1] != char_end):
            raise ValueError('Artifact message boundary crosses a token; no silent token overlap allowed')
        document_id = str(artifact.get("document_id", artifact.get("skill_id",
                                   artifact.get("id", index))))
        segments.append(Segment(start, end, tuple(tokens[start:end]), document_id))
        char_cursor, token_cursor = char_end, end
    # Query adapters apply to every online prompt token, including preamble and
    # gaps. Artifact encoding is frozen; subtract its spans below.
    mask = [True] * len(tokens)
    if segments:
        mask[segments[-1].end:] = [True] * (len(tokens) - segments[-1].end)
    # Explicit question supports official templates that put the question first.
    question = row.get("question", row.get("raw", {}).get("question"))
    if isinstance(question, str) and question:
        start = rendered.rfind(question)
        if start >= 0:
            for i, (a, b) in enumerate(offsets):
                if b > a and b > start and a < start + len(question):
                    mask[i] = True
    if not segments:
        # No reusable region: allow user text and generation suffix, keep system off.
        last_user = next((m.get("content") for m in reversed(messages)
                          if m.get("role") == "user"), None)
        if isinstance(last_user, str) and last_user:
            start = rendered.rfind(last_user)
            for i, (a, b) in enumerate(offsets):
                if start >= 0 and b > start:
                    mask[i] = True
    for segment in segments:
        mask[segment.start:segment.end] = [False] * segment.token_count
    return PromptPlan(tuple(tokens), tuple(segments), tuple(mask))


def build_concatenated_plan(tokenizer, rendered, artifacts) -> PromptPlan:
    """Explicit pure-concatenation protocol: tokenize each artifact separately.

    Full and PIC consume exactly this same token sequence. It need not equal
    tokenizing the concatenated string at once: cross-boundary BPE merges would
    prevent an artifact's tokens from being independently reusable. No separator
    or padding token is inserted. Only callers opting into this protocol use it.
    """
    tokens, segments, cursor = [], [], 0
    for index, artifact in enumerate(artifacts):
        content = artifact.get('content', artifact.get('text'))
        if not isinstance(content, str) or not content:
            raise ValueError('Concatenated artifact must have nonempty text')
        start = rendered.find(content, cursor)
        if start < 0 or (index and start != cursor):
            raise ValueError('Independent-tokenization artifacts must be adjacent in the prompt')
        if index == 0:
            tokens.extend(tokenizer.encode(rendered[:start], add_special_tokens=False))
        part = tuple(tokenizer.encode(content, add_special_tokens=False))
        if not part:
            raise ValueError('Artifact produced no tokens')
        identifier = str(artifact.get('document_id', artifact.get('skill_id', artifact.get('id', index))))
        segments.append(Segment(len(tokens), len(tokens)+len(part), part, identifier))
        tokens.extend(part)
        cursor = start + len(content)
    if not segments:
        raise ValueError('Independent-tokenization protocol requires artifacts')
    tokens.extend(tokenizer.encode(rendered[cursor:], add_special_tokens=False))
    mask = [True] * len(tokens)
    for segment in segments:
        mask[segment.start:segment.end] = [False] * segment.token_count
    return PromptPlan(tuple(tokens), tuple(segments), tuple(mask))
