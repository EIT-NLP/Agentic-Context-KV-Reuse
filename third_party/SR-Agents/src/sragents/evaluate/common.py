"""Shared helpers for answer extraction across datasets."""

import re

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from an LLM response.

    Handles both well-formed `<think>...</think>answer` and truncation
    cases where generation is cut mid-thinking (unclosed tag).
    """
    if "<think>" not in text:
        return text
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.lstrip()

__all__ = ["strip_think_tags", "extract_from_trigger", "within_eps"]


# Trigger phrases used across datasets
_TRIGGERS = (
    "The answer is:",
    "the answer is:",
    "Therefore, the answer is",
    "therefore, the answer is",
)


def extract_from_trigger(raw_output: str) -> str | None:
    """Extract answer after the last occurrence of a trigger phrase.

    Returns None if no trigger found.
    """
    best_pos = -1
    best_trigger = ""
    for trigger in _TRIGGERS:
        pos = raw_output.rfind(trigger)
        if pos > best_pos:
            best_pos = pos
            best_trigger = trigger

    if best_pos == -1:
        return None

    after = raw_output[best_pos + len(best_trigger) :]
    answer = after.split("\n")[0].strip()
    answer = answer.rstrip(".").rstrip("/").strip()
    return answer


def within_eps(
    pred: float,
    gt: float,
    eps_ratio: float = 0.04,
    abs_floor: float = 1e-9,
) -> bool:
    """Check if ``pred`` is within relative tolerance of ``gt``.

    A small absolute floor is applied so that a ground truth of exactly
    zero still admits tiny non-zero predictions (e.g. ``1e-10`` is
    accepted as a match for ``0``).
    """
    eps = max(abs(gt) * eps_ratio, abs_floor)
    return gt - eps <= pred <= gt + eps
