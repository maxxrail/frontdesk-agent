"""Extended thinking is off for every call this application makes.

Every call here is short and bounded: an SMS reply, a tool choice, a
pass/fail grade, a simulated client's next text. Newer models think by
default, and thinking tokens count against ``max_tokens``, so with the small
limits used here a reply can come back cut off or empty. In the PR #4 eval
runs that silently dropped 52 of 129 judge verdicts and ended simulated
conversations early. Thinking would also multiply the cost per call.

Each model family switches it off differently, and some cannot at all.
"""

from typing import Any

# (model prefix, request value) -- longest prefix first.
_OFF: tuple[tuple[str, dict[str, Any] | None], ...] = (
    # Rejects {"type": "disabled"}; "between_tools" is its off switch.
    ("claude-sonnet-5-5", {"type": "between_tools"}),
    # Thinks by default; "disabled" is accepted at effort high or below
    # (its default is medium).
    ("claude-haiku-5-5", {"type": "disabled"}),
    # Haiku 4.5 does not think unless asked; send nothing.
    ("claude-haiku-4-5", None),
)


def thinking_off(model: str) -> dict[str, Any] | None:
    """The ``thinking`` request value that turns thinking off for ``model``,
    or None when nothing needs sending (or nothing can be sent)."""
    for prefix, value in _OFF:
        if model.startswith(prefix):
            return dict(value) if value else None
    return None
