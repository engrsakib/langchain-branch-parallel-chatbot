"""Small shared helpers for input hygiene, UI formatting, and session state."""

import re
from collections.abc import Iterable, MutableMapping
from typing import Any, TypeVar

T = TypeVar("T")

# Control characters that should never reach a prompt. Tab and newline are kept:
# people paste code into this chatbot, and indentation is part of the question.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACES = re.compile(r"[^\S\n]+\n")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_BADGE_UNSAFE = re.compile(r"[\[\]]")


def sanitize_query(text: str, *, max_length: int | None = None) -> str:
    """Clean a raw user message without disturbing meaningful whitespace.

    Line breaks and indentation survive, so pasted code reaches the model intact;
    only control characters, trailing spaces and runs of blank lines are removed.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARACTERS.sub("", cleaned)
    cleaned = _TRAILING_SPACES.sub("\n", cleaned)
    cleaned = _EXCESS_BLANK_LINES.sub("\n\n", cleaned)
    cleaned = cleaned.strip()

    if max_length is not None and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned


def format_keyword_badges(keywords: Iterable[str], *, color: str = "gray") -> str:
    """Render keywords as Streamlit background badges on one line.

    Square brackets would terminate the badge directive early, so they are
    dropped rather than escaped.
    """
    badges = []
    for keyword in keywords:
        label = _BADGE_UNSAFE.sub("", str(keyword)).strip()
        if label:
            badges.append(f":{color}-background[{label}]")
    return " ".join(badges)


def format_confidence(confidence: float) -> str:
    """Format a 0.0-1.0 score as a whole percentage."""
    return f"{confidence:.0%}"


def state_list(state: MutableMapping[str, Any], key: str, item_type: type[T]) -> list[T]:
    """Read a list of ``item_type`` out of session state, repairing bad values.

    Streamlit keeps session state alive across script edits, so a reload can
    leave behind entries built from an older version of a class. Dropping those
    silently costs the user their scrollback; crashing costs them the whole tab.
    """
    value = state.get(key)
    if not isinstance(value, list):
        state[key] = []
        return state[key]

    kept = [item for item in value if isinstance(item, item_type)]
    if len(kept) != len(value):
        state[key] = kept
    return state[key]
