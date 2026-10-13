"""What every command shares: the reply, the error, and reading a command line."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class CommandError(Exception):
    pass


@dataclass
class Reply:
    text: str = ""
    changed: set = field(default_factory=set)  # buffers, data, fits, sources, masks, centre
    show: Optional[int] = None  # the buffer to show
    values: Dict[str, float] = field(default_factory=dict)  # numbers for the panels
    html: str = ""  # the text with colours and links, when there is such a version (ls)


def split_line(line: str) -> Tuple[str, List[str], Dict[str, str]]:
    """``verb``, positional words and KEY=value pairs of a command line."""
    try:
        tokens = shlex.split(line, comments=True)
    except ValueError as e:
        raise CommandError(f"cannot read the line: {e}")
    if not tokens:
        return "", [], {}
    words, keys = [], {}
    for token in tokens[1:]:
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", token)
        if m:
            keys[m.group(1).upper()] = m.group(2)
        else:
            words.append(token)
    return tokens[0].lower(), words, keys


def number(text, what: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        raise CommandError(f"{what} must be a number, not {text!r}")


def integer(text, what: str) -> int:
    value = number(text, what)
    if value != int(value):
        raise CommandError(f"{what} must be a whole number, not {text!r}")
    return int(value)
