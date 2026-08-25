"""Warn-and-default parsing for numeric environment knobs.

Single implementation shared by every orchestrator env var that carries a
number (spawn timeouts, cache eviction limits): invalid or out-of-range
values log a warning and fall back to the default instead of raising, so
a typo in an operator's shell profile never takes the server down.
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar

from loguru import logger

_Number = TypeVar("_Number", int, float)


def number_from_env(
    env_name: str,
    default: _Number,
    *,
    cast: Callable[[str], _Number],
    allow_zero: bool = True,
) -> _Number:
    """Read a non-negative number from ``env_name``; fall back to ``default``.

    ``cast`` is ``int`` or ``float``. ``allow_zero=True`` accepts ``0``
    (the conventional "disabled" value for cache limits); ``allow_zero=False``
    demands a strictly positive value (timeouts). Unset, unparsable, or
    out-of-range values log a warning and return ``default``.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning(f"{env_name}={raw!r} is not a number; using {default}.")
        return default
    if value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        logger.warning(f"{env_name}={raw!r} must be {bound}; using {default}.")
        return default
    return value
