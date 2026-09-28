"""Selector helpers shared by the data modules."""

from __future__ import annotations

import re
from collections.abc import Sequence

_RANGE_RE = re.compile(r"^(\d+)-(\d+)(?::(\d+))?$")


def expand_range_selectors(items: Sequence[int | str]) -> list[int | str]:
    """Expand inclusive ``start-stop[:step]`` range strings into integer selectors.

    Each item is a split selector: an ``int``, a non-range ``str`` key (e.g. a
    TIFF filename stem), or a range string such as ``"0-100"`` or ``"0-10:2"``.
    Range strings expand to their inclusive integer sequence (``"0-3"`` ->
    ``[0, 1, 2, 3]``, ``"0-10:2"`` -> ``[0, 2, 4, 6, 8, 10]``); ints and non-range
    strings pass through unchanged. Open-ended ranges (e.g. ``"0-"``) are not
    handled here because they need the sample-universe size; resolve those in the
    module that knows how many samples exist.
    """
    expanded: list[int | str] = []
    for item in items:
        if isinstance(item, str):
            match = _RANGE_RE.match(item.strip())
            if match is not None:
                start, stop = int(match.group(1)), int(match.group(2))
                step = int(match.group(3)) if match.group(3) is not None else 1
                if step <= 0:
                    raise ValueError(f"range selector {item!r}: step must be > 0")
                if start > stop:
                    raise ValueError(
                        f"range selector {item!r}: start {start} must be <= stop {stop}"
                    )
                expanded.extend(range(start, stop + 1, step))
                continue
        expanded.append(item)
    return expanded
