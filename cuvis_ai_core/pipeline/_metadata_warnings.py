"""Runtime warning for nodes that lack explicit category / tags metadata."""

from __future__ import annotations

import warnings

from cuvis_ai_schemas.enums import NodeCategory, NodeTag

from cuvis_ai_core.node.node import Node


class MissingNodeMetadataWarning(UserWarning):
    """Emitted when a Node enters a pipeline without explicit ``_category`` or ``_tags``."""


# One warning per class is enough — a pipeline with 100 instances of the same
# unannotated node should surface the issue once, not 100 times.
_warned_classes: set[type[Node]] = set()


def warn_if_metadata_missing(node: Node) -> None:
    """Issue a ``MissingNodeMetadataWarning`` if the node's class still
    inherits the default ``UNSPECIFIED`` category or empty tag set.

    Idempotent per class: subsequent instances of the same class are silent.
    The warning is suppressible via the standard ``warnings`` filters; tests
    that intentionally exercise unannotated nodes should wrap construction
    in ``warnings.catch_warnings()`` and call ``simplefilter("ignore",
    MissingNodeMetadataWarning)``.

    A class that explicitly opts in via ``_tags = frozenset({NodeTag.UNSPECIFIED})``
    is also treated as missing — that pseudo-annotation is semantically empty
    and we don't want plugin authors silencing the warning that way.
    """
    cls = type(node)
    if cls in _warned_classes:
        return

    issues: list[str] = []
    if cls.get_category() is NodeCategory.UNSPECIFIED:
        issues.append("_category")
    tags = cls.get_tags()
    if not tags or tags == frozenset({NodeTag.UNSPECIFIED}):
        issues.append("_tags")

    if not issues:
        return

    fields = " and ".join(issues)
    warnings.warn(
        f"Node class {cls.__module__}.{cls.__qualname__} is missing "
        f"explicit {fields}. Set them on the class body — see "
        f"ALL-5187 phase 3 for the decision flowchart.",
        MissingNodeMetadataWarning,
        stacklevel=3,
    )
    _warned_classes.add(cls)
