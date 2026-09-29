"""``auto_register_package`` reports a module that fails to import through loguru."""

import sys
from pathlib import Path

from loguru import logger

from cuvis_ai_core.utils.node_registry import NodeRegistry


def test_failed_module_import_is_a_loguru_warning(tmp_path: Path, monkeypatch):
    pkg = tmp_path / "broken_nodes_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n.startswith("broken_nodes_pkg")]:
        monkeypatch.delitem(sys.modules, name)

    messages: list[str] = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        count = NodeRegistry.auto_register_package("broken_nodes_pkg")
    finally:
        logger.remove(handler_id)

    assert count == 0
    assert any("broken_nodes_pkg.broken" in m and "boom" in m for m in messages)
