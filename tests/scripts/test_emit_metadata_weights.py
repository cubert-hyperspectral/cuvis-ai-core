"""Tests for the ``weights:`` projection of scripts/emit_metadata.py."""

from __future__ import annotations

import importlib
import textwrap
import uuid
from pathlib import Path

import pytest
import yaml
from cuvis_ai_schemas.plugin import PluginWeightEntry, load_plugin_manifest

from scripts.emit_metadata import _describe_weight_drift, _main, emit

REV = "6d25af14a085ff9d3e1342c35bae7c87de4811f4"
SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"

# The node class emit() introspects: MinMaxNormalizer(eps: float, use_running_stats: bool, **kwargs).
_NODE_SOURCE = "from tests.fixtures.mock_nodes import MinMaxNormalizer\n"

_WEIGHTS_TEMPLATE = """
from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

WEIGHTS = (
    PluginWeightEntry(
        name="demo_weights",
        display_name="Demo weights",
        used_for=["Backbone"],
        repo_id="cubert-gmbh/demo",
        filename="demo.pt",
        revision="{rev}",
        sha256="{sha}",
        size_bytes={size},
        aux_files=[AuxFile(path="config.json", size_bytes=12, sha256="{sha}")],
        license="Apache-2.0",
        license_file="LICENSE",
        selected_by={selected_by},
        default={default},
        explicit_path_hparams={explicit},
        description="A demo weight.",
    ),
)
"""


def _make_plugin(
    tmp_path: Path,
    monkeypatch,
    *,
    weights: bool = True,
    weights_module: str = "weights",
    selected_by: str | None = "use_running_stats",
    default: bool = True,
    explicit: tuple[str, ...] = ("eps",),
    size: int = 1234,
) -> tuple[str, Path]:
    """Create an importable fake plugin package; returns (package name, manifest path)."""
    package = f"wplugin_{uuid.uuid4().hex[:8]}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "node.py").write_text(_NODE_SOURCE, encoding="utf-8")
    if weights:
        (root / f"{weights_module}.py").write_text(
            textwrap.dedent(
                _WEIGHTS_TEMPLATE.format(
                    rev=REV,
                    sha=SHA,
                    size=size,
                    selected_by=repr(selected_by),
                    default=default,
                    explicit=list(explicit),
                )
            ),
            encoding="utf-8",
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    manifest = tmp_path / f"{package}.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": package,
                "path": f"./{package}",
                "package_name": f"cuvis-ai-{package}",
                "capabilities": [{"class_name": f"{package}.node.MinMaxNormalizer"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return package, manifest


def test_weights_module_present_writes_block_after_capabilities(tmp_path, monkeypatch):
    package, manifest = _make_plugin(tmp_path, monkeypatch)
    assert emit(manifest, check=False) is True
    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    keys = list(doc)
    assert keys.index("weights") == keys.index("capabilities") + 1
    assert keys.index("package_name") < keys.index(
        "capabilities"
    )  # untouched keys keep their place
    (row,) = doc["weights"]
    assert row["name"] == "demo_weights" and row["size_bytes"] == 1234
    assert row["selected_by"] == "use_running_stats" and row["default"] is True
    assert row["explicit_path_hparams"] == ["eps"]
    # Default-valued fields are dropped from the projection...
    assert "kind" not in row and "summary" not in row and "aliases" not in row
    # ...and come back on load, so the manifest validates to the live declaration.
    loaded = load_plugin_manifest(manifest)
    module = importlib.import_module(f"{package}.weights")
    assert list(loaded.weights) == list(module.WEIGHTS)
    assert loaded.capabilities[0].class_name == f"{package}.node.MinMaxNormalizer"
    assert emit(manifest, check=True) is True


def test_weights_module_absent_leaves_the_key_untouched(tmp_path, monkeypatch):
    _, manifest = _make_plugin(tmp_path, monkeypatch, weights=False)
    assert emit(manifest, check=False) is True
    assert "weights" not in yaml.safe_load(manifest.read_text(encoding="utf-8"))
    # A hand-written block on a plugin without a weights module is left alone too.
    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    doc["weights"] = [
        {
            "name": "handwritten",
            "display_name": "Hand written",
            "used_for": ["Backbone"],
            "repo_id": "cubert-gmbh/x",
            "filename": "x.pt",
            "revision": REV,
            "sha256": SHA,
            "size_bytes": 1,
            "license": "MIT",
        }
    ]
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert emit(manifest, check=True) is True
    assert emit(manifest, check=False) is True
    assert (
        yaml.safe_load(manifest.read_text(encoding="utf-8"))["weights"][0]["name"]
        == "handwritten"
    )


def test_weights_module_override_flag(tmp_path, monkeypatch):
    package, manifest = _make_plugin(
        tmp_path, monkeypatch, weights_module="model_files"
    )
    assert emit(manifest, check=False) is True
    assert "weights" not in yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert emit(manifest, check=False, weights_module=f"{package}.model_files") is True
    assert (
        yaml.safe_load(manifest.read_text(encoding="utf-8"))["weights"][0]["name"]
        == "demo_weights"
    )
    assert (
        _main(
            [
                "--manifest",
                str(manifest),
                "--check",
                "--weights-module",
                f"{package}.model_files",
            ]
        )
        == 0
    )


def test_referential_validation_names_row_and_hparam(tmp_path, monkeypatch):
    _, manifest = _make_plugin(tmp_path, monkeypatch, selected_by="model_type")
    with pytest.raises(
        ValueError, match="weight 'demo_weights' names hyper-parameter 'model_type'"
    ):
        emit(manifest, check=False)
    _, manifest = _make_plugin(tmp_path, monkeypatch, explicit=("checkpoint_path",))
    with pytest.raises(ValueError, match="'checkpoint_path'"):
        emit(manifest, check=True)
    assert _main(["--manifest", str(manifest)]) == 1


def test_check_detects_weight_drift_and_names_the_field(tmp_path, monkeypatch):
    _, manifest = _make_plugin(tmp_path, monkeypatch)
    assert emit(manifest, check=False) is True
    doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    doc["weights"][0]["revision"] = "f" * 40
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert emit(manifest, check=True) is False
    assert _main(["--manifest", str(manifest), "--check"]) == 1
    # A manifest without the block at all is drift for a plugin that declares weights.
    del doc["weights"]
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    assert emit(manifest, check=True) is False
    # Regenerating repairs it.
    assert emit(manifest, check=False) is True and emit(manifest, check=True) is True


def test_describe_weight_drift():
    base = PluginWeightEntry(
        name="a",
        display_name="A",
        used_for=["Backbone"],
        repo_id="cubert-gmbh/a",
        filename="a.pt",
        revision=REV,
        sha256=SHA,
        size_bytes=1,
        license="MIT",
    )
    changed = base.model_copy(update={"size_bytes": 2, "license": "Apache-2.0"})
    other = base.model_copy(update={"name": "b"})
    assert _describe_weight_drift(None, [base]) == "the manifest has no weights block"
    assert _describe_weight_drift([base], [changed]) == "'a': license, size_bytes"
    assert _describe_weight_drift([base], [base, other]) == "'b' only in WEIGHTS"
    assert _describe_weight_drift([base, other], [other]) == "'a' only in the manifest"
    assert _describe_weight_drift([base, other], [other, base]) == "row order"
    assert _describe_weight_drift([base], [base]) == "no difference"


def test_weights_module_without_the_tuple_is_an_error(tmp_path, monkeypatch):
    package, manifest = _make_plugin(tmp_path, monkeypatch, weights=False)
    (tmp_path / package / "weights.py").write_text(
        "NOT_WEIGHTS = ()\n", encoding="utf-8"
    )
    importlib.invalidate_caches()
    with pytest.raises(ValueError, match="defines no WEIGHTS tuple"):
        emit(manifest, check=False)
