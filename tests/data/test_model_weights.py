"""Tests for the model-weight registry, resolver, provisioning downloader and its CLI.

``huggingface_hub.hf_hub_download`` is replaced by a recorder that writes files
into the real Hugging Face cache layout under ``tmp_path``; every status,
resolve, export and import path runs against that layout without network.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from cuvis_ai_schemas.plugin import (
    AuxFile,
    GitPluginSource,
    PluginWeightEntry,
    write_plugin_manifest,
)
from filelock import FileLock

from cuvis_ai_core.data import _provisioning as prov
from cuvis_ai_core.data import model_weights as mw
from cuvis_ai_core.data._model_weights_cli import build_cli
from cuvis_ai_core.data.model_weights import (
    HF_ORG,
    TRAINED_PIPELINES,
    USED_FOR_LABELS,
    WEIGHTS_HOSTS,
    ModelDownloadError,
    ModelRegistryConflict,
    ModelWeights,
    ModelWeightsMissingError,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

REV_SAM3 = "6d25af14a085ff9d3e1342c35bae7c87de4811f4"
REV_ETAM = "3dfd0228d7774b94c24116cf729e03c209ff448a"
SAM3_BYTES = b"sam3-weights-bytes"
CONFIG_BYTES = b'{"model": "sam3"}'
ETAM_S_BYTES = b"efficienttam-small-bytes"
ETAM_TI_BYTES = b"efficienttam-tiny"
CONTENT = {
    "sam3.pt": SAM3_BYTES,
    "config.json": CONFIG_BYTES,
    "efficienttam_s.pt": ETAM_S_BYTES,
    "efficienttam_ti.pt": ETAM_TI_BYTES,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sam3_entry(**overrides) -> PluginWeightEntry:
    data = dict(
        name="sam3",
        display_name="SAM3",
        summary="Magic wand, propagation, text prompts",
        used_for=[
            "Point expansion",
            "Propagation",
            "Text prompts",
            "Segment everything",
        ],
        repo_id=f"{HF_ORG}/sam3",
        filename="sam3.pt",
        revision=REV_SAM3,
        sha256=_sha(SAM3_BYTES),
        size_bytes=len(SAM3_BYTES),
        aux_files=[
            AuxFile(
                path="config.json",
                size_bytes=len(CONFIG_BYTES),
                sha256=_sha(CONFIG_BYTES),
            )
        ],
        license="SAM License",
        license_file="LICENSE",
        explicit_path_hparams=["checkpoint_path"],
        description="SAM3 checkpoint (mirror of facebook/sam3)",
    )
    data.update(overrides)
    return PluginWeightEntry(**data)


def etam_entry(
    name: str, content: bytes, *, default: bool = False, aliases=()
) -> PluginWeightEntry:
    return PluginWeightEntry(
        name=name,
        display_name=f"RTSAM ({name})",
        summary="Point expansion, propagation",
        used_for=["Point expansion", "Propagation"],
        repo_id=f"{HF_ORG}/efficient-track-anything",
        filename=f"{name}.pt",
        revision=REV_ETAM,
        sha256=_sha(content),
        size_bytes=len(content),
        license="Apache-2.0",
        license_file="LICENSE",
        aliases=list(aliases),
        selected_by="model_type",
        default=default,
        explicit_path_hparams=["model_dir"],
    )


@pytest.fixture(autouse=True)
def fresh_registry():
    ModelWeights.reset()
    yield
    ModelWeights.reset()


@pytest.fixture
def registry() -> dict[str, PluginWeightEntry]:
    entries = {
        "sam3": sam3_entry(),
        "efficienttam_s": etam_entry(
            "efficienttam_s", ETAM_S_BYTES, default=True, aliases=("efficienttam",)
        ),
        "efficienttam_ti": etam_entry("efficienttam_ti", ETAM_TI_BYTES),
    }
    ModelWeights.register("sam3", [entries["sam3"]])
    ModelWeights.register(
        "rtsam2", [entries["efficienttam_s"], entries["efficienttam_ti"]]
    )
    return entries


def _snapshot_dir(
    cache: Path, entry: PluginWeightEntry, revision: str | None = None
) -> Path:
    return (
        cache
        / ModelWeights.cache_dir_token(entry.repo_id)
        / "snapshots"
        / (revision or entry.revision)
    )


def _seed(
    cache: Path,
    entry: PluginWeightEntry,
    *,
    revision: str | None = None,
    write_ref: bool = True,
    truncate: bool = False,
    skip_aux: bool = False,
) -> Path:
    """Lay out one weight (primary + aux) in the HF cache layout; returns the primary path."""
    snap = _snapshot_dir(cache, entry, revision)
    primary = snap / entry.filename
    primary.parent.mkdir(parents=True, exist_ok=True)
    data = CONTENT[Path(entry.filename).name]
    primary.write_bytes(data[:-1] if truncate else data)
    if not skip_aux:
        for aux in entry.aux_files:
            (snap / aux.path).parent.mkdir(parents=True, exist_ok=True)
            (snap / aux.path).write_bytes(CONTENT[Path(aux.path).name])
    if write_ref:
        ref = snap.parent.parent / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(snap.name)
    return primary


@pytest.fixture
def fake_hub(monkeypatch):
    """Install a recording hf_hub_download that writes CONTENT into the cache layout."""

    def factory(
        *, content: dict[str, bytes] | None = None, fail: BaseException | None = None
    ):
        table = dict(CONTENT if content is None else content)

        def _fake(
            repo_id,
            filename,
            revision=None,
            token=None,
            cache_dir=None,
            force_download=False,
        ):
            _fake.calls.append(
                {
                    "repo_id": repo_id,
                    "filename": filename,
                    "revision": revision,
                    "token": token,
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                }
            )
            if fail is not None:
                raise fail
            target = (
                Path(cache_dir)
                / ModelWeights.cache_dir_token(repo_id)
                / "snapshots"
                / (revision or "0" * 40)
                / filename
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(table.get(Path(filename).name, b"unknown-content"))
            return str(target)

        _fake.calls = []
        monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake)
        return _fake

    return factory


@pytest.fixture
def no_network(monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - only hit on regression
        raise AssertionError("hf_hub_download must not be called")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)


class Recorder(prov.ProgressEmitter):
    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.events: list[dict] = []

    def _write(self, event: dict) -> None:
        self.events.append(event)


def _hub_error(cls, status: int | None = None):
    exc = cls.__new__(cls)
    Exception.__init__(exc, "boom")
    exc.response = SimpleNamespace(status_code=status) if status is not None else None
    return exc


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


def test_builtin_rows_are_the_trained_pipelines():
    assert ModelWeights.names() == sorted(e.name for e in TRAINED_PIPELINES)
    for row in ModelWeights.rows():
        assert row.source == "dict"
        assert row.plugin == "dinomaly"
        assert row.entry.kind == "trained_pipeline"
        assert row.plugin_default is False
        assert row.entry.aux_files[0].path.endswith(".yaml")


def test_registry_rows_are_pinned_and_mirrored(registry):
    for row in ModelWeights.rows():
        e = row.entry
        assert e.repo_id.startswith(f"{HF_ORG}/"), row.name
        assert _HEX40.match(e.revision) and _HEX64.match(e.sha256), row.name
        for aux in e.aux_files:
            assert _HEX64.match(aux.sha256) and aux.size_bytes > 0
        assert set(e.used_for) <= set(USED_FOR_LABELS)


def test_register_is_idempotent_on_full_entry_equality(registry):
    before = ModelWeights.names()
    ModelWeights.register("sam3", [sam3_entry()])
    assert ModelWeights.names() == before
    with pytest.raises(ModelRegistryConflict, match="different content"):
        ModelWeights.register("sam3", [sam3_entry(summary="Something else")])
    with pytest.raises(ModelRegistryConflict, match="different content"):
        ModelWeights.register("other_plugin", [sam3_entry()])


def test_register_rejects_alias_or_name_collision_across_plugins(registry):
    clash_name = etam_entry("efficienttam", ETAM_TI_BYTES)  # equals rtsam2's alias
    with pytest.raises(ModelRegistryConflict, match="one namespace"):
        ModelWeights.register("another", [clash_name])
    clash_alias = sam3_entry(name="sam3_other", aliases=["sam3"], selected_by="variant")
    with pytest.raises(ModelRegistryConflict, match="one namespace"):
        ModelWeights.register("another", [clash_alias])
    assert "efficienttam" not in ModelWeights.names()


def test_register_rejects_unknown_used_for_label():
    with pytest.raises(ModelRegistryConflict, match="not in the shared vocabulary"):
        ModelWeights.register(
            "p", [sam3_entry(used_for=["Point expansion", "Teleportation"])]
        )


def test_get_resolves_aliases_and_names_unknown_keys(registry):
    assert ModelWeights.get("efficienttam").name == "efficienttam_s"
    assert ModelWeights.get("sam3").plugin == "sam3"
    with pytest.raises(
        ModelDownloadError, match="Unknown model 'nope'. Known: .*efficienttam_s"
    ):
        ModelWeights.get("nope")


def test_plugin_default_and_total_bytes_computed(registry):
    rows = {r.name: r for r in ModelWeights.rows()}
    assert rows["sam3"].plugin_default is True  # selected_by None
    assert rows["efficienttam_s"].plugin_default is True  # default=True
    assert rows["efficienttam_ti"].plugin_default is False
    assert rows["dinomaly_lentils_cir"].plugin_default is False  # trained pipeline
    assert rows["sam3"].total_bytes == len(SAM3_BYTES) + len(CONFIG_BYTES)
    assert rows["sam3"].family == "sam3"
    assert (
        rows["efficienttam_ti"].cache_dir_name
        == f"models--{HF_ORG}--efficient-track-anything"
    )


def test_list_payload_matches_the_schema_contract(registry):
    payload = ModelWeights.list_payload()
    assert payload["schema_version"] == 1
    assert payload["used_for_labels"] == list(USED_FOR_LABELS)
    assert (
        payload["weights_hosts"] == list(WEIGHTS_HOSTS)
        and "huggingface.co" in WEIGHTS_HOSTS
    )
    names = [m["name"] for m in payload["models"]]
    assert names == sorted(names)
    spec_keys = set(prov.load_schema("model_list")["$defs"]["spec"]["required"])
    for model in payload["models"]:
        assert set(model) == spec_keys, model["name"]
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(prov.load_schema("model_list")).validate(payload)


def test_index_json_is_deterministic(registry):
    payload = ModelWeights.list_payload()
    text = mw.index_json(payload)
    assert text == mw.index_json(ModelWeights.list_payload())
    assert text.endswith("}\n") and json.loads(text) == payload


def _write_manifest(dir_: Path, plugin: str, weights) -> Path:
    manifest = GitPluginSource.model_validate(
        {
            "name": plugin,
            "repo": f"https://github.com/cubert-hyperspectral/cuvis-ai-{plugin}.git",
            "tag": "v0.5.0",
            "capabilities": [{"class_name": f"cuvis_ai_{plugin}.node.Node"}],
            "weights": [w.model_dump(mode="json") for w in weights],
        }
    )
    path = dir_ / f"{plugin}.yaml"
    write_plugin_manifest(manifest, path)
    return path


def test_load_manifests_merges_weights_blocks(tmp_path):
    _write_manifest(tmp_path, "sam3", [sam3_entry()])
    _write_manifest(
        tmp_path, "rtsam2", [etam_entry("efficienttam_s", ETAM_S_BYTES, default=True)]
    )
    _write_manifest(tmp_path, "plain", [])
    assert ModelWeights.load_manifests([tmp_path]) == 2
    rows = {r.name: r for r in ModelWeights.rows()}
    assert rows["sam3"].source == "manifest" and rows["sam3"].plugin == "sam3"
    assert (
        rows["efficienttam_s"].plugin == "rtsam2"
        and rows["efficienttam_s"].pin_mismatch is False
    )
    assert (
        ModelWeights.load_manifests([tmp_path]) == 0
    )  # reading the same manifests again is a no-op


def test_two_manifests_same_name_raise_conflict(tmp_path):
    _write_manifest(tmp_path, "sam3", [sam3_entry()])
    _write_manifest(tmp_path, "sam3_fork", [sam3_entry()])
    with pytest.raises(ModelRegistryConflict, match="manifests of both"):
        ModelWeights.load_manifests([tmp_path])


def test_duplicate_plugin_name_across_dirs_errors(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write_manifest(tmp_path / "a", "sam3", [sam3_entry()])
    _write_manifest(tmp_path / "b", "sam3", [sam3_entry()])
    with pytest.raises(ValueError, match="Duplicate plugin name"):
        ModelWeights.load_manifests([tmp_path / "a", tmp_path / "b"])


def test_precedence_plugin_only_manifest_only_both_equal_both_mismatched(
    tmp_path, capsys
):
    other_rev = "f" * 40
    # manifest only
    _write_manifest(tmp_path, "sam3", [sam3_entry()])
    ModelWeights.load_manifests([tmp_path])
    assert ModelWeights.get("sam3").source == "manifest"
    # both equal: the plugin registers what the manifest already says
    ModelWeights.register("sam3", [sam3_entry()])
    row = ModelWeights.get("sam3")
    assert row.source == "plugin" and row.pin_mismatch is False
    assert capsys.readouterr().err == ""
    # plugin first, manifest with another pin: plugin wins, mismatch flagged, one warning
    ModelWeights.reset()
    ModelWeights.register("sam3", [sam3_entry()])
    _write_manifest(tmp_path, "sam3", [sam3_entry(revision=other_rev)])
    ModelWeights.load_manifests([tmp_path])
    ModelWeights.load_manifests([tmp_path])
    row = ModelWeights.get("sam3")
    assert (
        row.source == "plugin"
        and row.pin_mismatch is True
        and row.entry.revision == REV_SAM3
    )
    err = capsys.readouterr().err
    assert err.count("warning: weight 'sam3'") == 1
    assert REV_SAM3[:12] in err and other_rev[:12] in err
    assert ModelWeights.entries()[-1]["pin_mismatch"] is True or any(
        m["pin_mismatch"] for m in ModelWeights.entries()
    )
    # manifest first, plugin with another pin: the plugin replaces the row and flags it
    ModelWeights.reset()
    ModelWeights.load_manifests([tmp_path])  # manifest at other_rev
    ModelWeights.register("sam3", [sam3_entry()])
    row = ModelWeights.get("sam3")
    assert (
        row.source == "plugin"
        and row.entry.revision == REV_SAM3
        and row.pin_mismatch is True
    )


def test_default_plugins_dirs_follows_cuvis_ai_importability(monkeypatch, tmp_path):
    import importlib.util as ilu

    monkeypatch.setattr(ilu, "find_spec", lambda name: None)
    assert mw.default_plugins_dirs() == []
    plugins = tmp_path / "cuvis_ai" / "configs" / "plugins"
    plugins.mkdir(parents=True)
    fake_spec = SimpleNamespace(origin=str(tmp_path / "cuvis_ai" / "__init__.py"))
    monkeypatch.setattr(
        ilu, "find_spec", lambda name: fake_spec if name == "cuvis_ai" else None
    )
    assert mw.default_plugins_dirs() == [plugins]


# ----------------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------------


def test_status_absent_when_cache_empty(tmp_path, registry):
    st = ModelWeights.status("sam3", tmp_path)
    assert st.state == "absent" and st.present is False and st.exists is False
    assert st.partial_bytes is None and st.size_ok is None and st.aux_ok is False
    assert (
        st.path
        == tmp_path / f"models--{HF_ORG}--sam3" / "snapshots" / REV_SAM3 / "sam3.pt"
    )


def test_status_present_at_pinned_revision(tmp_path, registry):
    _seed(tmp_path, registry["sam3"])
    st = ModelWeights.status("sam3", tmp_path)
    assert st.state == "present" and st.present and st.size_ok and st.aux_ok
    assert st.sha256_ok is None  # not verified unless asked


def test_status_ignores_other_snapshot_and_refs_main(tmp_path, registry):
    _seed(tmp_path, registry["sam3"], revision="a" * 40, write_ref=True)
    assert ModelWeights.status("sam3", tmp_path).state == "absent"


def test_status_size_mismatch_is_damaged(tmp_path, registry):
    _seed(tmp_path, registry["sam3"], truncate=True)
    st = ModelWeights.status("sam3", tmp_path)
    assert (
        st.exists and st.size_ok is False and st.state == "damaged" and not st.present
    )


def test_status_missing_aux_is_damaged(tmp_path, registry):
    _seed(tmp_path, registry["sam3"], skip_aux=True)
    st = ModelWeights.status("sam3", tmp_path)
    assert st.aux_ok is False and st.state == "damaged"


def test_status_partial_reports_the_incomplete_blob(tmp_path, registry):
    blobs = tmp_path / f"models--{HF_ORG}--sam3" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"12345")
    st = ModelWeights.status("sam3", tmp_path)
    assert st.state == "partial" and st.partial_bytes == 5


def test_status_verify_sets_sha_ok_true_and_false(tmp_path, registry):
    _seed(tmp_path, registry["sam3"])
    assert ModelWeights.status("sam3", tmp_path, verify=True).sha256_ok is True
    snap = _snapshot_dir(tmp_path, registry["sam3"])
    (snap / "sam3.pt").write_bytes(b"x" * len(SAM3_BYTES))  # same size, other bytes
    st = ModelWeights.status("sam3", tmp_path, verify=True)
    assert st.state == "present" and st.sha256_ok is False


def test_status_all_cleans_stale_import_staging(tmp_path, registry):
    stale = tmp_path / ".import-deadbeef"
    stale.mkdir(parents=True)
    old = time.time() - 2 * 3600
    os.utime(stale, (old, old))
    fresh = tmp_path / ".import-cafe"
    fresh.mkdir()
    statuses = ModelWeights.status_all(None, tmp_path)
    assert len(statuses) == len(ModelWeights.names())
    assert not stale.exists() and fresh.exists()
    assert (
        ModelWeights.status_all(["sam3", "efficienttam"], tmp_path)[1].weight.name
        == "efficienttam_s"
    )


# ----------------------------------------------------------------------------
# resolve / materialize
# ----------------------------------------------------------------------------


def test_resolve_returns_cached_primary_without_downloading(
    tmp_path, registry, no_network
):
    primary = _seed(tmp_path, registry["sam3"])
    assert ModelWeights.resolve("sam3", cache_dir=tmp_path) == primary
    assert ModelWeights.resolve("efficienttam", cache_dir=tmp_path) if False else True


def test_resolve_offline_miss_names_the_provisioning_command(
    tmp_path, registry, monkeypatch, no_network
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelWeightsMissingError) as info:
        ModelWeights.resolve("sam3", cache_dir=tmp_path)
    msg = str(info.value)
    assert (
        "'sam3'" in msg and "is not in the model cache" in msg and str(tmp_path) in msg
    )
    assert (
        "download-model download sam3" in msg
        and "Settings > Cuvis.AI > Model weights" in msg
    )
    assert "explicit checkpoint path" in msg


def test_resolve_download_false_never_downloads_even_online(
    tmp_path, registry, monkeypatch, no_network
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(ModelWeightsMissingError):
        ModelWeights.resolve("sam3", download=False, cache_dir=tmp_path)


def test_resolve_online_miss_downloads_anonymously_at_the_pin(
    tmp_path, registry, monkeypatch, fake_hub
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "should-not-be-sent")
    fake = fake_hub()
    path = ModelWeights.resolve("sam3", cache_dir=tmp_path)
    assert path == _snapshot_dir(tmp_path, registry["sam3"]) / "sam3.pt"
    assert [c["filename"] for c in fake.calls] == ["sam3.pt", "config.json"]
    assert all(c["token"] is False and c["revision"] == REV_SAM3 for c in fake.calls)
    assert (
        tmp_path / f"models--{HF_ORG}--sam3" / "refs" / "main"
    ).read_text() == REV_SAM3
    assert ModelWeights.status("sam3", tmp_path).present


def test_resolve_accepts_alias_and_refetches_a_damaged_file(
    tmp_path, registry, monkeypatch, fake_hub
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _seed(tmp_path, registry["efficienttam_s"], truncate=True)
    fake = fake_hub()
    path = ModelWeights.resolve("efficienttam", cache_dir=tmp_path)
    assert path.read_bytes() == ETAM_S_BYTES and len(fake.calls) == 1


def test_resolve_cache_hit_needs_no_huggingface_hub(tmp_path, registry, monkeypatch):
    _seed(tmp_path, registry["sam3"])

    def _boom():
        raise AssertionError("huggingface_hub must not be imported on a cache hit")

    monkeypatch.setattr(ModelWeights, "_require_hf_hub", staticmethod(_boom))
    assert ModelWeights.resolve("sam3", cache_dir=tmp_path).exists()


def test_materialize_places_and_reuses_the_destination(tmp_path, registry, no_network):
    _seed(tmp_path, registry["efficienttam_ti"])
    seeded = tmp_path / "seeded"
    seeded.mkdir()
    (seeded / "efficienttam_ti.pt").write_bytes(b"seeded")
    # An existing destination is returned untouched: no cache lookup, no download.
    placed = ModelWeights.materialize("efficienttam_ti", seeded, cache_dir=tmp_path)
    assert placed.read_bytes() == b"seeded"
    dest = tmp_path / "dest"
    out = ModelWeights.materialize("efficienttam_ti", dest, cache_dir=tmp_path)
    assert out == dest / "efficienttam_ti.pt" and out.read_bytes() == ETAM_TI_BYTES
    renamed = ModelWeights.materialize(
        "efficienttam_ti", dest, filename="tiny.pt", cache_dir=tmp_path
    )
    assert renamed.name == "tiny.pt" and renamed.read_bytes() == ETAM_TI_BYTES


def test_materialize_copies_when_hardlink_is_impossible(
    tmp_path, registry, monkeypatch, no_network
):
    _seed(tmp_path, registry["efficienttam_ti"])

    def _no_link(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", _no_link)
    out = ModelWeights.materialize(
        "efficienttam_ti", tmp_path / "d", cache_dir=tmp_path
    )
    assert out.read_bytes() == ETAM_TI_BYTES and not list(
        (tmp_path / "d").glob("*.part")
    )


def test_materialize_links_the_blob_behind_a_symlinked_snapshot_entry(
    tmp_path, registry, no_network
):
    """POSIX caches store snapshot entries as symlinks into blobs/; the link must follow them."""
    entry = registry["efficienttam_ti"]
    primary = _seed(tmp_path, entry)
    blob = (
        tmp_path / ModelWeights.cache_dir_token(entry.repo_id) / "blobs" / entry.sha256
    )
    blob.parent.mkdir(parents=True, exist_ok=True)
    primary.replace(blob)
    try:
        os.symlink(os.path.relpath(blob, primary.parent), primary)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this machine")
    out = ModelWeights.materialize(
        "efficienttam_ti", tmp_path / "d", cache_dir=tmp_path
    )
    assert not out.is_symlink()
    assert out.read_bytes() == ETAM_TI_BYTES


def test_materialize_offline_miss_raises(tmp_path, registry, monkeypatch, no_network):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelWeightsMissingError):
        ModelWeights.materialize("sam3", tmp_path / "d", cache_dir=tmp_path)


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------


def test_download_skips_network_and_hashing_when_present(
    tmp_path, registry, monkeypatch, no_network
):
    _seed(tmp_path, registry["sam3"])
    monkeypatch.setattr(
        mw, "sha256_of", lambda p: pytest.fail("must not hash on a cache hit")
    )
    assert (
        ModelWeights.download("sam3", tmp_path)
        == _snapshot_dir(tmp_path, registry["sam3"]) / "sam3.pt"
    )


def test_download_force_refetches_and_writes_refs_main(tmp_path, registry, fake_hub):
    _seed(tmp_path, registry["sam3"], write_ref=False)
    fake = fake_hub()
    ModelWeights.download("sam3", tmp_path, force=True)
    assert len(fake.calls) == 2 and all(c["force_download"] for c in fake.calls)
    assert (
        tmp_path / f"models--{HF_ORG}--sam3" / "refs" / "main"
    ).read_text() == REV_SAM3


def test_download_sha_mismatch_raises(tmp_path, registry, fake_hub):
    fake_hub(content={"sam3.pt": b"tampered", "config.json": CONFIG_BYTES})
    with pytest.raises(ModelDownloadError, match="sha256 mismatch.*--force"):
        ModelWeights.download("sam3", tmp_path)


def test_download_many_preserves_order_and_fails_fast_on_unknown_names(
    tmp_path, registry, fake_hub
):
    fake = fake_hub()
    with pytest.raises(ModelDownloadError, match="Unknown model 'nope'"):
        ModelWeights.download_many(["sam3", "nope"], tmp_path)
    assert fake.calls == []  # nothing downloaded before the names were validated
    paths = ModelWeights.download_many(["efficienttam_ti", "sam3"], tmp_path)
    assert [p.name for p in paths] == ["efficienttam_ti.pt", "sam3.pt"]
    assert [c["filename"] for c in fake.calls] == [
        "efficienttam_ti.pt",
        "sam3.pt",
        "config.json",
    ]


def test_download_model_explicit_repo_forwards_the_token(
    tmp_path, registry, fake_hub, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    fake = fake_hub(content={"custom.bin": b"custom"})
    out = tmp_path / "out" / "custom.bin"
    path = ModelWeights.download_model(
        None,
        repo_id="someone/private",
        filename="custom.bin",
        revision="abc",
        cache_dir=tmp_path,
        out=out,
    )
    assert path == out and out.read_bytes() == b"custom"
    assert fake.calls[0]["token"] == "env-token" and fake.calls[0]["revision"] == "abc"
    ModelWeights.download_model(
        None,
        repo_id="someone/private",
        filename="custom.bin",
        token="explicit",
        cache_dir=tmp_path,
    )
    assert fake.calls[-1]["token"] == "explicit"
    with pytest.raises(ModelDownloadError, match="either a registry NAME or"):
        ModelWeights.download_model("sam3", repo_id="x/y", cache_dir=tmp_path)
    with pytest.raises(ModelDownloadError, match="Need a registry name or both"):
        ModelWeights.download_model(None, repo_id="x/y", cache_dir=tmp_path)


def test_download_progress_verifying_event_and_library_silence(
    tmp_path, registry, fake_hub, capsys
):
    fake_hub()
    rec = Recorder()
    ModelWeights.download("sam3", tmp_path, progress=rec)
    assert [e["event"] for e in rec.events] == [
        "verifying",
        "verifying",
    ]  # primary + aux
    assert capsys.readouterr().out == ""  # library methods never print to stdout


# ----------------------------------------------------------------------------
# error mapping
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, pattern",
    [
        (
            lambda u: _hub_error(u.RepositoryNotFoundError),
            "was not found or is not public .*registry mis-pin",
        ),
        (lambda u: _hub_error(u.GatedRepoError), "gated .*registry mis-pin"),
        (
            lambda u: _hub_error(u.RevisionNotFoundError),
            "has no file 'sam3.pt' at revision",
        ),
        (
            lambda u: _hub_error(u.EntryNotFoundError),
            "has no file 'sam3.pt' at revision",
        ),
        (lambda u: _hub_error(u.LocalEntryNotFoundError), "network is unavailable"),
        (
            lambda u: _hub_error(u.HfHubHTTPError, 429),
            "rate-limiting or unavailable \\(HTTP 429\\)",
        ),
        (
            lambda u: _hub_error(u.HfHubHTTPError, 503),
            "rate-limiting or unavailable \\(HTTP 503\\)",
        ),
        (lambda u: _hub_error(u.HfHubHTTPError, 418), "failed \\(HTTP 418\\)"),
        (lambda u: RuntimeError("weird"), "failed: weird"),
    ],
)
def test_hub_errors_map_to_user_sentences(
    tmp_path, registry, fake_hub, factory, pattern
):
    utils = pytest.importorskip("huggingface_hub.utils")
    fake_hub(fail=factory(utils))
    with pytest.raises(ModelDownloadError, match=pattern):
        ModelWeights.download("sam3", tmp_path)


def test_http_401_on_a_custom_repo_mentions_the_token(tmp_path, registry, fake_hub):
    utils = pytest.importorskip("huggingface_hub.utils")
    fake_hub(fail=_hub_error(utils.HfHubHTTPError, 401))
    with pytest.raises(ModelDownloadError, match="401.*--token"):
        ModelWeights.download_model(
            None, repo_id="x/y", filename="f.bin", cache_dir=tmp_path
        )


# ----------------------------------------------------------------------------
# export / import
# ----------------------------------------------------------------------------


def _export_two(tmp_path, registry):
    cache = tmp_path / "cache"
    _seed(cache, registry["sam3"])
    _seed(cache, registry["efficienttam_s"])
    export_dir = tmp_path / "export"
    results = ModelWeights.export_to(
        export_dir, ["sam3", "efficienttam_s", "efficienttam_ti"], cache
    )
    return cache, export_dir, results


def test_export_copies_pinned_snapshot_layout_and_manifest(tmp_path, registry):
    cache, export_dir, results = _export_two(tmp_path, registry)
    by_name = {r["name"]: r for r in results}
    assert by_name["sam3"]["exported"] and by_name["efficienttam_s"]["exported"]
    assert by_name["efficienttam_ti"] == {
        **by_name["efficienttam_ti"],
        "exported": False,
        "path": None,
    }
    assert (
        export_dir / f"models--{HF_ORG}--sam3" / "snapshots" / REV_SAM3 / "config.json"
    ).read_bytes() == CONFIG_BYTES
    assert (
        export_dir / f"models--{HF_ORG}--sam3" / "refs" / "main"
    ).read_text() == REV_SAM3
    manifest = json.loads((export_dir / mw.EXPORT_MANIFEST_NAME).read_text())
    assert manifest["schema_version"] == 1 and {
        m["name"] for m in manifest["models"]
    } == {"sam3", "efficienttam_s"}
    sam3_files = {
        f["path"]: f["sha256"]
        for f in next(m for m in manifest["models"] if m["name"] == "sam3")["files"]
    }
    assert sam3_files == {
        "sam3.pt": _sha(SAM3_BYTES),
        "config.json": _sha(CONFIG_BYTES),
    }
    # the export directory is itself a valid cache
    assert ModelWeights.status("sam3", export_dir).present


def test_import_copies_verifies_and_is_idempotent(tmp_path, registry):
    _, export_dir, _ = _export_two(tmp_path, registry)
    target = tmp_path / "target"
    results = ModelWeights.import_from(export_dir, None, target)
    assert {(r["name"], r["imported"]) for r in results} == {
        ("sam3", True),
        ("efficienttam_s", True),
    }
    assert (
        ModelWeights.status("sam3", target).present
        and ModelWeights.status("efficienttam_s", target).present
    )
    assert not list(target.glob(".import-*"))
    again = ModelWeights.import_from(export_dir, ["sam3"], target)
    assert again == [{**again[0], "imported": False, "reason": "present"}]


def test_import_is_all_or_nothing_on_a_tampered_file(tmp_path, registry):
    _, export_dir, _ = _export_two(tmp_path, registry)
    (
        export_dir
        / f"models--{HF_ORG}--efficient-track-anything"
        / "snapshots"
        / REV_ETAM
        / "efficienttam_s.pt"
    ).write_bytes(b"x" * len(ETAM_S_BYTES))
    target = tmp_path / "target"
    with pytest.raises(
        ModelDownloadError, match="checksum mismatch.*Nothing was changed"
    ):
        ModelWeights.import_from(export_dir, None, target)
    assert not list(target.glob("models--*")) and not list(target.glob(".import-*"))


def test_import_rejects_a_folder_without_manifest_and_skips_unknown_rows(
    tmp_path, registry
):
    with pytest.raises(ModelDownloadError, match="not an exported weights folder"):
        ModelWeights.import_from(tmp_path, None, tmp_path / "t")
    _, export_dir, _ = _export_two(tmp_path, registry)
    manifest_path = export_dir / mw.EXPORT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["models"].append({"name": "from_the_future", "repo_id": "x/y"})
    manifest_path.write_text(json.dumps(manifest))
    results = ModelWeights.import_from(export_dir, None, tmp_path / "t2")
    skipped = [r for r in results if r["name"] == "from_the_future"]
    assert skipped == [
        {
            "name": "from_the_future",
            "repo_id": "x/y",
            "imported": False,
            "reason": "unknown model",
        }
    ]


# ----------------------------------------------------------------------------
# remove
# ----------------------------------------------------------------------------


def test_remove_deletes_only_that_rows_files_in_a_shared_repo(tmp_path, registry):
    _seed(tmp_path, registry["efficienttam_s"])
    _seed(tmp_path, registry["efficienttam_ti"])
    freed = ModelWeights.remove("efficienttam_ti", tmp_path)
    assert freed == len(ETAM_TI_BYTES)
    assert ModelWeights.status("efficienttam_ti", tmp_path).state == "absent"
    assert ModelWeights.status("efficienttam_s", tmp_path).present
    assert ModelWeights.remove("efficienttam_s", tmp_path) == len(ETAM_S_BYTES)
    assert not (
        tmp_path / f"models--{HF_ORG}--efficient-track-anything"
    ).exists()  # pruned once empty
    assert ModelWeights.remove("efficienttam_s", tmp_path) == 0  # already gone


def test_remove_dir_handles_orphans_and_refuses_anything_else(tmp_path, registry):
    orphan = tmp_path / "models--facebook--sam3" / "snapshots" / "abc"
    orphan.mkdir(parents=True)
    (orphan / "sam3.pt").write_bytes(b"old")
    assert ModelWeights.remove_dir("models--facebook--sam3", tmp_path) == 3
    assert not (tmp_path / "models--facebook--sam3").exists()
    for bad in ("../x", "datasets", "models--a/b", "models--..--x"):
        with pytest.raises(ModelDownloadError, match="refusing"):
            ModelWeights.remove_dir(bad, tmp_path)
    with pytest.raises(ModelDownloadError, match="not a directory"):
        ModelWeights.remove_dir("models--nobody--here", tmp_path)


# ----------------------------------------------------------------------------
# the cache lock
# ----------------------------------------------------------------------------


def test_writing_ops_wait_for_the_root_lock(tmp_path, registry, fake_hub):
    fake_hub()
    tmp_path.mkdir(exist_ok=True)
    held = FileLock(str(tmp_path / prov.LOCK_FILENAME))
    held.acquire(timeout=0)
    rec = Recorder()
    done = threading.Event()

    def worker():
        ModelWeights.download("efficienttam_ti", tmp_path, progress=rec)
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert not done.wait(0.4)
    assert {"event": "waiting"} in rec.events
    held.release()
    assert done.wait(5)
    t.join()
    assert ModelWeights.status("efficienttam_ti", tmp_path).present


def test_lock_timeout_surfaces_as_a_download_error(
    tmp_path, registry, monkeypatch, fake_hub
):
    fake_hub()
    monkeypatch.setattr(mw, "root_lock", functools.partial(prov.root_lock, timeout=0.1))
    held = FileLock(str(tmp_path / prov.LOCK_FILENAME))
    held.acquire(timeout=0)
    try:
        with pytest.raises(
            ModelDownloadError, match="another operation is using the cache"
        ):
            ModelWeights.download("efficienttam_ti", tmp_path)
    finally:
        held.release()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, registry):
    plugins = tmp_path / "no-plugins"
    plugins.mkdir()
    cache = tmp_path / "cache"
    return SimpleNamespace(
        runner=CliRunner(),
        cli=build_cli(),
        common=["--plugins-dir", str(plugins), "--cache-dir", str(cache)],
        plugins=plugins,
        cache=cache,
    )


def _validate(name: str, payload) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(prov.load_schema(name)).validate(payload)


def test_cli_list_json_and_table(cli_env):
    result = cli_env.runner.invoke(
        cli_env.cli, ["list", "--json", "--plugins-dir", str(cli_env.plugins)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    _validate("model_list", payload)
    assert [m["name"] for m in payload["models"]][:1] == ["dinomaly_bedding_all6"]
    table = cli_env.runner.invoke(
        cli_env.cli, ["list", "--plugins-dir", str(cli_env.plugins)]
    )
    assert (
        table.exit_code == 0
        and "efficienttam_s" in table.stdout
        and "rtsam2" in table.stdout
    )


def test_cli_plugins_dir_loads_manifests(cli_env, tmp_path):
    ModelWeights.reset()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_manifest(manifests, "sam3", [sam3_entry()])
    result = cli_env.runner.invoke(
        cli_env.cli, ["list", "--json", "--plugins-dir", str(manifests)]
    )
    assert result.exit_code == 0, result.output
    rows = {m["name"]: m for m in json.loads(result.stdout)["models"]}
    assert rows["sam3"]["source"] == "manifest" and rows["sam3"]["plugin"] == "sam3"


def test_cli_status_json_shape_includes_cache_dir(cli_env, registry):
    _seed(cli_env.cache, registry["sam3"])
    result = cli_env.runner.invoke(
        cli_env.cli, ["status", "sam3", "efficienttam_ti", "--json", *cli_env.common]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    _validate("status", payload)
    assert payload["cache_dir"] == str(cli_env.cache)
    states = {m["name"]: m["state"] for m in payload["models"]}
    assert states == {"sam3": "present", "efficienttam_ti": "absent"}
    table = cli_env.runner.invoke(cli_env.cli, ["status", *cli_env.common])
    assert (
        table.exit_code == 0 and "present" in table.stdout and "absent" in table.stdout
    )


def test_cli_download_prints_paths_in_order_and_json_marks_downloaded(
    cli_env, fake_hub
):
    fake_hub()
    result = cli_env.runner.invoke(
        cli_env.cli, ["download", "efficienttam_ti", "sam3", *cli_env.common]
    )
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert [Path(line).name for line in lines] == ["efficienttam_ti.pt", "sam3.pt"]
    assert all(Path(line).is_absolute() for line in lines)
    as_json = cli_env.runner.invoke(
        cli_env.cli, ["download", "efficienttam_s", "--json", *cli_env.common]
    )
    assert as_json.exit_code == 0, as_json.output
    payload = json.loads(as_json.stdout)
    _validate("status", payload)
    assert (
        payload["models"][0]["downloaded"] is True
        and payload["models"][0]["state"] == "present"
    )


def test_cli_progress_json_lines_validate_and_end_with_done(cli_env, fake_hub):
    fake_hub()
    result = cli_env.runner.invoke(
        cli_env.cli, ["download", "sam3", "--progress-json", *cli_env.common]
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines()]
    for event in events:
        _validate("progress_event", event)
    assert [e["event"] for e in events] == ["verifying", "verifying", "done"]
    assert events[-1]["name"] == "sam3" and events[-1]["path"].endswith("sam3.pt")


def test_cli_download_failure_emits_error_event_and_exit_1(cli_env, fake_hub):
    utils = pytest.importorskip("huggingface_hub.utils")
    fake_hub(fail=_hub_error(utils.HfHubHTTPError, 429))
    result = cli_env.runner.invoke(
        cli_env.cli, ["download", "sam3", "--progress-json", *cli_env.common]
    )
    assert result.exit_code == 1
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["event"] == "error" and "HTTP 429" in events[-1]["message"]
    assert result.stderr.rstrip().splitlines()[-1].startswith("error: ")


def test_cli_usage_errors(cli_env):
    both = cli_env.runner.invoke(
        cli_env.cli, ["download", "sam3", "--json", "--progress-json", *cli_env.common]
    )
    assert both.exit_code == 2 and "mutually exclusive" in both.stderr
    unknown = cli_env.runner.invoke(cli_env.cli, ["download", "nope", *cli_env.common])
    assert unknown.exit_code == 1 and unknown.stderr.startswith(
        "error: Unknown model 'nope'"
    )
    none = cli_env.runner.invoke(cli_env.cli, ["download", *cli_env.common])
    assert none.exit_code == 2
    mixed = cli_env.runner.invoke(
        cli_env.cli,
        ["download", "sam3", "--repo-id", "x/y", "--filename", "f", *cli_env.common],
    )
    assert mixed.exit_code == 2


def test_cli_custom_repo_download_prints_the_path(cli_env, fake_hub):
    fake = fake_hub(content={"f.bin": b"f"})
    result = cli_env.runner.invoke(
        cli_env.cli,
        [
            "download",
            "--repo-id",
            "x/y",
            "--filename",
            "f.bin",
            "--token",
            "t",
            "--json",
            *cli_env.common,
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.strip().endswith("f.bin") and fake.calls[0]["token"] == "t"


def test_cli_export_import_roundtrip_json_shapes(cli_env, registry, tmp_path):
    _seed(cli_env.cache, registry["sam3"])
    export_dir = tmp_path / "exp"
    exported = cli_env.runner.invoke(
        cli_env.cli,
        ["export", "--to", str(export_dir), "sam3", "--json", *cli_env.common],
    )
    assert exported.exit_code == 0, exported.output
    payload = json.loads(exported.stdout)
    _validate("export", payload)
    assert payload["models"][0]["exported"] is True
    absent = cli_env.runner.invoke(
        cli_env.cli,
        ["export", "--to", str(export_dir), "efficienttam_ti", *cli_env.common],
    )
    assert absent.exit_code == 1 and "absent, skipped" in absent.stdout
    target = tmp_path / "second-cache"
    imported = cli_env.runner.invoke(
        cli_env.cli,
        [
            "import",
            str(export_dir),
            "--json",
            "--plugins-dir",
            str(cli_env.plugins),
            "--cache-dir",
            str(target),
        ],
    )
    assert imported.exit_code == 0, imported.output
    payload = json.loads(imported.stdout)
    _validate("status", payload)
    assert (
        payload["models"][0]["imported"] is True
        and payload["models"][0]["state"] == "present"
    )
    plain = cli_env.runner.invoke(
        cli_env.cli,
        [
            "import",
            str(export_dir),
            "--plugins-dir",
            str(cli_env.plugins),
            "--cache-dir",
            str(tmp_path / "third"),
        ],
    )
    assert plain.exit_code == 0 and plain.stdout.strip().endswith("sam3.pt")


def test_cli_remove_name_and_dir_forms_and_refusals(cli_env, registry):
    _seed(cli_env.cache, registry["sam3"])
    (cli_env.cache / "models--facebook--sam3" / "snapshots").mkdir(parents=True)
    result = cli_env.runner.invoke(
        cli_env.cli, ["remove", "sam3", "--json", *cli_env.common]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    _validate("remove", payload)
    assert payload["removed"] == [
        {"name": "sam3", "freed_bytes": len(SAM3_BYTES) + len(CONFIG_BYTES)}
    ]
    orphan = cli_env.runner.invoke(
        cli_env.cli, ["remove", "--dir", "models--facebook--sam3", *cli_env.common]
    )
    assert orphan.exit_code == 0 and "freed" in orphan.stdout
    refused = cli_env.runner.invoke(
        cli_env.cli, ["remove", "--dir", "..", *cli_env.common]
    )
    assert refused.exit_code == 1 and "refusing" in refused.stderr
    neither = cli_env.runner.invoke(cli_env.cli, ["remove", *cli_env.common])
    assert neither.exit_code == 2


def test_cli_schema_prints_shipped_files(cli_env):
    for name in ("model_list", "status", "progress_event", "export", "remove"):
        result = cli_env.runner.invoke(cli_env.cli, ["schema", name])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == prov.load_schema(name)
    assert cli_env.runner.invoke(cli_env.cli, ["schema", "nope"]).exit_code == 2


def test_library_methods_write_nothing_to_stdout(tmp_path, registry, fake_hub, capsys):
    fake_hub()
    _seed(tmp_path, registry["efficienttam_ti"])
    ModelWeights.status_all(None, tmp_path)
    ModelWeights.resolve("efficienttam_ti", cache_dir=tmp_path)
    ModelWeights.download("sam3", tmp_path)
    ModelWeights.export_to(tmp_path / "e", ["sam3"], tmp_path)
    ModelWeights.import_from(tmp_path / "e", None, tmp_path / "i")
    ModelWeights.remove("sam3", tmp_path)
    assert capsys.readouterr().out == ""
