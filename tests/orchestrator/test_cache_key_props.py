"""Property-based tests for cuvis_ai_core.orchestrator.cache_key.

Guards mathematical invariants: determinism, input-sensitivity, round-trip.
No CUDA, no I/O, no model downloads.
"""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

# ruff: noqa: E402 — imports must follow importorskip to avoid ImportError on collection
import json

from hypothesis import given, settings
from hypothesis import strategies as st

from cuvis_ai_core.orchestrator.cache_key import (
    COMPOSER_SCHEMA_VERSION,
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
    compute_cache_key,
    spec_hash_of,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_core_kinds = st.sampled_from(["pypi", "git", "local"])


@st.composite
def _core_source(draw):
    kind = draw(_core_kinds)
    identity = draw(
        st.text(
            min_size=1,
            max_size=40,
            alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        )
    )
    return CoreSource(kind=kind, identity=identity)


@st.composite
def _git_plugin(draw):
    name = draw(
        st.text(
            min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-"
        )
    )
    repo = draw(
        st.text(
            min_size=5,
            max_size=60,
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789/:._-",
        )
    )
    sha = draw(st.text(min_size=40, max_size=40, alphabet="0123456789abcdef"))
    tag = draw(
        st.text(
            min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789._-"
        )
    )
    return ResolvedGitPlugin(name=name, repo=repo, sha=sha, tag=tag)


@st.composite
def _local_plugin(draw, dirty=False):
    from pathlib import Path

    name = draw(
        st.text(
            min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-"
        )
    )
    path = Path("/tmp") / draw(
        st.text(
            min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_"
        )
    )
    pkg = draw(
        st.text(
            min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-"
        )
    )
    sha256 = draw(st.text(min_size=1, max_size=64, alphabet="0123456789abcdef"))
    return ResolvedLocalPlugin(
        name=name,
        path=path,
        package_name=pkg,
        pyproject_sha256=sha256,
        git_head=None,
        dirty=dirty,
    )


_spec_hash_text = st.text(min_size=0, max_size=200)


# ---------------------------------------------------------------------------
# spec_hash_of properties
# ---------------------------------------------------------------------------


@given(content=_spec_hash_text)
def test_spec_hash_of_is_deterministic(content: str) -> None:
    assert spec_hash_of(content) == spec_hash_of(content)


@given(content=_spec_hash_text)
def test_spec_hash_of_is_hex_64_chars(content: str) -> None:
    h = spec_hash_of(content)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


@given(a=_spec_hash_text, b=_spec_hash_text)
def test_spec_hash_of_is_input_sensitive(a: str, b: str) -> None:
    if a != b:
        assert spec_hash_of(a) != spec_hash_of(b)


# ---------------------------------------------------------------------------
# compute_cache_key / CacheKey properties
# ---------------------------------------------------------------------------


@given(core=_core_source(), spec=_spec_hash_text)
@settings(max_examples=50)
def test_compute_cache_key_is_deterministic(core: CoreSource, spec: str) -> None:
    key_a = compute_cache_key(
        core_source=core,
        plugins=(),
        spec_hash=spec,
        python_version="3.11.0",
        platform_tag="linux-x86_64",
    )
    key_b = compute_cache_key(
        core_source=core,
        plugins=(),
        spec_hash=spec,
        python_version="3.11.0",
        platform_tag="linux-x86_64",
    )
    assert key_a.directory_name() == key_b.directory_name()


@given(core=_core_source(), spec_a=_spec_hash_text, spec_b=_spec_hash_text)
@settings(max_examples=50)
def test_compute_cache_key_is_input_sensitive_on_spec(
    core: CoreSource, spec_a: str, spec_b: str
) -> None:
    if spec_a == spec_b:
        return
    key_a = compute_cache_key(
        core_source=core,
        plugins=(),
        spec_hash=spec_a,
        python_version="3.11.0",
        platform_tag="linux-x86_64",
    )
    key_b = compute_cache_key(
        core_source=core,
        plugins=(),
        spec_hash=spec_b,
        python_version="3.11.0",
        platform_tag="linux-x86_64",
    )
    assert key_a.directory_name() != key_b.directory_name()


@given(core=_core_source(), spec=_spec_hash_text)
@settings(max_examples=50)
def test_cache_key_directory_name_is_idempotent(core: CoreSource, spec: str) -> None:
    key = compute_cache_key(
        core_source=core,
        plugins=(),
        spec_hash=spec,
        python_version="3.11.0",
        platform_tag="linux-x86_64",
    )
    assert key.directory_name() == key.directory_name()


@given(core=_core_source(), spec=_spec_hash_text)
@settings(max_examples=50)
def test_cache_key_serialise_round_trips_via_json(core: CoreSource, spec: str) -> None:
    key = compute_cache_key(
        core_source=core,
        plugins=(),
        spec_hash=spec,
        python_version="3.11.0",
        platform_tag="linux-x86_64",
    )
    serialised = key.serialise()
    # Must be JSON-serializable and round-trip through json.dumps/loads
    raw = json.dumps(serialised)
    recovered = json.loads(raw)
    assert recovered["spec_hash"] == spec
    assert recovered["schema_version"] == COMPOSER_SCHEMA_VERSION
