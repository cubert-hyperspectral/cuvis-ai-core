"""Property-based tests for cuvis_ai_core.utils.provision helpers.

Verifies that _spec_for produces valid PEP-508 specs and that
format_install_command invariants hold. No CUDA, no I/O.
"""

pytest = __import__("pytest")
pytest.importorskip("hypothesis")

# ruff: noqa: E402 — imports must follow importorskip to avoid ImportError on collection
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from packaging.requirements import Requirement

from cuvis_ai_core.orchestrator.cache_key import ResolvedGitPlugin, ResolvedLocalPlugin
from cuvis_ai_core.utils import provision as prov

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git"
_name_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-_"
_hex_alphabet = "0123456789abcdef"

_package_name = st.text(min_size=1, max_size=30, alphabet=_name_alphabet).filter(
    lambda s: s[0].isalpha() and s[-1].isalnum()
)

_extras = st.frozensets(
    st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
    max_size=3,
).map(tuple)

_sha40 = st.text(min_size=40, max_size=40, alphabet=_hex_alphabet)
_tag = st.text(
    min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789._-"
)


@st.composite
def _git_plugin(draw):
    return ResolvedGitPlugin(
        name="sam3",
        repo=_REPO,
        sha=draw(_sha40),
        tag=draw(_tag),
        package_name=draw(_package_name),
        extras=draw(_extras),
    )


@st.composite
def _local_plugin(draw, tmp_path=Path("/tmp/test_pkg")):
    return ResolvedLocalPlugin(
        name="dl",
        path=tmp_path,
        package_name=draw(_package_name),
        pyproject_sha256=draw(_sha40),
        git_head=None,
        dirty=False,
        extras=(),
    )


# ---------------------------------------------------------------------------
# _spec_for: PEP-508 round-trip
# ---------------------------------------------------------------------------


@given(plugin=_git_plugin(), pin=st.sampled_from(["tag", "sha"]))
@settings(max_examples=100)
def test_spec_for_git_produces_pep508_parseable_spec(
    plugin: ResolvedGitPlugin, pin: str
) -> None:
    spec = prov._spec_for(plugin, pin)
    # Requirement(...) raises InvalidRequirement on malformed specs
    req = Requirement(spec)
    assert req.name == plugin.package_name
    if plugin.extras:
        assert set(req.extras) == set(plugin.extras)


@given(plugin=_git_plugin())
@settings(max_examples=80)
def test_spec_for_git_tag_uses_tag(plugin: ResolvedGitPlugin) -> None:
    spec = prov._spec_for(plugin, "tag")
    assert plugin.tag in spec


@given(plugin=_git_plugin())
@settings(max_examples=80)
def test_spec_for_git_sha_uses_sha(plugin: ResolvedGitPlugin) -> None:
    spec = prov._spec_for(plugin, "sha")
    assert plugin.sha in spec


# ---------------------------------------------------------------------------
# format_install_command invariants
# ---------------------------------------------------------------------------


@given(specs=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
def test_format_install_command_non_empty_contains_all_specs(
    specs: list[str],
) -> None:
    cmd = prov.format_install_command(specs)
    for spec in specs:
        assert spec in cmd


def test_format_install_command_empty_signals_nothing_to_install() -> None:
    cmd = prov.format_install_command([])
    assert "nothing to install" in cmd.lower() or not cmd.strip().startswith("uv")


@given(specs=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
def test_format_install_command_magic_uses_pip_magic(specs: list[str]) -> None:
    cmd = prov.format_install_command(specs, magic=True)
    assert cmd.startswith("%pip install")


@given(specs=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
def test_format_install_command_non_magic_uses_uv(specs: list[str]) -> None:
    cmd = prov.format_install_command(specs, magic=False)
    assert "uv pip install" in cmd
