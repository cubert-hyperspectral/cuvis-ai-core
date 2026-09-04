"""Build, upload and audit the ``cubert-gmbh`` Hugging Face mirrors of third-party weights.

Cuvis.AI plugins load pretrained weights (SAM3, EfficientTAM, DINOv2, the CLIP
backbone and the AdaCLIP heads) that upstream publishes in five different ways:
a gated Hugging Face repo, an ungated third-party Hugging Face repo, a public
file server, a CDN URL and Google Drive links. ``ModelWeights`` in
``cuvis_ai_core.data.model_weights`` points only at byte-identical mirrors of
those files under the ``cubert-gmbh`` organisation, so provisioning is
commit-pinned, sha256-verified and needs no per-user account or token.

This tool is the reproducible way to build those mirrors. It is a maintainer
tool, not part of the installed package (``tools/`` is not packaged).

Commands (run from the repo root with the project environment)::

    uv run --no-sync python tools/mirror_weights.py plan all
    uv run --no-sync python tools/mirror_weights.py upload efficient-track-anything
    uv run --no-sync python tools/mirror_weights.py check all

``plan`` resolves every file (reusing the local Hugging Face cache and other
known local copies, downloading otherwise), verifies the sha256 pins, renders the
model card into the work dir and prints the registry entries that ``upload``
would produce, without touching the Hub. ``upload`` does the same and then
creates one commit per mirror repo (the repo is created private and made public
and ungated once the commit has landed) and prints the registry entries with the
mirror commit as ``revision``. ``check`` audits an existing mirror: sha256 pins
against the Hub metadata, LICENSE bytes against a fresh upstream copy, repo
visibility and gating, and the core registry (``ModelWeights._models`` of the
checkout you run from) against the Hub.

The AdaCLIP heads live on Google Drive; ``gdown`` is imported lazily, so run the
tool with ``uv run --no-sync --with gdown ...`` when the adaclip mirror needs a
fresh download.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template

import click

HF_ORG = "cubert-gmbh"
_SHA_READ_CHUNK = 1 << 20
_HOME = Path.home()
_DEFAULT_WORK_DIR = _HOME / ".cache" / "cuvis_ai" / "mirror_weights"


# ----------------------------------------------------------------------------
# Source table
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """Where a mirrored file comes from upstream."""

    kind: str  # "hf" | "url" | "gdrive"
    repo_id: str | None = None
    filename: str | None = None
    revision: str | None = None
    url: str | None = None
    file_id: str | None = None
    needs_token: bool = False

    def label(self) -> str:
        if self.kind == "hf":
            return f"`{self.repo_id}` / `{self.filename}`"
        if self.kind == "url":
            return f"<{self.url}>"
        return f"Google Drive file `{self.file_id}`"

    def revision_label(self, pinned: bool) -> str:
        if self.kind == "hf":
            return f"`{self.revision}`"
        if self.kind == "gdrive":
            return f"`{self.file_id}`"
        return "n/a (URL; sha256 pinned)" if pinned else "n/a (URL; sha256 recorded)"


@dataclass(frozen=True)
class MirrorFile:
    """One file in a mirror repo."""

    name: str
    source: Source
    sha256: str | None = None  # upstream pin when known; None = record on first run
    local_candidates: tuple[Path, ...] = ()
    registry_name: str | None = None  # core registry key when this is a checkpoint
    description: str = ""
    aux_of: str | None = None  # registry_name this file is a companion of


@dataclass(frozen=True)
class Mirror:
    """One repo under the ``cubert-gmbh`` organisation."""

    repo: str
    title: str
    plugin: str
    license_id: str  # Hugging Face front-matter id
    license_name: str  # human label recorded in the registry
    upstream_md: str
    files: tuple[MirrorFile, ...]
    license_text: str
    base_model: str | None = None
    extra_tags: tuple[str, ...] = ()
    license_front_matter_extra: tuple[str, ...] = ()

    @property
    def repo_id(self) -> str:
        return f"{HF_ORG}/{self.repo}"


_SAM3_REV = "3c879f39826c281e95690f02c7821c4de09afae7"
_ETAM_REV = "9bdd8ab585b19ef95f9c9ed847ac9478301890b4"
_ETAM_REPO = "yunyangx/efficient-track-anything"
_DINOMALY_REPO_DIR = Path("D:/code-repos/cuvis-ai-dinomaly/cuvis-ai-dinomaly")

_SAM_LICENSE_TEXT = """\
These files are Meta's "SAM Materials" and are distributed under the **SAM License**
(Last Updated: November 19, 2025), reproduced verbatim in `LICENSE`. By downloading
or using them you agree to be bound by that Agreement. Section 1.b.i allows
redistribution only under the same terms and with a copy of the Agreement, which is
what this repository does.

Obligations that travel with the files:

- You may not use, or permit others to use, the SAM Materials for any activities
  subject to the International Traffic in Arms Regulations (ITAR) or end uses
  prohibited by Trade Controls, including those related to military or warfare
  purposes, nuclear industries or applications, espionage, or the development or use
  of guns or illegal weapons (section 1.b.v). You must not be the target of Trade
  Controls, and your use must comply with Trade Controls and applicable laws,
  including privacy and data-protection laws (sections 1.b.iii and 1.b.v).
- Publications of research performed with the SAM Materials must acknowledge their
  use (section 1.b.ii). Reverse engineering the SAM Materials is not permitted
  (section 1.b.iv).
- The SAM Materials are provided "as is" without warranties (section 3); Meta may
  modify the Agreement (section 8), so check the upstream text for the current
  version.
- If you allege in litigation that the SAM Materials or their outputs infringe
  intellectual property or other rights you own or can license, your licence
  terminates on the day the claim is filed, and you indemnify Meta against
  third-party claims arising from your use or distribution of the SAM Materials
  (section 5.b). On termination you must delete and cease use of the SAM Materials
  (section 6).

The upstream repository `facebook/sam3` is gated; this mirror is ungated. Cubert GmbH
redistributes these files as a Licensee under section 1.b.i and is itself bound by
the Agreement, including the indemnity in section 5.b, for its own distribution. You
are bound directly and separately by the same Agreement, including sections 5.b and
6 above. Cubert GmbH is not affiliated with or endorsed by Meta.
"""

_APACHE_TEXT = Template(
    """\
Apache License 2.0. The `LICENSE` file is the upstream licence text taken verbatim
from $origin at the mirror date. Copyright remains with the upstream authors; Cubert
GmbH claims no rights in these files.
"""
)

_CLIP_TEXT = """\
The `openai/CLIP` repository is published under the MIT License, Copyright (c) 2021
OpenAI (`LICENSE`, verbatim). OpenAI publishes this checkpoint through that
repository's `clip.load("ViT-L/14@336px")` loader (served from
`openaipublic.azureedge.net`; the sha256 is the hash OpenAI encodes in the download
URL) without a separate licence statement for the weights, and OpenAI's own Hugging
Face copies (for example `openai/clip-vit-large-patch14-336`) carry no licence tag.
The file is redistributed here unchanged, as released; the `license: unknown` tag
reflects that no explicit licence statement covers the weights. If you are a rights
holder and object to this redistribution, open a discussion on this repository and
the file will be taken down.
"""

_ADACLIP_TEXT = """\
The AdaCLIP code and these released checkpoints come from the AdaCLIP project
(Yunkang Cao et al., "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for
Zero-Shot Anomaly Detection", ECCV 2024, <https://github.com/caoyunkang/AdaCLIP>).
The project publishes its code under the MIT License (`LICENSE`, verbatim). The
authors published these weights through Google Drive links in the project README
without a separate licence statement; they are redistributed here unchanged, as
released, so that Cuvis.AI can provision them without Google Drive.

The `license: unknown` tag reflects that no licence statement covers the weights; the
`LICENSE` file is the project's code licence. The checkpoints were trained on
auxiliary anomaly-detection datasets (MVTec AD, VisA, ClinicDB, ColonDB). MVTec AD is
licensed CC BY-NC-SA 4.0; check the training-data licences for your use. Naming: the
upstream README's weights table labels the Drive file `pretrained_mvtec_colondb.pth`
as "MVTec AD & ClinicDB" and `pretrained_visa_clinicdb.pth` as "VisA & ColonDB",
while its Train section pairs MVTec AD with ColonDB and VisA with ClinicDB, matching
the file names. This mirror keeps the upstream file names and renames nothing. The
AdaCLIP authors have been notified of this mirror. If you are a rights holder and
object to this redistribution, open a discussion on this repository and the files
will be taken down.
"""

MIRRORS: dict[str, Mirror] = {
    "sam3": Mirror(
        repo="sam3",
        title="SAM 3 (Segment Anything with Concepts) checkpoint",
        plugin="cuvis-ai-sam3",
        license_id="other",
        license_name="SAM License",
        license_front_matter_extra=(
            "license_name: sam-license",
            "license_link: LICENSE",
        ),
        upstream_md=f"[`facebook/sam3`](https://huggingface.co/facebook/sam3) at revision `{_SAM3_REV}`",
        base_model="facebook/sam3",
        extra_tags=("sam3", "segmentation"),
        license_text=_SAM_LICENSE_TEXT,
        files=(
            MirrorFile(
                name="sam3.pt",
                source=Source(
                    "hf",
                    repo_id="facebook/sam3",
                    filename="sam3.pt",
                    revision=_SAM3_REV,
                    needs_token=True,
                ),
                sha256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
                registry_name="sam3",
                description="SAM3 checkpoint (mirror of facebook/sam3)",
            ),
            MirrorFile(
                name="config.json",
                source=Source(
                    "hf",
                    repo_id="facebook/sam3",
                    filename="config.json",
                    revision=_SAM3_REV,
                    needs_token=True,
                ),
                sha256="4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
                aux_of="sam3",
            ),
            MirrorFile(
                name="LICENSE",
                source=Source(
                    "hf",
                    repo_id="facebook/sam3",
                    filename="LICENSE",
                    revision=_SAM3_REV,
                    needs_token=True,
                ),
                sha256="bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448",
            ),
        ),
    ),
    "efficient-track-anything": Mirror(
        repo="efficient-track-anything",
        title="EfficientTAM checkpoints",
        plugin="cuvis-ai-rtsam2",
        license_id="apache-2.0",
        license_name="Apache-2.0",
        upstream_md=(
            f"[`{_ETAM_REPO}`](https://huggingface.co/{_ETAM_REPO}) at revision "
            f"`{_ETAM_REV}` (code and licence: <https://github.com/yformer/EfficientTAM>)"
        ),
        base_model=_ETAM_REPO,
        extra_tags=("efficienttam", "video-object-segmentation"),
        license_text=_APACHE_TEXT.substitute(
            origin="<https://github.com/yformer/EfficientTAM> (the upstream Hugging Face "
            "repository carries the `apache-2.0` tag but ships no LICENSE file)"
        ),
        files=tuple(
            MirrorFile(
                name=fn,
                source=Source(
                    "hf", repo_id=_ETAM_REPO, filename=fn, revision=_ETAM_REV
                ),
                sha256=sha,
                registry_name=reg,
                description=desc,
            )
            for fn, sha, reg, desc in (
                (
                    "efficienttam_s.pt",
                    "2b572be30d9e96ee29c8d785fe157c6b079ede7d56fbc8a3671d4120e63c89cd",
                    "efficienttam_s",
                    "EfficientTAM small checkpoint (RTSAM2 default)",
                ),
                (
                    "efficienttam_ti.pt",
                    "acbb17b28cca1f860acee09c9ecb6efdb732080dc7a85a07292c31813175fa7d",
                    "efficienttam_ti",
                    "EfficientTAM tiny checkpoint (RTSAM2)",
                ),
                (
                    "efficienttam_s_512x512.pt",
                    "67b5840012737ed2c94a4cb8787c5c1b27b3a946045d2e602e65ef77230b6085",
                    "efficienttam_s_512x512",
                    "EfficientTAM small checkpoint, 512x512 input (RTSAM2)",
                ),
                (
                    "efficienttam_ti_512x512.pt",
                    "7d4d652a465f0081391050932f45d8a66768ccc99c8ea393ce8a5927e83f3b9b",
                    "efficienttam_ti_512x512",
                    "EfficientTAM tiny checkpoint, 512x512 input (RTSAM2)",
                ),
            )
        )
        + (
            MirrorFile(
                name="LICENSE",
                source=Source(
                    "url",
                    url="https://raw.githubusercontent.com/yformer/EfficientTAM/main/LICENSE",
                ),
                sha256="c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
            ),
        ),
    ),
    "dinov2": Mirror(
        repo="dinov2",
        title="DINOv2 ViT-B/14 with registers, pretrained backbone",
        plugin="cuvis-ai-dinomaly",
        license_id="apache-2.0",
        license_name="Apache-2.0",
        upstream_md=(
            "[`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2), "
            "checkpoint served from `dl.fbaipublicfiles.com`"
        ),
        extra_tags=("dinov2", "vision-transformer", "backbone"),
        license_text=_APACHE_TEXT.substitute(
            origin="<https://github.com/facebookresearch/dinov2>"
        ),
        files=(
            MirrorFile(
                name="dinov2_vitb14_reg4_pretrain.pth",
                source=Source(
                    "url",
                    url=(
                        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
                        "dinov2_vitb14_reg4_pretrain.pth"
                    ),
                ),
                sha256="73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
                local_candidates=(
                    _DINOMALY_REPO_DIR
                    / "pre_trained"
                    / "dinov2_vitb14_reg4_pretrain.pth",
                ),
                registry_name="dinov2_vitb14_reg4",
                description="DINOv2 ViT-B/14 reg4 backbone (Dinomaly encoder)",
            ),
            MirrorFile(
                name="LICENSE",
                source=Source(
                    "url",
                    url="https://raw.githubusercontent.com/facebookresearch/dinov2/main/LICENSE",
                ),
                sha256="600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
            ),
        ),
    ),
    "clip": Mirror(
        repo="clip",
        title="OpenAI CLIP ViT-L/14@336px checkpoint",
        plugin="cuvis-ai-adaclip",
        license_id="unknown",
        license_name="unspecified (code: MIT)",
        upstream_md=(
            "[`openai/CLIP`](https://github.com/openai/CLIP), checkpoint served from "
            "`openaipublic.azureedge.net`"
        ),
        extra_tags=("clip", "vision-language", "backbone"),
        license_text=_CLIP_TEXT,
        files=(
            MirrorFile(
                name="ViT-L-14-336px.pt",
                source=Source(
                    "url",
                    url=(
                        "https://openaipublic.azureedge.net/clip/models/"
                        "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/"
                        "ViT-L-14-336px.pt"
                    ),
                ),
                sha256="3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
                local_candidates=(_HOME / ".cache" / "clip" / "ViT-L-14-336px.pt",),
                registry_name="clip_vit_l_14_336",
                description="OpenAI CLIP ViT-L/14@336px backbone (AdaCLIP)",
            ),
            MirrorFile(
                name="LICENSE",
                source=Source(
                    "url",
                    url="https://raw.githubusercontent.com/openai/CLIP/main/LICENSE",
                ),
                sha256="987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db",
            ),
        ),
    ),
    "adaclip": Mirror(
        repo="adaclip",
        title="AdaCLIP pretrained prompt weights",
        plugin="cuvis-ai-adaclip",
        license_id="unknown",
        license_name="unspecified (code: MIT)",
        upstream_md=(
            "[`caoyunkang/AdaCLIP`](https://github.com/caoyunkang/AdaCLIP) (weights "
            "published through Google Drive links in the project README)"
        ),
        extra_tags=("adaclip", "anomaly-detection", "zero-shot"),
        license_text=_ADACLIP_TEXT,
        files=(
            MirrorFile(
                name="pretrained_all.pth",
                source=Source("gdrive", file_id="1Cgkfx3GAaSYnXPLolx-P7pFqYV0IVzZF"),
                sha256="33e8d3db1cb4aab030866b8b70a46e10aa27ebf2c23b5463cb07f2574addd98c",
                local_candidates=(
                    _HOME / ".cache" / "cuvis_ai" / "adaclip" / "pretrained_all.pth",
                ),
                registry_name="adaclip_all",
                description="AdaCLIP heads trained on all auxiliary datasets",
            ),
            # The Drive file names and the upstream README row labels disagree
            # (the README calls 1xVX... "MVTec AD & ClinicDB"); the mirror keeps the
            # file names exactly as published.
            MirrorFile(
                name="pretrained_mvtec_colondb.pth",
                source=Source("gdrive", file_id="1xVXANHGuJBRx59rqPRir7iqbkYzq45W0"),
                sha256="be51a42c052bd4cf060e54f503a1f5d0b2a3b899bc8dc2e243042f18b215427e",
                registry_name="adaclip_mvtec_colondb",
                description=(
                    "AdaCLIP heads, upstream file pretrained_mvtec_colondb.pth "
                    "(README row: MVTec AD and ClinicDB)"
                ),
            ),
            MirrorFile(
                name="pretrained_visa_clinicdb.pth",
                source=Source("gdrive", file_id="1QGmPB0ByPZQ7FucvGODMSz7r5Ke5wx9W"),
                sha256="3deabbbaf1e412cfdfcb42923a500b986f4b9ee96ccbc7a735d89dbc87df44c8",
                registry_name="adaclip_visa_clinicdb",
                description=(
                    "AdaCLIP heads, upstream file pretrained_visa_clinicdb.pth "
                    "(README row: VisA and ColonDB)"
                ),
            ),
            MirrorFile(
                name="LICENSE",
                source=Source(
                    "url",
                    url="https://raw.githubusercontent.com/caoyunkang/AdaCLIP/main/LICENSE",
                ),
                sha256="58bf3cbb252fb8ee158f71b5eefa0f93e24632f587926659eb2638aa0df6c618",
            ),
        ),
    ),
}


# ----------------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------------

_CARD = Template(
    """\
---
license: $license_id
$license_extra$base_model_line
tags:
$tags
---

# $title (mirror)

Byte-identical mirror of $upstream_md for the
[Cuvis.AI](https://github.com/cubert-hyperspectral/cuvis-ai) plugins. No fine-tuning,
conversion or re-serialization: the files below are the upstream release as published,
with the upstream licence text in `LICENSE`.

## Why this mirror exists

Cuvis.AI provisions model weights once into a shared cache (`download-model download
<name>` from `cuvis-ai-core`) and then runs its pipelines in an offline, token-free
runtime that only reads that cache. Hosting the exact upstream files under the
`cubert-gmbh` organisation makes that provisioning reproducible (commit-pinned and
sha256-verified) and takes the per-user Hugging Face account and token out of the
users' path.

## Files and provenance

| File | Upstream source | Upstream revision / id | sha256 | Size |
|---|---|---|---|---|
$rows

Mirrored on $date by Cubert GmbH from the sources above. The sha256 values are the
upstream values; `tools/mirror_weights.py check` in `cuvis-ai-core` re-verifies this
repository against them and against the upstream licence text.

## Licence

$license_text
## Usage with Cuvis.AI

```text
$usage
```

`download-model` (from `cuvis-ai-core`) provisions the file into the shared Hugging
Face cache and verifies its sha256. From cuvis-ai-core 0.16.0 on,
`ModelWeights.resolve("<name>")` returns the cached path and the `$plugin` nodes
resolve their weights through it, so a pipeline needs no manual step once the
weights are provisioned; earlier plugin releases still fetch from their original
upstream sources.
"""
)


def _render_card(mirror: Mirror, resolved: dict[str, tuple[Path, str]]) -> str:
    rows = []
    for f in mirror.files:
        path, sha = resolved[f.name]
        rows.append(
            f"| `{f.name}` | {f.source.label()} | "
            f"{f.source.revision_label(pinned=f.sha256 is not None)} | "
            f"`{sha}` | {_human_size(path.stat().st_size)} |"
        )
    names = [f.registry_name for f in mirror.files if f.registry_name]
    usage = "\n".join(f"uv run download-model download {n}" for n in names)
    tags = "\n".join(f"  - {t}" for t in ("mirror", "cuvis-ai", *mirror.extra_tags))
    extra = "".join(f"{line}\n" for line in mirror.license_front_matter_extra)
    base = f"base_model: {mirror.base_model}\n" if mirror.base_model else ""
    return _CARD.substitute(
        license_id=mirror.license_id,
        license_extra=extra,
        base_model_line=base.rstrip("\n") if not extra else base.rstrip("\n"),
        tags=tags,
        title=mirror.title,
        upstream_md=mirror.upstream_md,
        rows="\n".join(rows),
        date=_dt.date.today().isoformat(),
        license_text=mirror.license_text,
        usage=usage,
        plugin=mirror.plugin,
    ).replace("\n\ntags:", "\ntags:")


# ----------------------------------------------------------------------------
# Resolution: local copy, HF cache, URL, Google Drive
# ----------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_SHA_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(n: int) -> str:
    # Decimal units, the same convention the Hugging Face file browser shows.
    if n >= 10**9:
        return f"{n / 10**9:.2f} GB"
    if n >= 10**6:
        return f"{n / 10**6:.1f} MB"
    return f"{n / 10**3:.1f} kB"


def _log(msg: str) -> None:
    click.echo(msg, err=True)


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    _log(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
        expected = resp.length  # Content-Length, when the server sends one
        with part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=_SHA_READ_CHUNK)
            written = out.tell()
    # http.client returns b"" on a dropped connection instead of raising, so a
    # short body must be caught here or a truncated file gets hashed and mirrored.
    if expected is not None and written != expected:
        part.unlink(missing_ok=True)
        raise click.ClickException(
            f"short read for {url}: got {written} of {expected} bytes"
        )
    os.replace(part, dest)
    return dest


def _download_gdrive(file_id: str, dest_dir: Path, expected_name: str) -> Path:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - maintainer tool
        raise click.ClickException(
            "gdown is not installed; run with `uv run --no-sync --with gdown ...`"
        ) from exc
    dest_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  downloading Google Drive file {file_id}")
    # A directory output keeps the original Drive filename, which is what we mirror.
    out = gdown.download(id=file_id, output=str(dest_dir) + os.sep, quiet=True)
    if out is None:
        raise click.ClickException(f"Google Drive download failed for {file_id}")
    got = Path(out)
    if got.name != expected_name:
        raise click.ClickException(
            f"Google Drive file {file_id} is named '{got.name}' upstream, the table "
            f"expects '{expected_name}'. Fix the table; filenames must match upstream."
        )
    return got


def _resolve_file(
    mirror: Mirror, f: MirrorFile, work_dir: Path, token: str | None
) -> Path:
    """Return a local path holding ``f`` with the upstream bytes."""
    staged = work_dir / mirror.repo / f.name
    candidates = [staged, *f.local_candidates]
    for cand in candidates:
        if cand.is_file():
            if f.sha256 and _sha256(cand) != f.sha256:
                _log(f"  {cand} does not match the pinned sha256; ignoring it")
                continue
            return cand
    src = f.source
    if src.kind == "hf":
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=src.repo_id,
                filename=src.filename,
                revision=src.revision,
                token=token if src.needs_token else None,
            )
        )
    if src.kind == "url":
        return _download_url(src.url, staged)
    if src.kind == "gdrive":
        return _download_gdrive(src.file_id, staged.parent, f.name)
    raise click.ClickException(f"unknown source kind {src.kind!r}")


def _resolve_mirror(
    mirror: Mirror, work_dir: Path, token: str | None
) -> dict[str, tuple[Path, str]]:
    resolved: dict[str, tuple[Path, str]] = {}
    for f in mirror.files:
        _log(f"[{mirror.repo}] {f.name}")
        path = _resolve_file(mirror, f, work_dir, token)
        sha = _sha256(path)
        if f.sha256 and sha != f.sha256:
            raise click.ClickException(
                f"{mirror.repo}/{f.name}: sha256 {sha} != pinned {f.sha256} ({path})"
            )
        _log(f"  {path} sha256={sha}{' (pin OK)' if f.sha256 else ' (recorded)'}")
        resolved[f.name] = (path, sha)
    return resolved


# ----------------------------------------------------------------------------
# Registry entries
# ----------------------------------------------------------------------------


def _registry_entries(
    mirror: Mirror, resolved: dict[str, tuple[Path, str]], revision: str
) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for f in mirror.files:
        if not f.registry_name:
            continue
        aux = {
            a.name: resolved[a.name][1]
            for a in mirror.files
            if a.aux_of == f.registry_name
        }
        entry = {
            "repo_id": mirror.repo_id,
            "filename": f.name,
            "revision": revision,
            "sha256": resolved[f.name][1],
        }
        if aux:
            entry["aux_files"] = aux
        entry.update(
            {
                "plugin": mirror.plugin,
                "license": mirror.license_name,
                "description": f.description,
            }
        )
        entries[f.registry_name] = entry
    return entries


def _print_registry(entries: dict[str, dict]) -> None:
    for name, e in entries.items():
        click.echo(f'        "{name}": {{')
        for k, v in e.items():
            click.echo(f"            {json.dumps(k)}: {json.dumps(v)},")
        click.echo("        },")


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------


def _select(names: tuple[str, ...]) -> list[Mirror]:
    if not names or "all" in names:
        return list(MIRRORS.values())
    unknown = [n for n in names if n not in MIRRORS]
    if unknown:
        raise click.ClickException(
            f"unknown mirror(s) {unknown}; known: {list(MIRRORS)}"
        )
    return [MIRRORS[n] for n in names]


@click.group()
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_WORK_DIR,
    show_default=True,
    help="Staging dir for downloads, rendered cards and result JSON.",
)
@click.option(
    "--token", default=None, help="HF token (else the logged-in token / $HF_TOKEN)."
)
@click.pass_context
def cli(ctx: click.Context, work_dir: Path, token: str | None) -> None:
    """Build, upload and audit the cubert-gmbh weight mirrors."""
    ctx.obj = {"work_dir": work_dir, "token": token or os.getenv("HF_TOKEN")}


@cli.command()
@click.argument("names", nargs=-1)
@click.pass_context
def plan(ctx: click.Context, names: tuple[str, ...]) -> None:
    """Resolve, hash and render everything without touching the Hub."""
    work_dir, token = ctx.obj["work_dir"], ctx.obj["token"]
    for mirror in _select(names):
        resolved = _resolve_mirror(mirror, work_dir, token)
        card = work_dir / mirror.repo / "README.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(_render_card(mirror, resolved), encoding="utf-8", newline="\n")
        _log(f"[{mirror.repo}] card rendered to {card}")
        click.echo(f"# {mirror.repo_id} (revision unknown until upload)")
        _print_registry(_registry_entries(mirror, resolved, "<mirror commit>"))


@cli.command()
@click.argument("names", nargs=-1)
@click.option(
    "--message",
    default=None,
    help="Commit message (default: 'Mirror <upstream> for Cuvis.AI').",
)
@click.pass_context
def upload(ctx: click.Context, names: tuple[str, ...], message: str | None) -> None:
    """Create (public, ungated) repos and push one commit per mirror."""
    from huggingface_hub import CommitOperationAdd, HfApi

    work_dir, token = ctx.obj["work_dir"], ctx.obj["token"]
    api = HfApi(token=token)
    for mirror in _select(names):
        resolved = _resolve_mirror(mirror, work_dir, token)
        card_text = _render_card(mirror, resolved)
        card = work_dir / mirror.repo / "README.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(card_text, encoding="utf-8", newline="\n")

        # Private until the commit lands, so a failed upload never leaves an empty
        # public repo on the organisation page.
        api.create_repo(
            mirror.repo_id, repo_type="model", visibility="private", exist_ok=True
        )
        ops = [
            CommitOperationAdd(
                path_in_repo=f.name, path_or_fileobj=str(resolved[f.name][0])
            )
            for f in mirror.files
        ]
        ops.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card))
        )
        _log(f"[{mirror.repo}] uploading {len(ops)} files to {mirror.repo_id}")
        try:
            info = api.create_commit(
                repo_id=mirror.repo_id,
                repo_type="model",
                operations=ops,
                commit_message=message or f"Mirror {mirror.title} for Cuvis.AI",
                commit_description=(
                    "Byte-identical upstream files plus LICENSE and model card; "
                    "see README.md for provenance and sha256 values."
                ),
            )
        except Exception as exc:
            raise click.ClickException(
                f"{mirror.repo_id}: upload failed ({exc}). The repo stays private "
                f"and incomplete; re-run `upload {mirror.repo}`."
            ) from exc
        api.update_repo_settings(mirror.repo_id, visibility="public", gated=False)
        revision = info.oid
        _log(f"[{mirror.repo}] commit {revision} -> {info.commit_url}")
        entries = _registry_entries(mirror, resolved, revision)
        result = work_dir / f"{mirror.repo}.result.json"
        result.write_text(
            json.dumps(
                {
                    "repo_id": mirror.repo_id,
                    "revision": revision,
                    "files": {n: s for n, (_, s) in resolved.items()},
                    "registry": entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        click.echo(f"# {mirror.repo_id} @ {revision}")
        _print_registry(entries)


@cli.command()
@click.argument("names", nargs=-1)
@click.pass_context
def check(ctx: click.Context, names: tuple[str, ...]) -> None:
    """Audit mirrors: sha256 pins, upstream LICENSE text, visibility, gating, and
    the core registry (``ModelWeights._models`` of the checkout you run from)
    against the Hub."""
    from huggingface_hub import HfApi, hf_hub_download

    from cuvis_ai_core.data.model_weights import ModelWeights

    work_dir, token = ctx.obj["work_dir"], ctx.obj["token"]
    api = HfApi(token=token)
    failures = 0

    def report(ok: bool, message: str) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        click.echo(("OK   " if ok else "FAIL ") + message)

    def hub_sha(mirror: Mirror, sib) -> str:
        # LFS files carry their sha256 in the metadata; small files are fetched.
        lfs_sha = getattr(sib.lfs, "sha256", None) if sib.lfs else None
        if lfs_sha is not None:
            return lfs_sha
        local = Path(
            hf_hub_download(
                mirror.repo_id, sib.rfilename, cache_dir=work_dir / "_check"
            )
        )
        return _sha256(local)

    selected = _select(names)
    for mirror in selected:
        info = api.model_info(mirror.repo_id, files_metadata=True)
        report(
            not info.private and not info.gated,
            f"{mirror.repo_id}: private={info.private} gated={info.gated}",
        )
        by_name = {s.rfilename: s for s in info.siblings or []}
        report("README.md" in by_name, f"{mirror.repo_id}: README.md (model card)")
        for f in mirror.files:
            sib = by_name.get(f.name)
            if sib is None:
                report(False, f"{mirror.repo_id}/{f.name}: missing on the Hub")
                continue
            actual = hub_sha(mirror, sib)
            if f.sha256:
                report(
                    actual == f.sha256,
                    f"{mirror.repo_id}/{f.name}: sha256 pin ({actual[:12]}...)",
                )
            else:
                click.echo(f"INFO {mirror.repo_id}/{f.name}: sha256 {actual} (no pin)")
            if f.name == "LICENSE":
                # Always a fresh upstream copy, never the staged one, so a licence
                # change upstream is noticed on every audit.
                fresh_dir = work_dir / "_upstream"
                (fresh_dir / mirror.repo / f.name).unlink(missing_ok=True)
                upstream = _resolve_file(mirror, f, fresh_dir, token)
                report(
                    _sha256(upstream) == actual,
                    f"{mirror.repo_id}/LICENSE: matches the upstream text",
                )
        # The runtime registry must agree with the Hub: every file it names exists
        # with the pinned sha256, and its revision is a commit of the mirror repo.
        for name, entry in ModelWeights._models.items():
            if entry["repo_id"] != mirror.repo_id:
                continue
            pinned = {entry["filename"]: entry["sha256"]}
            pinned.update(entry.get("aux_files") or {})
            for fname, sha in pinned.items():
                sib = by_name.get(fname)
                if sib is None:
                    report(False, f"registry {name}: {fname} missing on the Hub")
                    continue
                report(
                    hub_sha(mirror, sib) == sha, f"registry {name}: sha256 of {fname}"
                )
            revision = entry.get("revision")
            try:
                rev_info = api.repo_info(mirror.repo_id, revision=revision)
                report(
                    rev_info.sha == revision, f"registry {name}: revision {revision}"
                )
            except Exception as exc:  # unknown revision surfaces as an HTTP error
                report(False, f"registry {name}: revision {revision} ({exc})")
    if not names or "all" in names:
        covered = {m.repo_id for m in selected}
        for name, entry in ModelWeights._models.items():
            report(
                entry["repo_id"] in covered,
                f"registry {name}: repo {entry['repo_id']} has a mirror table entry",
            )
    if failures:
        raise SystemExit(f"{failures} check(s) failed")
    click.echo("all checks passed")


if __name__ == "__main__":
    cli()
