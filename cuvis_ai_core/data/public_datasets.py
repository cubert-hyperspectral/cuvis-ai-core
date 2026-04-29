"""Public dataset registry and downloader for CUVIS.AI.

Downloads datasets from Hugging Face Hub using ``huggingface_hub.snapshot_download``.
"""

import os
from pathlib import Path


class PublicDatasets:
    """Registry and downloader for public CUVIS.AI datasets."""

    @classmethod
    def download_dataset(
        cls,
        dataset_name: str,
        *,
        download_path: str = ".",
        force: bool = False,
    ) -> bool:
        """Download a dataset by name.

        Args:
            dataset_name: Key in the dataset registry.
            download_path: Directory to download into.
            force: Re-download even if the target directory exists.

        Returns:
            True on success (or if data already present), False on error.
        """
        try:
            dset = cls._datasets[dataset_name]
        except KeyError:
            print(f"Dataset '{dataset_name}' not found.")
            print(f"Available: {', '.join(cls._canonical_names())}")
            return False

        target_dir = os.path.join(download_path, dset["target_dir"])

        if os.path.isdir(target_dir) and not force:
            print(f"Dataset '{dataset_name}' already exists at {target_dir}")
            print("Use force=True to re-download.")
            return True

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub is not installed.")
            print("Install with: pip install huggingface-hub")
            return False

        if not os.path.exists(download_path):
            os.makedirs(download_path)

        print(f"Downloading '{dataset_name}' from Hugging Face...")
        print(f"  Repository:  {dset['repo_id']}")
        print(f"  Size:        {dset['size']}")
        print(f"  Destination: {target_dir}")

        try:
            snapshot_download(
                repo_id=dset["repo_id"],
                repo_type="dataset",
                local_dir=target_dir,
            )
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Manual download: https://huggingface.co/datasets/{dset['repo_id']}")
            return False

        print(f"Downloaded '{dataset_name}' successfully.")
        return True

    @classmethod
    def list_datasets(cls, verbose: bool = False) -> None:
        """Print available datasets.

        Args:
            verbose: Show extra detail (file listing after download).
        """
        seen_ids: set[int] = set()
        aliases: dict[int, list[str]] = {}

        # Collect aliases
        for name, data in cls._datasets.items():
            obj_id = id(data)
            aliases.setdefault(obj_id, []).append(name)

        print(f"{'Name':<25s} {'Size':>8s}  Description")
        print("-" * 70)

        for name, data in cls._datasets.items():
            obj_id = id(data)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)

            aka = [a for a in aliases[obj_id] if a != name]
            alias_str = f"  (alias: {', '.join(aka)})" if aka else ""
            print(f"  {name:<23s} {data['size']:>8s}  {data['description']}{alias_str}")

            if verbose:
                print(f"    repo: {data['repo_id']}")
                print(f"    dir:  {data['target_dir']}")

    @classmethod
    def get_target_dir(cls, dataset_name: str) -> str:
        """Return the target directory name for a dataset.

        Args:
            dataset_name: Key in the dataset registry.

        Returns:
            Directory name the dataset downloads into.

        Raises:
            KeyError: If *dataset_name* is not in the registry.
        """
        return cls._datasets[dataset_name]["target_dir"]

    @classmethod
    def _canonical_names(cls) -> list[str]:
        """Return dataset names without aliases."""
        seen: set[int] = set()
        names: list[str] = []
        for name, data in cls._datasets.items():
            obj_id = id(data)
            if obj_id not in seen:
                seen.add(obj_id)
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Dataset registry
    # ------------------------------------------------------------------

    _datasets: dict[str, dict] = {
        "Lentils_Anomaly": {
            "repo_id": "cubert-gmbh/XMR_Lentils",
            "target_dir": "Lentils",
            "description": "Lentils anomaly detection dataset",
            "size": "~200MB",
        },
        "Blood_Perfusion": {
            "repo_id": "cubert-gmbh/XMR_Blood_Perfusion",
            "target_dir": "XMR_Blood_Perfusion",
            "description": "XMR blood perfusion reflectance dataset",
            "size": "~7GB",
        },
        "Demo_Object_Tracking": {
            "repo_id": "cubert-gmbh/XMR_Demo_Object_Tracking",
            "target_dir": "XMR_Demo_Object_Tracking",
            "description": "Hyperspectral multi-person tracking demo — passive SAM3 and active spectral-ink sessions",
            "size": "~25GB",
        },
    }

    # Convenience aliases
    _datasets["lentils"] = _datasets["Lentils_Anomaly"]
    _datasets["blood_perfusion"] = _datasets["Blood_Perfusion"]
    _datasets["demo_object_tracking"] = _datasets["Demo_Object_Tracking"]
    _datasets["demo-object-tracking"] = _datasets["Demo_Object_Tracking"]


def download_data_cli() -> None:
    """CLI entry point for dataset management (``uv run dataset``)."""
    import shutil

    import click

    @click.group()
    def cli() -> None:
        """CUVIS.AI dataset management."""

    @cli.command("list")
    @click.option("--verbose", "-v", is_flag=True, help="Show extra detail.")
    def list_cmd(verbose: bool) -> None:
        """List available datasets."""
        PublicDatasets.list_datasets(verbose=verbose)

    @cli.command()
    @click.argument("name")
    @click.option(
        "--data-dir",
        type=click.Path(path_type=Path),
        default=Path.cwd() / "data",
        help="Data directory (default: ./data).",
    )
    @click.option("--force", is_flag=True, help="Re-download even if data exists.")
    def download(name: str, data_dir: Path, force: bool) -> None:
        """Download a dataset by NAME."""
        data_dir.mkdir(parents=True, exist_ok=True)

        success = PublicDatasets.download_dataset(
            name,
            download_path=str(data_dir),
            force=force,
        )

        if not success:
            raise SystemExit(1)

        # Post-download validation
        try:
            target = data_dir / PublicDatasets.get_target_dir(name)
        except KeyError:
            return
        if target.exists():
            cu3s_files = list(target.rglob("*.cu3s"))
            if cu3s_files:
                click.echo(f"\nValidation: found {len(cu3s_files)} .cu3s file(s)")
                for f in cu3s_files[:5]:
                    click.echo(f"  - {f.relative_to(data_dir)}")
            else:
                click.echo("\nWarning: no .cu3s files found in downloaded data")

        # Lentils symlink for case-insensitive access
        if name.lower() in ("lentils", "lentilsanomaly"):
            upper = data_dir / "Lentils"
            lower = data_dir / "lentils"
            if upper.exists() and not lower.exists():
                try:
                    lower.symlink_to("Lentils", target_is_directory=True)
                    click.echo("Created symlink: lentils -> Lentils")
                except OSError:
                    click.echo(
                        "Could not create symlink, copying Lentils -> lentils ..."
                    )
                    shutil.copytree(upper, lower)

    cli()
