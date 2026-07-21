"""Headless reference driver for the external-GUI training flow.

Drives a running production server through the exact call sequence the
CuvisNEXT training wizard uses, and asserts the wire behavior the GUI relies
on.

Author the frozen splits.json first (the reference implementation of the GUI
authoring semantics, from the cuvis-ai-dataloader env):

    resolve-splits --data-module cu3s \
        --data-arg data_dir=<cu3s folder> --data-arg frames=measurements \
        --data-arg recursive=true --strategy stratified --ad-aware --seed 7 \
        --out lentils_gui.json

Then run this driver against a server whose environment has the cuvis-ai
preset configs installed (the trainrun path below points into that package):

    python -m cuvis_ai_core.grpc.production_server   # terminal 1
    python examples/cuvisnext_train_flow.py \
        --trainrun <site-packages>/cuvis_ai/configs/trainrun/dinomaly_rgb_cuvisnext.yaml \
        --plugins-dir <site-packages>/cuvis_ai/configs/plugins \
        --data-dir D:/data/lentils_cu3s \
        --splits-json D:/experiments/splits/lentils_gui.json   # terminal 2

Sequence (each step is the GUI contract, not just a smoke test):

1. CreateSession + LoadPlugin per manifest the preset needs (plugin resolution
   for RestoreTrainRun runs against this session's catalog).
2. RestoreTrainRun(preset, session_id) -> restored TrainRunConfig.
3. SetTrainRunConfig with the per-run fills (data_dir, splits_path, epochs,
   output_dir, tags.cuvisnext.*) and the pipeline reference stripped.
4. Train(STATISTICAL) to terminal COMPLETE.
5. Train(GRADIENT), asserting epoch-level train_loss AND val_loss appear on
   the stream; optional one-epoch wall-clock benchmark.
6. StopTrain mid-gradient on a second run -> terminal CANCELLED.
7. StopTrain in the between-phase window -> the next Train stream terminates
   immediately with CANCELLED; SetTrainRunConfig clears the flag.
8. GetTrainStatus reflects the last streamed response.
9. SaveTrainRun, then RestoreTrainRun(saved) into a fresh prepared session and
   a short Train to prove the persisted run replays.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import grpc
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc

STATUS_NAME = {
    cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED: "UNSPECIFIED",
    cuvis_ai_pb2.TRAIN_STATUS_RUNNING: "RUNNING",
    cuvis_ai_pb2.TRAIN_STATUS_COMPLETE: "COMPLETE",
    cuvis_ai_pb2.TRAIN_STATUS_ERROR: "ERROR",
    cuvis_ai_pb2.TRAIN_STATUS_CANCELLED: "CANCELLED",
}


def _check(condition: bool, message: str) -> None:
    """Fail the flow loudly on a broken contract expectation."""
    if not condition:
        click.secho(f"FAIL: {message}", fg="red", err=True)
        sys.exit(1)
    click.secho(f"  ok: {message}", fg="green")


def _load_plugins(stub, session_id: str, plugins_dir: Path, names: list[str]) -> None:
    """Register each named manifest into the session, path-resolved like a real client."""
    for name in names:
        manifest_path = plugins_dir / f"{name}.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and "repo" not in manifest:
            raw_path = Path(manifest.get("path", "."))
            if not raw_path.is_absolute():
                manifest["path"] = str((manifest_path.parent / raw_path).resolve())
        response = stub.LoadPlugin(
            cuvis_ai_pb2.LoadPluginRequest(
                session_id=session_id,
                manifest=cuvis_ai_pb2.PluginManifest(
                    config_bytes=json.dumps(manifest).encode("utf-8")
                ),
            )
        )
        _check(
            response.registered_plugin == name and not response.error,
            f"LoadPlugin({name})",
        )


def _prepared_session(stub, plugins_dir: Path, plugin_names: list[str]) -> str:
    """CreateSession + LoadPlugin preamble — the wizard's session preparation."""
    session_id = stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id
    _load_plugins(stub, session_id, plugins_dir, plugin_names)
    return session_id


def _filled_trainrun_bytes(
    restored: cuvis_ai_pb2.TrainRunConfig,
    *,
    data_dir: Path,
    splits_json: Path,
    epochs: int,
    output_dir: Path,
) -> bytes:
    """Apply the GUI's per-run edits to the restored trainrun config.

    Mirrors the wizard exactly: fill data_dir + splits_path, set the epoch
    count, redirect outputs, record the GUI provenance in tags, and strip the
    ``pipeline`` reference (SetTrainRunConfig rejects it — the pipeline is
    already attached to the session by RestoreTrainRun).
    """
    config = json.loads(restored.config_bytes)
    config.pop("pipeline", None)
    config["data"]["params"]["data_dir"] = data_dir.resolve().as_posix()
    config["data"]["splits"]["splits_path"] = splits_json.resolve().as_posix()
    config["training"]["max_epochs"] = epochs
    config["training"]["default_root_dir"] = output_dir.resolve().as_posix()
    checkpoint = config["training"].get("callbacks", {}).get("checkpoint")
    if checkpoint:
        checkpoint["dirpath"] = (output_dir.resolve() / "checkpoints").as_posix()
    config["output_dir"] = output_dir.resolve().as_posix()
    config.setdefault("tags", {})["cuvisnext.flow"] = "headless-verification"
    return json.dumps(config).encode("utf-8")


def _set_trainrun(stub, session_id: str, config_bytes: bytes) -> None:
    response = stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=config_bytes),
        )
    )
    _check(response.success, "SetTrainRunConfig")


def _drain(stream) -> list[cuvis_ai_pb2.TrainResponse]:
    return list(stream)


@click.command()
@click.option("--server", default="localhost:50051", show_default=True)
@click.option(
    "--trainrun",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Absolute path of the packaged preset trainrun YAML.",
)
@click.option(
    "--plugins-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of plugin manifests (cuvis_ai/configs/plugins).",
)
@click.option(
    "--plugins",
    default="cuvis_ai_builtin,dinomaly,cuvis_ai_dataloader",
    show_default=True,
    help="Comma-separated manifest names to register before RestoreTrainRun.",
)
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="cu3s dataset folder (frames=measurements universe).",
)
@click.option(
    "--splits-json",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Frozen GUI-authored splits.json.",
)
@click.option(
    "--epochs", default=2, show_default=True, help="Gradient epochs for the full run."
)
@click.option(
    "--output-dir",
    default=Path("outputs/cuvisnext_flow"),
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--benchmark/--no-benchmark",
    default=False,
    show_default=True,
    help="Time the first full gradient epoch (the pre-GUI benchmark gate).",
)
def main(
    server: str,
    trainrun: Path,
    plugins_dir: Path,
    plugins: str,
    data_dir: Path,
    splits_json: Path,
    epochs: int,
    output_dir: Path,
    benchmark: bool,
) -> None:
    """Drive the full external-GUI training flow against a live server."""
    plugin_names = [p.strip() for p in plugins.split(",") if p.strip()]
    channel = grpc.insecure_channel(
        server, options=[("grpc.max_receive_message_length", 300 * 1024 * 1024)]
    )
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    click.secho("== 1/2: prepared session + RestoreTrainRun(preset) ==", bold=True)
    session_id = _prepared_session(stub, plugins_dir, plugin_names)
    restore = stub.RestoreTrainRun(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(trainrun.resolve()), session_id=session_id
        )
    )
    _check(
        restore.session_id == session_id, "RestoreTrainRun kept the prepared session"
    )

    click.secho("== 3: SetTrainRunConfig with per-run fills ==", bold=True)
    filled = _filled_trainrun_bytes(
        restore.trainrun,
        data_dir=data_dir,
        splits_json=splits_json,
        epochs=epochs,
        output_dir=output_dir,
    )
    _set_trainrun(stub, session_id, filled)

    click.secho("== 4: statistical phase ==", bold=True)
    stat = _drain(
        stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        )
    )
    _check(
        stat[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
        f"statistical terminal status is COMPLETE (got {STATUS_NAME[stat[-1].status]})",
    )

    click.secho("== 5: gradient phase (loss curve contract) ==", bold=True)
    epoch_started = time.monotonic()
    first_epoch_seconds: float | None = None
    responses: list[cuvis_ai_pb2.TrainResponse] = []
    for response in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
        )
    ):
        responses.append(response)
        if benchmark and first_epoch_seconds is None and response.context.epoch >= 1:
            first_epoch_seconds = time.monotonic() - epoch_started
    _check(
        responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
        f"gradient terminal status is COMPLETE (got {STATUS_NAME[responses[-1].status]})",
    )
    loss_keys = {key for r in responses for key in r.losses}
    _check("train_loss" in loss_keys, "epoch-level train_loss streamed")
    _check("val_loss" in loss_keys, "epoch-level val_loss streamed")
    if benchmark and first_epoch_seconds is not None:
        gate = (
            "PROCEED (<= 120 s)"
            if first_epoch_seconds <= 120
            else "NPZ-FALLBACK DISCUSSION"
        )
        click.secho(
            f"  benchmark: first epoch {first_epoch_seconds:.1f} s -> {gate}",
            fg="yellow",
        )

    click.secho("== 6: StopTrain mid-gradient -> CANCELLED ==", bold=True)
    _set_trainrun(stub, session_id, filled)  # new run boundary
    stat2 = _drain(
        stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        )
    )
    _check(
        stat2[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE, "second run stat phase"
    )
    cancelled: list[cuvis_ai_pb2.TrainResponse] = []
    for response in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
        )
    ):
        cancelled.append(response)
        if len(cancelled) == 3:
            stop = stub.StopTrain(cuvis_ai_pb2.StopTrainRequest(session_id=session_id))
            _check(stop.accepted, "StopTrain accepted mid-stream")
    _check(
        cancelled[-1].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED,
        f"mid-run terminal status is CANCELLED (got {STATUS_NAME[cancelled[-1].status]})",
    )

    click.secho("== 7: between-phase StopTrain window ==", bold=True)
    _set_trainrun(stub, session_id, filled)  # clears the previous stop
    stat3 = _drain(
        stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        )
    )
    _check(
        stat3[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE, "third run stat phase"
    )
    stop = stub.StopTrain(cuvis_ai_pb2.StopTrainRequest(session_id=session_id))
    _check(stop.accepted, "StopTrain accepted between phases")
    late = _drain(
        stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
            )
        )
    )
    _check(
        len(late) == 1 and late[0].status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED,
        "gradient stream after between-phase stop terminates immediately CANCELLED",
    )

    click.secho("== 8: GetTrainStatus reflects the stream ==", bold=True)
    status = stub.GetTrainStatus(
        cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)
    )
    _check(
        status.latest_progress.status == cuvis_ai_pb2.TRAIN_STATUS_CANCELLED,
        "GetTrainStatus reports the terminal CANCELLED",
    )

    click.secho("== 9: SaveTrainRun -> RestoreTrainRun(saved) -> replay ==", bold=True)
    _set_trainrun(stub, session_id, filled)
    stat4 = _drain(
        stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        )
    )
    _check(
        stat4[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
        "replay-source stat phase",
    )
    saved = stub.SaveTrainRun(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=session_id,
            trainrun_path=str((output_dir / "saved" / "trainrun.yaml").resolve()),
            save_weights=True,
        )
    )
    _check(saved.success, f"SaveTrainRun -> {saved.trainrun_path}")

    replay_session = _prepared_session(stub, plugins_dir, plugin_names)
    replay = stub.RestoreTrainRun(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=saved.trainrun_path, session_id=replay_session
        )
    )
    _check(
        replay.session_id == replay_session,
        "saved trainrun restored into fresh session",
    )
    replay_stat = _drain(
        stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=replay_session,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        )
    )
    _check(
        replay_stat[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
        "restored run trains (statistical) to COMPLETE",
    )

    for sid in (session_id, replay_session):
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=sid))
    click.secho("ALL CHECKS PASSED", fg="green", bold=True)


if __name__ == "__main__":
    main()
