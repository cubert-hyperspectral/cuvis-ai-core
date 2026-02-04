"""Factory functions for creating PyTorch Lightning callbacks from configuration."""

from cuvis_ai_schemas.training import CallbacksConfig


def create_callbacks_from_config(config: CallbacksConfig | None) -> list:
    """Create PyTorch Lightning callback instances from configuration."""
    if config is None:
        return []

    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    callbacks = []

    if config.early_stopping:
        for es_config in config.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=es_config.monitor,
                    patience=es_config.patience,
                    mode=es_config.mode,
                    min_delta=es_config.min_delta,
                    stopping_threshold=es_config.stopping_threshold,
                    verbose=es_config.verbose,
                    strict=es_config.strict,
                    check_finite=es_config.check_finite,
                    divergence_threshold=es_config.divergence_threshold,
                    check_on_train_epoch_end=es_config.check_on_train_epoch_end,
                    log_rank_zero_only=es_config.log_rank_zero_only,
                )
            )

    if config.checkpoint is not None:
        mc_config = config.checkpoint
        callbacks.append(
            ModelCheckpoint(
                dirpath=mc_config.dirpath,
                filename=mc_config.filename,
                monitor=mc_config.monitor,
                mode=mc_config.mode,
                save_top_k=mc_config.save_top_k,
                save_last=mc_config.save_last,
                verbose=mc_config.verbose,
                auto_insert_metric_name=mc_config.auto_insert_metric_name,
                every_n_epochs=mc_config.every_n_epochs,
                save_on_exception=mc_config.save_on_exception,
                save_weights_only=mc_config.save_weights_only,
                every_n_train_steps=mc_config.every_n_train_steps,
                train_time_interval=mc_config.train_time_interval,
                save_on_train_epoch_end=mc_config.save_on_train_epoch_end,
                enable_version_counter=mc_config.enable_version_counter,
            )
        )

    if config.learning_rate_monitor is not None:
        lr_config = config.learning_rate_monitor
        callbacks.append(
            LearningRateMonitor(
                logging_interval=lr_config.logging_interval,
                log_momentum=lr_config.log_momentum,
                log_weight_decay=lr_config.log_weight_decay,
            )
        )

    return callbacks


__all__ = ["create_callbacks_from_config"]
