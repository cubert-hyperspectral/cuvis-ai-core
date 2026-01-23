"""Utility functions and helpers for cuvis.ai."""


def restore_pipeline(*args, **kwargs):
    from cuvis_ai_core.utils.restore import restore_pipeline as _restore_pipeline

    return _restore_pipeline(*args, **kwargs)


def restore_trainrun(*args, **kwargs):
    from cuvis_ai_core.utils.restore import restore_trainrun as _restore_trainrun

    return _restore_trainrun(*args, **kwargs)


__all__ = ["restore_pipeline", "restore_trainrun"]
