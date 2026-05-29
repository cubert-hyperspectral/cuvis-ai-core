"""Child runtime process — gRPC server the parent talks to.

Boots inside a composed per-pipeline venv and hosts the RunRuntime
service. Reuses the existing in-process service classes
(``PipelineService`` / ``InferenceService`` / ``TrainingService``)
but skips the manifest-driven plugin install path — every plugin in
the child's venv is already a real installed package.
"""

from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

__all__ = ["RunRuntimeServicer"]
