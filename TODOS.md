# TODOs

## Map `UvRunnerError` to a structured gRPC status on the orchestrator servicer path

- **What:** Decorate the parent orchestrator servicer methods (`grpc/service.py`, e.g.
  `LoadPipeline`) so compose failures like `UvRunnerError` surface as a structured status
  (`FAILED_PRECONDITION`) instead of `UNKNOWN` with message text only.
- **Why:** Today `TheiaService.LoadPipeline` delegates undecorated, so any compose-time error
  reaches clients as `StatusCode.UNKNOWN`. The error *message* is now descriptive (uv/git
  resolution hardening), but clients that want to branch programmatically (retry, surface a
  setup dialog, distinguish "env broken" from "pipeline invalid") have to parse strings.
- **Pros:** Clean client contract for every host (CuvisNEXT and beyond); aligns the parent
  servicer with the `@grpc_handler` mapping the in-process child services already use
  (`grpc/error_handling.py` maps `FileNotFoundError` → `NOT_FOUND` etc.).
- **Cons:** Touches the parent servicer's error-handling architecture; needs a deliberate
  decision on which exceptions map to which codes and whether `@grpc_handler` is reused or a
  dedicated mapping is added; risk of changing behavior clients already depend on.
- **Context:** Identified during the CuvisNEXT MR !191 uv-discovery review (2026-07-18). The
  orchestrator composes per-pipeline envs by spawning `uv`; a missing binary used to surface as
  an opaque `[WinError 2]`. The resolution/diagnosability fix landed in
  `orchestrator/uv_runner.py` (`CUVIS_UV` → `shutil.which` → `uv.find_uv_bin`, clear
  `UvRunnerError`) and `orchestrator/runtime_project.py` (`git` named when missing). This item
  is the remaining structural half.
- **Depends on / blocked by:** the uv/git resolution hardening (same change set as this file).
