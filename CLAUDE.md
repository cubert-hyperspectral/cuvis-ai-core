# cuvis-ai-core

The **framework** underneath Cuvis.AI. Provides the `Node` base class and typed port system,
the pipeline graph builder/executor, the plugin loader, serialization & restore, a
PyTorch-Lightning training stack, and a gRPC server with 10+ services. The server is a thin
orchestrator that never imports plugin modules — each run composes its own venv.

## Part of the Cuvis.AI ecosystem

`cuvis-ai-schemas` (contracts) → **`cuvis-ai-core`** (this repo) → `cuvis-ai` (node library
+ CLIs) → plugins. `cuvis-ai-cookbook` = examples; `cuvis-ai-agentic-skills` = Claude Code
plugin; `dev-docs` = internal ticket docs. This package depends on `cuvis-ai-schemas`.

## Layout

- `cuvis_ai_core/node/` — `Node` base class (`nn.Module` + ABC + Serializable).
- `cuvis_ai_core/pipeline/` — pipeline graph construction & execution.
- `cuvis_ai_core/orchestrator/` + `run_runtime/` — per-run child-env venv composition, cache,
  and the runtime that executes a pipeline/trainrun in that env.
- `cuvis_ai_core/training/` — Lightning datamodules, `GradientTrainer`, `StatisticalTrainer`, `Predictor`.
- `cuvis_ai_core/grpc/` — gRPC service implementations + session management.
- `cuvis_ai_core/deciders/`, `data/`, `utils/` — decision nodes, dataset loaders, restore/plugin CLIs.

## Build & test

- Install (dev): `uv sync --all-extras --dev`. Use `uv`, never bare `pip`.
- Tests: `uv run pytest`; coverage `uv run pytest --cov=cuvis_ai_core --cov-report=term-missing`.
- Markers: `unit`, `integration`, `grpc`, `pipeline`, `slow`, `gpu`, `stress`, `requires_data`.
- CLI entry points: `restore-pipeline`, `restore-trainrun`, `dataset`, `suggest-plugins-fix`.
- **Test data:** the data-backed tests (training round-trips, `test_config_preservation`)
  expect the Lentils session at `data/Lentils/Lentils_000.cu3s`. Fetch it with
  `uv run dataset download Lentils` (lands in `./data/Lentils/`); `uv run dataset list`
  shows the full registry. Without it those tests fail/skip on the missing file — an env
  gap, not a regression.

## Key abstractions

- `cuvis_ai_core.node.Node` — subclasses declare `INPUT_SPECS`/`OUTPUT_SPECS` as class dicts
  of `PortSpec` (from `cuvis_ai_schemas.pipeline`); `super().__init__(**params, **kwargs)` last.
- `cuvis_ai_core.utils.node_registry.NodeRegistry` — loads built-in + plugin nodes.
- gRPC surface is **session-based**: LoadPipeline / Inference / Train / RestoreTrainRun /
  CloseSession. **Read the service files before reasoning about RPC names** — don't trust
  doc-named RPCs without confirming they exist.
- **Two plugin-loading paths.** Production is the gRPC orchestrator: the server never
  imports plugins; each run composes a per-run venv with the plugins pre-installed and a
  child runtime imports them with no `sys.path` mutation. The `restore-pipeline` /
  `restore-trainrun` CLIs are the **developer inner-loop**: they eager-load plugins into the
  *current* venv (`uv pip install` + `sys.path` + `NodeRegistry.load_plugins`). Keep the CLI
  path — no env compose, no child spawn, no gRPC; it picks up an editable plugin's working
  tree immediately, is debuggable in one process, and the agentic-skills tooling shells out
  to it. It is explicitly **not** the production path.

## Conventions

- ruff line length **88**; interrogate docstring coverage **80%**; `.githooks` pre-commit runs
  `ruff format` + `ruff check --fix`.
- `register_buffer` is only for fitted statistical state; pretrained network weights stay as
  `nn.Module` params with `requires_grad_(False)`.
- New node classes omit the `Node` suffix (e.g. `SpectralAngleMapper`, not `SpectralAngleMapperNode`).
- No Jira IDs / "Phase N" / migration tags in shipped code, comments, or docstrings.
- No Claude/AI mentions or `Co-Authored-By` trailers in commit messages.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **cuvis-ai-core** (6270 symbols, 12579 relationships, 256 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/cuvis-ai-core/context` | Codebase overview, check index freshness |
| `gitnexus://repo/cuvis-ai-core/clusters` | All functional areas |
| `gitnexus://repo/cuvis-ai-core/processes` | All execution flows |
| `gitnexus://repo/cuvis-ai-core/process/{name}` | Step-by-step execution trace |

## Cross-Repo Groups

This repository is listed under GitNexus **group(s): cuvis-ai-group** (see `~/.gitnexus/groups/`). For cross-repo analysis, use MCP tools `impact`, `query`, and `context` with `repo` set to `@<groupName>` or `@<groupName>/<memberPath>` (paths match keys in that group’s `group.yaml`). Use `group_list` / `group_sync` for membership and sync. From the terminal: `npx gitnexus group list`, `npx gitnexus group sync <name>`, `npx gitnexus group impact <name> --target <symbol> --repo <group-path>`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
