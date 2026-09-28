"""Per-run child-env orchestrator.

Composes a cached venv per pipeline plugin set, then spawns a child
runtime service inside it. The server itself never imports plugin
modules. Import the submodules directly (``composer``, ``cache_key``,
``plugin_capabilities``, ``venv_paths``, ...); the package exports nothing.
"""
