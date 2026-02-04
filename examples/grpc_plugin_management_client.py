"""
gRPC Plugin Management Client Example

This example demonstrates how to use the gRPC API to:
1. Create a session
2. Load plugins dynamically
3. List available nodes (including plugin nodes)
4. Build pipelines using plugin nodes
5. Query plugin information
6. Manage plugin cache

Prerequisites:
    - cuvis-ai-core server running on localhost:50051
    - Plugin repository accessible (Git or local)
"""

import grpc
from loguru import logger

from cuvis_ai_core.grpc.v1 import cuvis_ai_core_pb2, cuvis_ai_core_pb2_grpc
from cuvis_ai_schemas.plugin import (
    GitPluginConfig,
    LocalPluginConfig,
    PluginManifest,
)


def example_1_basic_plugin_loading():
    """Example 1: Load a Git plugin and use it in a pipeline."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Plugin Loading")
    logger.info("=" * 60)

    # Connect to server
    channel = grpc.insecure_channel("localhost:50051")
    client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        # 1. Create session
        session_resp = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
        session_id = session_resp.session_id
        logger.info(f"✓ Session created: {session_id}")

        # 2. Create plugin manifest
        manifest = PluginManifest(
            plugins={
                "my_custom_plugin": GitPluginConfig(
                    repo="git@example.com:org/my-plugin.git",
                    tag="v1.0.0",
                    provides=["my_plugin.nodes.CustomDetector"],
                )
            }
        )

        # 3. Load plugins
        manifest_json = manifest.model_dump_json().encode()
        plugin_resp = client.LoadPlugins(
            cuvis_ai_core_pb2.LoadPluginsRequest(
                session_id=session_id,
                manifest=cuvis_ai_core_pb2.PluginManifest(config_bytes=manifest_json),
            )
        )

        logger.info(f"✓ Loaded plugins: {list(plugin_resp.loaded_plugins)}")
        if plugin_resp.failed_plugins:
            logger.warning(f"✗ Failed plugins: {dict(plugin_resp.failed_plugins)}")

        # 4. List available nodes
        nodes_resp = client.ListAvailableNodes(
            cuvis_ai_core_pb2.ListAvailableNodesRequest(session_id=session_id)
        )

        logger.info(f"✓ Available nodes ({len(nodes_resp.nodes)}):")
        for node in nodes_resp.nodes[:5]:  # Show first 5
            source_info = (
                f"from plugin '{node.plugin_name}'" if node.plugin_name else "builtin"
            )
            logger.info(f"  • {node.class_name} ({source_info})")

        # 5. Build pipeline using plugin node
        pipeline_yaml = """
metadata:
  name: Custom Plugin Pipeline
nodes:
  - name: detector
    class: my_plugin.nodes.CustomDetector
    params:
      threshold: 0.5
connections: []
"""

        pipeline_resp = client.LoadPipeline(
            cuvis_ai_core_pb2.LoadPipelineRequest(
                session_id=session_id,
                config=cuvis_ai_core_pb2.PipelineConfig(
                    config_bytes=pipeline_yaml.encode()
                ),
            )
        )
        logger.info(f"✓ Pipeline loaded: {pipeline_resp.pipeline_id}")

        # 6. Close session (auto-cleanup plugins)
        client.CloseSession(
            cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_id)
        )
        logger.info("✓ Session closed (plugins auto-cleaned)")

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()


def example_2_local_plugin():
    """Example 2: Load a local plugin from filesystem."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Local Plugin Loading")
    logger.info("=" * 60)

    channel = grpc.insecure_channel("localhost:50051")
    client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        # Create session
        session_resp = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
        session_id = session_resp.session_id
        logger.info(f"✓ Session created: {session_id}")

        # Load local plugin
        manifest = PluginManifest(
            plugins={
                "local_nodes": LocalPluginConfig(
                    path="/path/to/local/plugin",
                    provides=[
                        "local_plugin.CustomNormalizer",
                        "local_plugin.CustomSelector",
                    ],
                )
            }
        )

        manifest_json = manifest.model_dump_json().encode()
        plugin_resp = client.LoadPlugins(
            cuvis_ai_core_pb2.LoadPluginsRequest(
                session_id=session_id,
                manifest=cuvis_ai_core_pb2.PluginManifest(config_bytes=manifest_json),
            )
        )

        logger.info(f"✓ Loaded local plugin: {list(plugin_resp.loaded_plugins)}")

        # Close session
        client.CloseSession(
            cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_id)
        )
        logger.info("✓ Session closed")

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()


def example_3_plugin_introspection():
    """Example 3: Query plugin information."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Plugin Introspection")
    logger.info("=" * 60)

    channel = grpc.insecure_channel("localhost:50051")
    client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        # Create session and load plugins
        session_resp = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        manifest = PluginManifest(
            plugins={
                "plugin_a": GitPluginConfig(
                    repo="git@example.com:org/plugin-a.git",
                    tag="v1.0.0",
                    provides=["plugin_a.NodeA", "plugin_a.NodeB"],
                ),
                "plugin_b": LocalPluginConfig(
                    path="/path/to/plugin-b", provides=["plugin_b.NodeC"]
                ),
            }
        )

        client.LoadPlugins(
            cuvis_ai_core_pb2.LoadPluginsRequest(
                session_id=session_id,
                manifest=cuvis_ai_core_pb2.PluginManifest(
                    config_bytes=manifest.model_dump_json().encode()
                ),
            )
        )

        # List all loaded plugins
        plugins_resp = client.ListLoadedPlugins(
            cuvis_ai_core_pb2.ListLoadedPluginsRequest(session_id=session_id)
        )

        logger.info(f"✓ Loaded plugins ({len(plugins_resp.plugins)}):")
        for plugin in plugins_resp.plugins:
            logger.info(f"\n  Plugin: {plugin.name}")
            logger.info(f"    Type: {plugin.type}")
            logger.info(f"    Source: {plugin.source}")
            if plugin.tag:
                logger.info(f"    Tag: {plugin.tag}")
            logger.info("    Provides:")
            for class_path in plugin.provides:
                logger.info(f"      • {class_path}")

        # Get specific plugin info
        plugin_info_resp = client.GetPluginInfo(
            cuvis_ai_core_pb2.GetPluginInfoRequest(
                session_id=session_id, plugin_name="plugin_a"
            )
        )

        logger.info("\n✓ Plugin 'plugin_a' details:")
        logger.info(f"    Type: {plugin_info_resp.plugin.type}")
        logger.info(f"    Source: {plugin_info_resp.plugin.source}")
        logger.info(f"    Provides: {list(plugin_info_resp.plugin.provides)}")

        # Close session
        client.CloseSession(
            cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_id)
        )

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()


def example_4_cache_management():
    """Example 4: Manage plugin cache."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Cache Management")
    logger.info("=" * 60)

    channel = grpc.insecure_channel("localhost:50051")
    client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        # Clear specific plugin cache
        cache_resp = client.ClearPluginCache(
            cuvis_ai_core_pb2.ClearPluginCacheRequest(plugin_name="my_plugin")
        )
        logger.info(f"✓ Cleared 'my_plugin' cache: {cache_resp.cleared_count} repo(s)")

        # Clear all plugin caches
        cache_resp = client.ClearPluginCache(
            cuvis_ai_core_pb2.ClearPluginCacheRequest(plugin_name="")
        )
        logger.info(f"✓ Cleared all caches: {cache_resp.cleared_count} repo(s)")

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()


def example_5_session_isolation():
    """Example 5: Demonstrate session isolation."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Session Isolation")
    logger.info("=" * 60)

    channel = grpc.insecure_channel("localhost:50051")
    client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        # Create Session A with plugin X
        session_a = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
        logger.info(f"✓ Created Session A: {session_a.session_id}")

        manifest_a = PluginManifest(
            plugins={
                "plugin_x": LocalPluginConfig(
                    path="/path/to/plugin-x", provides=["plugin_x.NodeX"]
                )
            }
        )

        client.LoadPlugins(
            cuvis_ai_core_pb2.LoadPluginsRequest(
                session_id=session_a.session_id,
                manifest=cuvis_ai_core_pb2.PluginManifest(
                    config_bytes=manifest_a.model_dump_json().encode()
                ),
            )
        )
        logger.info("✓ Loaded plugin_x in Session A")

        # Create Session B with plugin Y
        session_b = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
        logger.info(f"✓ Created Session B: {session_b.session_id}")

        manifest_b = PluginManifest(
            plugins={
                "plugin_y": LocalPluginConfig(
                    path="/path/to/plugin-y", provides=["plugin_y.NodeY"]
                )
            }
        )

        client.LoadPlugins(
            cuvis_ai_core_pb2.LoadPluginsRequest(
                session_id=session_b.session_id,
                manifest=cuvis_ai_core_pb2.PluginManifest(
                    config_bytes=manifest_b.model_dump_json().encode()
                ),
            )
        )
        logger.info("✓ Loaded plugin_y in Session B")

        # Verify Session A only sees plugin_x
        plugins_a = client.ListLoadedPlugins(
            cuvis_ai_core_pb2.ListLoadedPluginsRequest(session_id=session_a.session_id)
        )
        logger.info(f"✓ Session A plugins: {[p.name for p in plugins_a.plugins]}")

        # Verify Session B only sees plugin_y
        plugins_b = client.ListLoadedPlugins(
            cuvis_ai_core_pb2.ListLoadedPluginsRequest(session_id=session_b.session_id)
        )
        logger.info(f"✓ Session B plugins: {[p.name for p in plugins_b.plugins]}")

        # Close sessions
        client.CloseSession(
            cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_a.session_id)
        )
        client.CloseSession(
            cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_b.session_id)
        )
        logger.info("✓ Both sessions closed")

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()


def example_6_error_handling():
    """Example 6: Handle plugin loading errors gracefully."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Error Handling")
    logger.info("=" * 60)

    channel = grpc.insecure_channel("localhost:50051")
    client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        session_resp = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Try to load plugins with some invalid configurations
        manifest = PluginManifest(
            plugins={
                "valid_plugin": LocalPluginConfig(
                    path="/valid/path", provides=["valid.Node"]
                ),
                "invalid_plugin": GitPluginConfig(
                    repo="git@invalid-repo.git",  # Invalid repo
                    tag="v0.0.0",
                    provides=["invalid.Node"],
                ),
            }
        )

        plugin_resp = client.LoadPlugins(
            cuvis_ai_core_pb2.LoadPluginsRequest(
                session_id=session_id,
                manifest=cuvis_ai_core_pb2.PluginManifest(
                    config_bytes=manifest.model_dump_json().encode()
                ),
            )
        )

        # Check results
        if plugin_resp.loaded_plugins:
            logger.info(f"✓ Successfully loaded: {list(plugin_resp.loaded_plugins)}")

        if plugin_resp.failed_plugins:
            logger.warning("✗ Failed to load:")
            for name, error in plugin_resp.failed_plugins.items():
                logger.warning(f"    {name}: {error}")

        # Close session
        client.CloseSession(
            cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_id)
        )

    except grpc.RpcError as e:
        logger.error(f"gRPC Error: {e.code()}: {e.details()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        channel.close()


if __name__ == "__main__":
    logger.info("cuvis-ai-core gRPC Plugin Management Client Examples")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Make sure the cuvis-ai-core gRPC server is running:")
    logger.info("  uv run python -m cuvis_ai_core.grpc.production_server")
    logger.info("")

    # Run examples
    example_1_basic_plugin_loading()
    example_2_local_plugin()
    example_3_plugin_introspection()
    example_4_cache_management()
    example_5_session_isolation()
    example_6_error_handling()

    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)
