"""
FastAPI application for the Methanol APC Environment.

Exposes the MethanolAPCEnvironment over HTTP and WebSocket endpoints,
compatible with the OpenEnv EnvClient.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with: uv sync"
    ) from e

try:
    from models import MethanolAPCAction, MethanolAPCObservation
except ImportError:
    from ..models import MethanolAPCAction, MethanolAPCObservation

try:
    from methanol_environment import MethanolAPCEnvironment
except ImportError:
    from .methanol_environment import MethanolAPCEnvironment

app = create_app(
    MethanolAPCEnvironment,
    MethanolAPCAction,
    MethanolAPCObservation,
    env_name="methanol_apc",
    max_concurrent_envs=1,
)

# Mount custom digital twin UI at /twin
import os
if os.getenv("ENABLE_WEB_INTERFACE", "").lower() in ("true", "1", "yes", ""):
    try:
        try:
            from custom_ui import create_custom_ui
        except ImportError:
            from .custom_ui import create_custom_ui
        
        import gradio as gr
        custom_ui = create_custom_ui()
        app = gr.mount_gradio_app(app, custom_ui, path="/twin")
    except Exception:
        pass  # custom UI is optional — don't break the API if it fails


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run server`` or ``python -m methanol_apc_env.server.app``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
