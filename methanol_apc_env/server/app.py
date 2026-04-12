"""
FastAPI application for the Methanol APC Environment.

Exposes the MethanolAPCEnvironment over HTTP and WebSocket endpoints,
compatible with the OpenEnv EnvClient. Mounts MS-AOS dashboard as
the sole web UI at /web (no default Playground tab).
"""

import os

try:
    from openenv.core.env_server.http_server import create_app, create_fastapi_app
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

# Check if web interface is enabled (HF Spaces sets this)
_enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")

if _enable_web:
    # Build the API-only FastAPI app (no default Gradio)
    app = create_fastapi_app(
        MethanolAPCEnvironment,
        MethanolAPCAction,
        MethanolAPCObservation,
        max_concurrent_envs=1,
    )

    # Mount MS-AOS as the SOLE UI at /web
    try:
        import gradio as gr
        try:
            from msaos_ui import build_msaos_ui
        except ImportError:
            from .msaos_ui import build_msaos_ui

        from openenv.core.env_server.web_interface import WebManager
        from openenv.core.env_server.gradio_ui import _extract_action_fields
        from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_THEME

        # Create web manager for the /web/* endpoints
        web_manager = WebManager(MethanolAPCEnvironment, MethanolAPCAction, MethanolAPCObservation)

        # Add web endpoints that the UI needs
        from fastapi import HTTPException, status
        from typing import Dict, Any

        @app.post("/web/reset")
        async def web_reset(request: Dict[str, Any]):
            return await web_manager.reset_environment(request)

        @app.post("/web/step")
        async def web_step(request: Dict[str, Any]):
            action_data = request.get("action", {})
            return await web_manager.step_environment(action_data)

        @app.get("/web/state")
        async def web_state():
            try:
                return web_manager.get_state()
            except RuntimeError as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        # Build and mount MS-AOS as the only Gradio app
        msaos_blocks = build_msaos_ui()
        app = gr.mount_gradio_app(app, msaos_blocks, path="/web")

        # Redirect root to /web
        from starlette.responses import RedirectResponse

        @app.get("/")
        async def root_redirect():
            return RedirectResponse(url="/web/")

    except Exception as e:
        # Fallback: if MS-AOS fails, use default OpenEnv UI
        import sys
        print(f"MS-AOS UI failed: {e}, falling back to default", file=sys.stderr)
        app = create_app(
            MethanolAPCEnvironment,
            MethanolAPCAction,
            MethanolAPCObservation,
            env_name="methanol_apc",
            max_concurrent_envs=1,
        )
else:
    # No web interface - just API
    app = create_app(
        MethanolAPCEnvironment,
        MethanolAPCAction,
        MethanolAPCObservation,
        env_name="methanol_apc",
        max_concurrent_envs=1,
    )


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run server`` or ``python -m methanol_apc_env.server.app``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
