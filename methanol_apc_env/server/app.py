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

# Mount 3D Digital Twin visualisation as static files
from pathlib import Path as _Path
from starlette.staticfiles import StaticFiles as _StaticFiles

_static_dir = _Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/viz", _StaticFiles(directory=str(_static_dir), html=True), name="viz")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run server`` or ``python -m methanol_apc_env.server.app``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
