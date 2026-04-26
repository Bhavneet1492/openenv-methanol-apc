"""
FastAPI application for the Methanol APC Environment.

Exposes the MethanolAPCEnvironment over HTTP and WebSocket endpoints,
compatible with the OpenEnv EnvClient.
"""

import os

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

MAX_CONCURRENT_ENVS = int(os.environ.get("MAX_CONCURRENT_ENVS", "1"))

app = create_app(
    MethanolAPCEnvironment,
    MethanolAPCAction,
    MethanolAPCObservation,
    env_name="methanol_apc",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

# Mount 3D Digital Twin visualisation as static files
from pathlib import Path as _Path
from starlette.staticfiles import StaticFiles as _StaticFiles

_static_dir = _Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/viz", _StaticFiles(directory=str(_static_dir), html=True), name="viz")


# ── Azure Digital Twins proxy endpoint for 3D visualization ──
@app.get("/adt/state")
async def adt_state():
    """Return merged plant state from all Azure DT twins.

    The 3D visualization polls this endpoint every 2s to show
    live twin data from the cloud. Returns {} if ADT not configured.
    """
    try:
        from integrations.azure_digital_twins import AzureDigitalTwinIntegration, TWIN_IDS
    except ImportError:
        try:
            from ..integrations.azure_digital_twins import AzureDigitalTwinIntegration, TWIN_IDS
        except ImportError:
            return {"error": "ADT module not available"}

    # Lazily init a shared ADT client (cached on app state)
    if not hasattr(app.state, "_adt"):
        app.state._adt = AzureDigitalTwinIntegration()
    adt = app.state._adt
    if not adt.is_available:
        return {"error": "ADT not connected"}

    # Read key twins and merge into a flat dict matching S fields in 3d-plant.html
    state = {}
    reactor = adt.get_twin_state(TWIN_IDS["reactor"])
    if reactor:
        state["temperature"] = reactor.get("temperature", 250)
        state["pressure"] = reactor.get("pressure", 80)
        state["catalyst_health"] = reactor.get("catalystHealth", 1.0)
        state["reaction_rate"] = reactor.get("reactionRate", 0)
        state["selectivity"] = reactor.get("selectivity", 0.995)
        state["bed_temps"] = [
            reactor.get("bed1Temp", 250), reactor.get("bed2Temp", 252),
            reactor.get("bed3Temp", 254), reactor.get("bed4Temp", 256),
        ]

    plant = adt.get_twin_state(TWIN_IDS["plant"])
    if plant:
        state["cumulative_profit"] = plant.get("cumulativeProfit", 0)
        state["methanol_produced"] = plant.get("totalMethanolProduced", 0)
        state["step_number"] = plant.get("stepNumber", 0)

    feed = adt.get_twin_state(TWIN_IDS["syngas_feed"])
    if feed:
        state["feed_rate_h2"] = feed.get("feedRateH2", 5)
        state["feed_rate_co"] = feed.get("feedRateCO", 2.5)
        state["h2_co_ratio"] = feed.get("h2CoRatio", 2.0)
        state["reformer_outlet_temp"] = feed.get("reformerOutletTemp", 850)

    comp = adt.get_twin_state(TWIN_IDS["compressor"])
    if comp:
        state["compressor_power"] = comp.get("power", 65)

    cool = adt.get_twin_state(TWIN_IDS["cooling_tower"])
    if cool:
        state["cooling_water_flow"] = cool.get("coolingWaterFlow", 40)

    recycle = adt.get_twin_state(TWIN_IDS["recycle_loop"])
    if recycle:
        state["recycle_ratio"] = recycle.get("recycleRatio", 3.5)
        state["purge_rate"] = recycle.get("purgeRate", 0)
        state["flare_valve"] = recycle.get("flareValve", 0)

    distill = adt.get_twin_state(TWIN_IDS["distillation"])
    if distill:
        state["product_purity"] = distill.get("productPurity", 0.9985)
        state["distillation_reflux"] = distill.get("refluxRatio", 3.0)
        state["reboiler_duty"] = distill.get("reboilerDuty", 50)

    return state


# ── Override /web/ to serve 3D Digital Twin instead of default OpenEnv UI ──
from starlette.responses import RedirectResponse as _RedirectResponse, FileResponse as _FileResponse
from starlette.routing import Route as _Route

_3d_plant_path = _Path(__file__).parent / "static" / "3d-plant.html"

# Remove any existing /web routes mounted by create_app
app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/web'))]

# Mount 3D plant at /web/
async def _serve_3d_plant(request):
    return _FileResponse(str(_3d_plant_path), media_type="text/html")

app.routes.insert(0, _Route("/web", endpoint=_serve_3d_plant, methods=["GET"]))
app.routes.insert(0, _Route("/web/", endpoint=_serve_3d_plant, methods=["GET"]))

# Root redirects to /web/
app.routes.insert(0, _Route("/", endpoint=lambda request: _RedirectResponse(url="/web/"), methods=["GET"]))


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run server`` or ``python -m methanol_apc_env.server.app``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
