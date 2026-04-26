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


# ── Model inference endpoint for Testing mode ──
_loaded_models = {}  # cache: model_id -> (model, tokenizer)

AVAILABLE_MODELS = {
    "unsloth": {"id": "glitchfilter/methanol-apc", "label": "Unsloth (Qwen2.5-3B)"},
    "trl": {"id": "glitchfilter/methanol-apc-grpo-qwen2.5-3b", "label": "TRL GRPO (Qwen2.5-3B)"},
}

SYSTEM_PROMPT = (
    "You control a methanol synthesis reactor. Output a JSON object with these fields: "
    "feed_rate_h2 (0-10 mol/s), feed_rate_co (0-5 mol/s), cooling_water_flow (0-100 L/min), "
    "compressor_power (0-100 kW). The reactor is exothermic: 240-260C is optimal, >300C = shutdown. "
    "Maintain H2/CO ratio near 2.0. Revenue is $0.74/kg methanol."
)

def _load_model(model_key):
    """Lazy-load a LoRA adapter. Cached after first load."""
    if model_key in _loaded_models:
        return _loaded_models[model_key]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        info = AVAILABLE_MODELS[model_key]
        adapter_id = info["id"]

        # Determine base model from adapter_config
        from huggingface_hub import hf_hub_download
        import json
        cfg_path = hf_hub_download(adapter_id, "adapter_config.json")
        with open(cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, adapter_id)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _loaded_models[model_key] = (model, tokenizer)
        return (model, tokenizer)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_key}: {e}")


def _obs_to_text(obs_dict):
    """Convert observation dict to compact sensor text for the model prompt."""
    parts = []
    for k in ["temperature", "pressure", "feed_rate_h2", "feed_rate_co", "h2_co_ratio",
              "cooling_water_flow", "catalyst_health", "reaction_rate", "methanol_produced",
              "cumulative_profit", "step_number", "max_steps"]:
        v = obs_dict.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    task = obs_dict.get("task_name", "")
    if task:
        parts.append(f"task={task}")
    return " ".join(parts)


@app.get("/model/list")
async def list_models():
    """Return available trained models."""
    return {"models": {k: v["label"] for k, v in AVAILABLE_MODELS.items()}}


@app.post("/model/step")
async def model_step(request):
    """Run one step using a trained model: load adapter, generate action, step env."""
    import json as _json
    body = await request.json()
    model_key = body.get("model", "trl")
    obs_dict = body.get("observation", {})

    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}"}

    try:
        model, tokenizer = _load_model(model_key)
    except Exception as e:
        return {"error": f"Model load failed: {str(e)[:200]}"}

    # Build prompt
    sensor_text = _obs_to_text(obs_dict)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sensors:\n{sensor_text}\n\nAction JSON:"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=150, temperature=0.3,
            do_sample=True, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse action from model response
    try:
        text = response.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        action_dict = _json.loads(text[start:end])
    except Exception:
        action_dict = {"feed_rate_h2": 3.0, "feed_rate_co": 1.5,
                       "cooling_water_flow": 60.0, "compressor_power": 50.0}

    return {
        "action": action_dict,
        "raw_response": response[:300],
        "model": AVAILABLE_MODELS[model_key]["label"],
    }


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
