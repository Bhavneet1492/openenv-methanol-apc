"""
Custom Gradio UI for the Methanol APC Environment.

Replaces the default OpenEnv web interface with a professional
industrial digital twin control panel.
"""

import json
import math
from typing import Dict, List, Optional

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from methanol_environment import MethanolAPCEnvironment
    from reactor_sim import (
        Ea_R1, Ea_R2, Ea_R3, k0_R1, k0_R2, k0_R3,
        DELTA_H_R1_298, DELTA_H_R2_298, DELTA_H_R3_298,
        U_BASE, A_HX, M_REACTOR, CP_REACTOR, BED_POROSITY,
        PELLET_DIAMETER, BED_LENGTH, ETA,
        T_OPTIMAL_MAX, T_SINTERING, EMERGENCY_SHUTDOWN_TEMP,
        METHANOL_PRICE, SYNGAS_PRICE, ELECTRICITY_PRICE, COOLING_WATER_PRICE,
        VALVE_RATE_LIMIT, COOLING_RATE_LIMIT, COMPRESSOR_RATE_LIMIT,
    )
except ImportError:
    from .methanol_environment import MethanolAPCEnvironment
    from .reactor_sim import (
        Ea_R1, Ea_R2, Ea_R3, k0_R1, k0_R2, k0_R3,
        DELTA_H_R1_298, DELTA_H_R2_298, DELTA_H_R3_298,
        U_BASE, A_HX, M_REACTOR, CP_REACTOR, BED_POROSITY,
        PELLET_DIAMETER, BED_LENGTH, ETA,
        T_OPTIMAL_MAX, T_SINTERING, EMERGENCY_SHUTDOWN_TEMP,
        METHANOL_PRICE, SYNGAS_PRICE, ELECTRICITY_PRICE, COOLING_WATER_PRICE,
        VALVE_RATE_LIMIT, COOLING_RATE_LIMIT, COMPRESSOR_RATE_LIMIT,
    )

try:
    from models import MethanolAPCAction
except ImportError:
    from ..models import MethanolAPCAction

try:
    from tasks import TASKS
except ImportError:
    from .tasks import TASKS

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
_env = MethanolAPCEnvironment()
_history: List[Dict] = []


def _obs_to_dict(obs) -> Dict:
    return {
        "step": obs.step_number,
        "temperature": obs.temperature,
        "pressure": obs.pressure,
        "reaction_rate": obs.reaction_rate,
        "methanol_produced": obs.methanol_produced,
        "catalyst_health": obs.catalyst_health,
        "cumulative_profit": obs.cumulative_profit,
        "profit_this_step": obs.profit_this_step,
        "h2_co_ratio": obs.h2_co_ratio,
        "feed_rate_h2": obs.feed_rate_h2,
        "feed_rate_co": obs.feed_rate_co,
        "cooling_water_flow": obs.cooling_water_flow,
        "cooling_water_temp": obs.cooling_water_temp,
        "compressor_power": getattr(obs, "compressor_power", 0),
        "temperature_trend": obs.temperature_trend,
        "safety_warning": obs.safety_warning,
        "reward": obs.reward,
        "done": obs.done,
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _build_main_chart(history: List[Dict]) -> go.Figure:
    if not history:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.5, 0.3, 0.2],
                           subplot_titles=("Temperature & Rate", "Profit", "Catalyst"))
        fig.update_layout(height=500, margin=dict(l=50, r=30, t=40, b=30),
                         template="plotly_dark")
        return fig

    steps = [h["step"] for h in history]
    temps = [h["temperature"] for h in history]
    rates = [h["reaction_rate"] for h in history]
    profits = [h["cumulative_profit"] for h in history]
    catalysts = [h["catalyst_health"] * 100 for h in history]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("Temperature (°C) & Rate (mol/s)", "Cumulative Profit ($)", "Catalyst Health (%)"),
        specs=[[{"secondary_y": True}], [{}], [{}]],
    )

    # Temperature
    fig.add_trace(go.Scatter(x=steps, y=temps, name="Temperature",
                            line=dict(color="#FF6B6B", width=2)), row=1, col=1)
    # Shutdown line
    fig.add_hline(y=EMERGENCY_SHUTDOWN_TEMP, line_dash="dash", line_color="red",
                  annotation_text="SHUTDOWN", row=1, col=1)
    fig.add_hline(y=T_OPTIMAL_MAX, line_dash="dash", line_color="yellow",
                  annotation_text="WARNING", row=1, col=1)
    fig.add_hline(y=250, line_dash="dot", line_color="green",
                  annotation_text="TARGET", row=1, col=1)
    # Rate on secondary y
    fig.add_trace(go.Scatter(x=steps, y=rates, name="Rate",
                            line=dict(color="#4ECDC4", width=2, dash="dot")),
                 row=1, col=1, secondary_y=True)

    # Profit
    fig.add_trace(go.Scatter(x=steps, y=profits, name="Profit",
                            fill="tozeroy", line=dict(color="#45B7D1", width=2)),
                 row=2, col=1)

    # Catalyst
    fig.add_trace(go.Scatter(x=steps, y=catalysts, name="Catalyst",
                            fill="tozeroy", line=dict(color="#96CEB4", width=2)),
                 row=3, col=1)

    fig.update_layout(
        height=500, margin=dict(l=50, r=30, t=40, b=30),
        template="plotly_dark", showlegend=False,
        font=dict(size=11),
    )
    fig.update_yaxes(range=[100, 320], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


def _build_economics_chart(history: List[Dict]) -> go.Figure:
    if not history:
        fig = go.Figure()
        fig.update_layout(height=300, template="plotly_dark",
                         margin=dict(l=50, r=30, t=40, b=30))
        return fig

    steps = [h["step"] for h in history]
    step_profits = [h["profit_this_step"] for h in history]
    cum_profits = [h["cumulative_profit"] for h in history]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=steps, y=step_profits, name="Step Profit",
                        marker_color="#45B7D1", opacity=0.6))
    fig.add_trace(go.Scatter(x=steps, y=cum_profits, name="Cumulative",
                            line=dict(color="#FF6B6B", width=2)),
                 secondary_y=True)
    fig.update_layout(height=300, template="plotly_dark",
                     margin=dict(l=50, r=30, t=40, b=30),
                     title="Economics", font=dict(size=11))
    return fig


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------
def do_reset(task_name):
    global _history
    _history = []
    obs = _env.reset(task_name=task_name)
    d = _obs_to_dict(obs)
    _history.append(d)
    status = _format_status(d)
    return (
        _build_main_chart(_history),
        _build_economics_chart(_history),
        status,
        f"Step 0/{obs.max_steps} | Task: {task_name}",
        _format_economics(d),
    )


def do_step(h2, co, cooling, compressor):
    action = MethanolAPCAction(
        feed_rate_h2=h2, feed_rate_co=co,
        cooling_water_flow=cooling, compressor_power=compressor,
    )
    obs = _env.step(action)
    d = _obs_to_dict(obs)
    _history.append(d)
    status = _format_status(d)
    score_text = ""
    if obs.done:
        score = _env.get_final_score()
        score_text = f"\n\n**EPISODE DONE — Score: {score:.4f}**"
    return (
        _build_main_chart(_history),
        _build_economics_chart(_history),
        status + score_text,
        f"Step {obs.step_number}/{obs.max_steps} | {'DONE' if obs.done else 'Running'}",
        _format_economics(d),
    )


def do_multi_step(h2, co, cooling, compressor, n_steps):
    n = int(n_steps)
    for _ in range(n):
        result = do_step(h2, co, cooling, compressor)
        if _history[-1].get("done", False):
            break
    return result


def _format_status(d: Dict) -> str:
    T = d["temperature"]
    if T > 290:
        warn = "🔴 **CRITICAL** — Approaching shutdown!"
    elif T > 270:
        warn = "🟡 **WARNING** — Above optimal range"
    else:
        warn = "🟢 **NORMAL**"

    return f"""
{warn}

| Metric | Value |
|--------|-------|
| 🌡️ Temperature | **{T:.1f}°C** ({d['temperature_trend']:+.1f}°C/step) |
| ⚡ Pressure | {d['pressure']:.1f} bar |
| ⚗️ Reaction Rate | {d['reaction_rate']:.4f} mol/s |
| 📊 H₂/CO Ratio | {d['h2_co_ratio']:.2f} (target: 2.0) |
| 🧬 Catalyst | {d['catalyst_health']:.2%} |
| ⚗️ Methanol | {d['methanol_produced']:.1f} kg |
| 💰 Total Profit | ${d['cumulative_profit']:.2f} |
| 📈 Reward | {d['reward']:.4f} |
"""


def _format_economics(d: Dict) -> str:
    return f"""
| Item | Value |
|------|-------|
| Step Profit | ${d['profit_this_step']:.4f} |
| Cumulative | ${d['cumulative_profit']:.2f} |
| MeOH Produced | {d['methanol_produced']:.1f} kg |
| MeOH Price | ${METHANOL_PRICE}/kg (${METHANOL_PRICE*1000:.0f}/MT) |
| Syngas Price | ${SYNGAS_PRICE}/mol |
| Electricity | ${ELECTRICITY_PRICE}/kWh |
"""


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------
def create_custom_ui() -> gr.Blocks:
    task_names = sorted(TASKS.keys())

    with gr.Blocks(
        title="Methanol APC — Digital Twin",
    ) as ui:
        gr.Markdown("# 🧪 Methanol APC — Industrial Digital Twin Control Panel")
        gr.Markdown("*ICI Low-Pressure Process | Cu/ZnO/Al₂O₃ Catalyst | 3-Reaction Model*")

        # --- Top bar: Config + Task ---
        with gr.Row():
            task_dd = gr.Dropdown(choices=task_names, value="startup", label="Task", scale=2)
            step_info = gr.Textbox(value="Step 0/50 | Task: startup", label="Status", interactive=False, scale=3)
            reset_btn = gr.Button("🔄 Reset", variant="primary", scale=1)

        # --- Main two-column layout ---
        with gr.Row():
            # LEFT: Controls + Status (40%)
            with gr.Column(scale=2):
                gr.Markdown("### Agent Controls")
                h2_sl = gr.Slider(0, 10, value=4.0, step=0.5, label="H₂ Feed Rate (mol/s)")
                co_sl = gr.Slider(0, 5, value=2.0, step=0.25, label="CO Feed Rate (mol/s)")
                cool_sl = gr.Slider(0, 100, value=60, step=5, label="Cooling Water Flow (L/min)")
                comp_sl = gr.Slider(0, 100, value=50, step=5, label="Compressor Power (kW)")

                with gr.Row():
                    step_btn = gr.Button("▶ Step", variant="primary")
                    step5_btn = gr.Button("▶▶ 5 Steps")
                    step10_btn = gr.Button("▶▶▶ 10 Steps")

                gr.Markdown("### Reactor Status")
                status_md = gr.Markdown("Press Reset to start", elem_classes=["status-panel"])

                gr.Markdown("### Economics")
                econ_md = gr.Markdown("—")

            # RIGHT: Charts (60%)
            with gr.Column(scale=3):
                with gr.Tab("Temperature + Rate + Catalyst"):
                    main_chart = gr.Plot(label="Process Variables")
                with gr.Tab("Economics"):
                    econ_chart = gr.Plot(label="Economics")

        # --- Disturbance + Advanced ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Disturbance Injection")
                dist_cool = gr.Slider(10, 50, value=25, step=1, label="Cooling Water Temp Override (°C)")
                dist_btn = gr.Button("💥 Inject Disturbance")
            with gr.Column():
                gr.Markdown("### Advanced Setpoints")
                tgt_temp = gr.Slider(200, 280, value=250, step=5, label="Target Temperature (°C)")
                tgt_ratio = gr.Slider(1.0, 4.0, value=2.0, step=0.1, label="H₂/CO Ratio Target")

        # --- Physics Parameters (Accordion) ---
        with gr.Accordion("⚙️ Physics Parameters (click to expand)", open=False):
            with gr.Tab("Kinetics"):
                with gr.Row():
                    gr.Slider(30000, 120000, value=Ea_R1, step=1000, label="Ea R1 (J/mol)", interactive=True)
                    gr.Slider(30000, 120000, value=Ea_R2, step=1000, label="Ea R2 (J/mol)", interactive=True)
                    gr.Slider(50000, 150000, value=Ea_R3, step=1000, label="Ea R3 (J/mol)", interactive=True)
                with gr.Row():
                    gr.Slider(0.3, 1.0, value=ETA, step=0.05, label="η (effectiveness)", interactive=True)
            with gr.Tab("Thermodynamics"):
                with gr.Row():
                    gr.Number(value=DELTA_H_R1_298, label="ΔH R1 (J/mol)", interactive=True)
                    gr.Number(value=DELTA_H_R2_298, label="ΔH R2 (J/mol)", interactive=True)
                    gr.Number(value=DELTA_H_R3_298, label="ΔH R3 (J/mol)", interactive=True)
            with gr.Tab("Heat Transfer"):
                with gr.Row():
                    gr.Slider(100, 500, value=U_BASE, step=10, label="U base (W/m²K)", interactive=True)
                    gr.Slider(2, 50, value=A_HX, step=1, label="A_hx (m²)", interactive=True)
            with gr.Tab("Reactor"):
                with gr.Row():
                    gr.Slider(1000, 20000, value=M_REACTOR, step=500, label="Mass (kg)", interactive=True)
                    gr.Slider(3, 12, value=BED_LENGTH, step=0.5, label="Bed Length (m)", interactive=True)
                    gr.Slider(0.3, 0.6, value=BED_POROSITY, step=0.05, label="Porosity", interactive=True)
            with gr.Tab("Economics"):
                with gr.Row():
                    gr.Slider(0.30, 1.50, value=METHANOL_PRICE, step=0.01, label="MeOH Price ($/kg)", interactive=True)
                    gr.Slider(0.001, 0.01, value=SYNGAS_PRICE, step=0.001, label="Syngas ($/mol)", interactive=True)
                    gr.Slider(0.02, 0.20, value=ELECTRICITY_PRICE, step=0.01, label="Elec ($/kWh)", interactive=True)
            with gr.Tab("Safety"):
                with gr.Row():
                    gr.Slider(280, 350, value=EMERGENCY_SHUTDOWN_TEMP, step=5, label="Shutdown Temp (°C)", interactive=True)
                    gr.Slider(250, 300, value=T_OPTIMAL_MAX, step=5, label="T Optimal Max (°C)", interactive=True)
                    gr.Slider(280, 350, value=T_SINTERING, step=5, label="T Sintering (°C)", interactive=True)

        # --- Step Log (Collapsed) ---
        with gr.Accordion("📋 Raw Step Log", open=False):
            gr.Markdown("*Step data available after running episodes*")

        # --- Wire events ---
        outputs = [main_chart, econ_chart, status_md, step_info, econ_md]

        reset_btn.click(do_reset, inputs=[task_dd], outputs=outputs)
        step_btn.click(do_step, inputs=[h2_sl, co_sl, cool_sl, comp_sl], outputs=outputs)
        step5_btn.click(do_multi_step, inputs=[h2_sl, co_sl, cool_sl, comp_sl, gr.Number(value=5, visible=False)], outputs=outputs)
        step10_btn.click(do_multi_step, inputs=[h2_sl, co_sl, cool_sl, comp_sl, gr.Number(value=10, visible=False)], outputs=outputs)

    return ui
