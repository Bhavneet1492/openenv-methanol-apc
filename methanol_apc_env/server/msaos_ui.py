"""MS-AOS: Methanol Synthesis Autonomous Operating System UI.

Professional dark-mode dashboard for the Methanol APC Environment
with 3D Digital Twin, Multi-Agent orchestrator, MCP tool feed,
reward tracker, and safety perimeter.

Designed for the OpenEnv hackathon - shows task selection, live
reward trajectory, baseline inference, and system health.
"""

from __future__ import annotations

import json
import math
import random
import time

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# ================================================================
# CUSTOM CSS - Dark mode, glassmorphism, industrial aesthetic
# ================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    background-color: #0b0e14 !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 100% !important;
}
.dark {
    --background-fill-primary: #0b0e14 !important;
    --background-fill-secondary: #111827 !important;
    --border-color-primary: #1e293b !important;
    --text-color: #e2e8f0 !important;
}
/* Header */
.header-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #00f5ff !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.header-sub {
    color: #64748b !important;
    font-size: 0.85rem !important;
}
/* Agent cards */
.agent-card {
    background: rgba(22, 27, 34, 0.8) !important;
    border-left: 4px solid #00f5ff !important;
    border-radius: 6px !important;
    backdrop-filter: blur(12px) !important;
    padding: 8px !important;
    margin-bottom: 4px !important;
}
.agent-card-warn {
    background: rgba(22, 27, 34, 0.8) !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 6px !important;
    backdrop-filter: blur(12px) !important;
}
.agent-card-super {
    background: rgba(22, 27, 34, 0.8) !important;
    border-left: 4px solid #a855f7 !important;
    border-radius: 6px !important;
    backdrop-filter: blur(12px) !important;
}
/* MCP log */
.mcp-log textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #000000 !important;
    color: #00f5ff !important;
    border: 1px solid #1e293b !important;
}
/* Safety button */
.safety-btn {
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    border: 2px solid #ef4444 !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}
/* Glass panels */
.glass-panel {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
}
/* Telemetry numbers */
.telemetry-num input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important;
    color: #00f5ff !important;
    text-align: center !important;
}
/* Section headers */
.section-header {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #1e293b !important;
    padding-bottom: 4px !important;
    margin-bottom: 8px !important;
}
"""


# ================================================================
# 3D PLANT VISUALIZATION (Plotly)
# ================================================================

def create_plant_3d(temperature=250, pressure=80, catalyst=1.0, rate=3.5):
    """Create 3D digital twin visualization of the methanol plant."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    # Plant nodes: x=process position, y=elevation, z=width
    nodes = {
        "Desulfurizer":   (0, 1, 0),
        "Reformer":       (1.5, 1.5, 0),
        "Heat Exchanger": (3, 1, 0),
        "Compressor":     (4.5, 1, 0),
        "Reactor Bed 1":  (6, 2.5, 0),
        "Reactor Bed 2":  (6, 2.0, 0),
        "Reactor Bed 3":  (6, 1.5, 0),
        "Reactor Bed 4":  (6, 1.0, 0),
        "Separator":      (7.5, 1, 0),
        "Distillation":   (9, 2, 0),
        "Product":        (10.5, 1, 0),
    }

    # Color based on temperature/health
    def node_color(name):
        if "Reactor" in name:
            t_norm = min(1, max(0, (temperature - 220) / 80))
            r = int(255 * t_norm)
            g = int(255 * (1 - t_norm * 0.7))
            b = int(100 * (1 - t_norm))
            return f"rgb({r},{g},{b})"
        if "Reformer" in name:
            return "rgb(245, 158, 11)"  # amber
        if "Product" in name:
            return "rgb(16, 185, 129)"  # emerald
        if "Distillation" in name:
            return "rgb(168, 85, 247)"  # purple
        return "rgb(0, 245, 255)"  # cyan

    names = list(nodes.keys())
    x = [nodes[n][0] for n in names]
    y = [nodes[n][1] for n in names]
    z = [nodes[n][2] for n in names]
    colors = [node_color(n) for n in names]
    sizes = [18 if "Reactor" in n else 14 for n in names]

    # Flow connections
    connections = [
        (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7),
        (7,8), (8,9), (9,10),
    ]

    fig = go.Figure()

    # Add flow lines
    for i, j in connections:
        color = "rgba(0,245,255,0.3)" if j < 4 else "rgba(59,130,246,0.4)"
        if j >= 9:
            color = "rgba(16,185,129,0.4)"
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # Recycle loop (Separator back to Compressor)
    fig.add_trace(go.Scatter3d(
        x=[x[8], x[8], x[3], x[3]],
        y=[y[8], 0.3, 0.3, y[3]],
        z=[0, 0, 0, 0],
        mode="lines",
        line=dict(color="rgba(245,158,11,0.4)", width=2, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=0.9,
                    line=dict(color="white", width=0.5)),
        text=names,
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0b0e14",
            camera=dict(eye=dict(x=1.5, y=0.8, z=0.6)),
        ),
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        margin=dict(l=0, r=0, t=0, b=0),
        height=350,
    )
    return fig


def create_reward_chart(rewards=None):
    """Create reward trajectory line chart."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    if rewards is None:
        rewards = []

    fig = go.Figure()
    if rewards:
        fig.add_trace(go.Scatter(
            x=list(range(len(rewards))),
            y=rewards,
            mode="lines",
            line=dict(color="#00f5ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,245,255,0.1)",
        ))
    fig.update_layout(
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#111827",
        font=dict(color="#94a3b8"),
        xaxis=dict(title="Step", gridcolor="#1e293b"),
        yaxis=dict(title="Reward", range=[0, 1], gridcolor="#1e293b"),
        margin=dict(l=40, r=10, t=10, b=30),
        height=180,
    )
    return fig


# ================================================================
# ENVIRONMENT INTERACTION
# ================================================================

class UIState:
    """Shared state for the UI session."""
    def __init__(self):
        self.env = None
        self.rewards = []
        self.step_count = 0
        self.mcp_log = ""
        self.task = "optimization"

    def get_env(self):
        if self.env is None:
            try:
                from methanol_apc_env.server.methanol_environment import MethanolAPCEnvironment
                self.env = MethanolAPCEnvironment()
            except Exception:
                return None
        return self.env

ui_state = UIState()


def reset_env(task_name):
    """Reset environment with selected task."""
    ui_state.task = task_name
    ui_state.rewards = []
    ui_state.step_count = 0
    ui_state.mcp_log = f"[SYSTEM] Reset environment with task: {task_name}\n"

    env = ui_state.get_env()
    if env is None:
        return (
            create_plant_3d(), create_reward_chart(),
            250.0, 80.0, 1.0, 0.0, 0.0, 0.0,
            "IDLE", "IDLE", "IDLE", "IDLE",
            ui_state.mcp_log,
            "Environment not available",
            0, "---",
        )

    try:
        from methanol_apc_env.models import MethanolAPCAction
        obs = env.reset(task_name=task_name, seed=42)
        ui_state.mcp_log += f"[RESET] T={obs.temperature:.1f}C P={obs.pressure:.1f}bar\n"
        return _build_outputs(obs)
    except Exception as e:
        return _default_outputs(f"Reset error: {e}")


def step_env(h2, co, cooling, compressor, purge, recycle, preheat,
             fuel, steam, reflux, reboiler, flare):
    """Execute one step with given action."""
    env = ui_state.get_env()
    if env is None:
        return _default_outputs("Environment not available")

    try:
        from methanol_apc_env.models import MethanolAPCAction
        action = MethanolAPCAction(
            feed_rate_h2=h2, feed_rate_co=co,
            cooling_water_flow=cooling, compressor_power=compressor,
            purge_valve_position=purge, recycle_ratio=recycle,
            feed_preheat_temp=preheat, reformer_fuel_gas=fuel,
            reformer_steam_flow=steam, distillation_reflux=reflux,
            reboiler_duty=reboiler, flare_valve=flare,
        )
        obs = env.step(action)
        ui_state.step_count += 1
        ui_state.rewards.append(obs.reward)

        # Simulate MCP tool calls
        if ui_state.step_count % 5 == 0:
            ui_state.mcp_log += f"[MCP] get_energy_pricing() -> gas=$0.002/mol elec=$0.08/kWh\n"
        if obs.temperature > 270:
            ui_state.mcp_log += f"[MCP] get_catalyst_status(T={obs.temperature:.0f}) -> WARNING: sintering risk\n"

        ui_state.mcp_log += (
            f"[STEP {ui_state.step_count}] "
            f"T={obs.temperature:.1f}C "
            f"rate={obs.reaction_rate:.3f} "
            f"reward={obs.reward:.3f}"
        )
        if obs.safety_warning:
            ui_state.mcp_log += f" !! {obs.safety_warning}"
        ui_state.mcp_log += "\n"

        return _build_outputs(obs)
    except Exception as e:
        return _default_outputs(f"Step error: {e}")


def _build_outputs(obs):
    """Convert observation to all UI outputs."""
    # Agent status
    def agent_status(name, temp, health):
        if temp > 280:
            return f"CRITICAL | T={temp:.0f}C"
        if temp > 265:
            return f"ACTIVE | T={temp:.0f}C"
        return f"NOMINAL | T={temp:.0f}C"

    smr = f"ACTIVE | S/C={getattr(obs, 'steam_to_carbon', 3.0):.1f}"
    syn = agent_status("Synthesis", obs.temperature, obs.catalyst_health)
    pur = f"ACTIVE | Purity={getattr(obs, 'product_purity', 0.998):.1%}"
    sup = f"COORDINATING | Score={obs.reward:.3f}"

    # Safety status
    safety = "NOMINAL"
    if obs.temperature > 290:
        safety = "!! CRITICAL - IMMINENT SHUTDOWN !!"
    elif obs.temperature > 280:
        safety = "WARNING - Catalyst sintering zone"
    elif obs.temperature > 270:
        safety = "CAUTION - Approaching limits"

    return (
        create_plant_3d(obs.temperature, obs.pressure, obs.catalyst_health, obs.reaction_rate),
        create_reward_chart(ui_state.rewards),
        obs.temperature, obs.pressure, obs.catalyst_health,
        obs.reaction_rate, obs.cumulative_profit, obs.methanol_produced,
        smr, syn, pur, sup,
        ui_state.mcp_log,
        safety,
        ui_state.step_count,
        f"T={obs.temperature:.1f}C | P={obs.pressure:.1f}bar | Cat={obs.catalyst_health:.0%} | Rate={obs.reaction_rate:.3f}mol/s",
    )


def _default_outputs(msg=""):
    return (
        create_plant_3d(), create_reward_chart(),
        250.0, 80.0, 1.0, 0.0, 0.0, 0.0,
        "OFFLINE", "OFFLINE", "OFFLINE", "OFFLINE",
        msg + "\n",
        "---",
        0, "---",
    )


# ================================================================
# BUILD THE GRADIO UI
# ================================================================

def build_msaos_ui(
    web_manager=None, action_fields=None, metadata=None,
    is_chat_env=None, title=None, quick_start_md=None,
):
    """Build the MS-AOS dashboard. Compatible with OpenEnv gradio_builder."""
    import gradio as gr

    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue="cyan",
            neutral_hue="slate",
        ),
        title="MS-AOS | Methanol Synthesis",
    ) as demo:

        # === HEADER ===
        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("""
                <div style="padding:8px 0;">
                    <span style="font-family:'Inter',sans-serif;font-size:1.3rem;font-weight:700;color:#00f5ff;letter-spacing:2px;">
                        MS-AOS
                    </span>
                    <span style="color:#475569;font-size:0.9rem;margin-left:12px;">
                        Methanol Synthesis Autonomous Operating System
                    </span>
                </div>
                """)
            with gr.Column(scale=1):
                task_select = gr.Dropdown(
                    choices=[
                        ("L1: Setpoint Tracking (Easy)", "optimization"),
                        ("L2: Cold Start (Medium)", "startup"),
                        ("L3: Disturbance Rejection (Medium)", "disturbance_rejection"),
                        ("L4: Emergency Recovery (Hard)", "emergency_recovery"),
                        ("L5: Aged Catalyst (Hard)", "aged_catalyst"),
                        ("L6: Multi-Disturbance (Expert)", "multi_disturbance"),
                    ],
                    value="optimization",
                    label="Task",
                    interactive=True,
                )

        # === MAIN 3-COLUMN LAYOUT ===
        with gr.Row():

            # --- LEFT: Agent Orchestrator + MCP ---
            with gr.Column(scale=1, min_width=280):
                gr.HTML('<div class="section-header">MULTI-AGENT ORCHESTRATOR</div>')

                smr_status = gr.Textbox(label="Reformer Agent", value="IDLE", interactive=False, elem_classes="agent-card")
                syn_status = gr.Textbox(label="Synthesis Agent", value="IDLE", interactive=False, elem_classes="agent-card")
                pur_status = gr.Textbox(label="Purification Agent", value="IDLE", interactive=False, elem_classes="agent-card-warn")
                sup_status = gr.Textbox(label="Supervisory Agent", value="IDLE", interactive=False, elem_classes="agent-card-super")

                gr.HTML('<div class="section-header" style="margin-top:12px;">MCP TOOL-USE LOG</div>')
                mcp_feed = gr.TextArea(
                    value="> Waiting for environment reset...",
                    lines=10,
                    interactive=False,
                    elem_classes="mcp-log",
                    show_label=False,
                )

            # --- CENTER: Digital Twin + Telemetry ---
            with gr.Column(scale=3, min_width=500):
                gr.HTML('<div class="section-header">3D DIGITAL TWIN</div>')
                plant_plot = gr.Plot(value=create_plant_3d(), show_label=False)

                gr.HTML('<div class="section-header">REWARD TRAJECTORY</div>')
                reward_plot = gr.Plot(value=create_reward_chart(), show_label=False)

                gr.HTML('<div class="section-header">PROCESS TELEMETRY</div>')
                with gr.Row():
                    tel_temp = gr.Number(label="Temp (C)", value=250.0, interactive=False, elem_classes="telemetry-num")
                    tel_pres = gr.Number(label="Press (bar)", value=80.0, interactive=False, elem_classes="telemetry-num")
                    tel_cat = gr.Number(label="Catalyst", value=1.0, interactive=False, elem_classes="telemetry-num")
                    tel_rate = gr.Number(label="Rate (mol/s)", value=0.0, interactive=False, elem_classes="telemetry-num")
                    tel_profit = gr.Number(label="Profit ($)", value=0.0, interactive=False, elem_classes="telemetry-num")
                    tel_meoh = gr.Number(label="MeOH (kg)", value=0.0, interactive=False, elem_classes="telemetry-num")

            # --- RIGHT: Controls + Safety ---
            with gr.Column(scale=1, min_width=280):
                gr.HTML('<div class="section-header">CONTROL INPUTS</div>')

                h2_in = gr.Slider(0, 10, value=5.0, step=0.1, label="H2 Feed (mol/s)")
                co_in = gr.Slider(0, 5, value=2.5, step=0.1, label="CO Feed (mol/s)")
                cool_in = gr.Slider(0, 100, value=40, step=1, label="Cooling (L/min)")
                comp_in = gr.Slider(0, 100, value=65, step=1, label="Compressor (kW)")

                with gr.Accordion("Advanced Controls", open=False):
                    purge_in = gr.Slider(0, 100, value=2, step=0.5, label="Purge Valve (%)")
                    recycle_in = gr.Slider(0, 8, value=3.5, step=0.1, label="Recycle Ratio")
                    preheat_in = gr.Slider(0, 300, value=200, step=5, label="Preheat (C)")
                    fuel_in = gr.Slider(0, 20, value=5, step=0.5, label="Reformer Fuel")
                    steam_in = gr.Slider(0, 50, value=15, step=1, label="Reformer Steam")
                    reflux_in = gr.Slider(0, 10, value=3, step=0.1, label="Reflux Ratio")
                    reboiler_in = gr.Slider(0, 200, value=50, step=5, label="Reboiler (kW)")
                    flare_in = gr.Slider(0, 100, value=0, step=1, label="Flare Valve (%)")

                with gr.Row():
                    reset_btn = gr.Button("RESET", variant="secondary")
                    step_btn = gr.Button("STEP", variant="primary")

                gr.HTML('<div class="section-header" style="margin-top:12px;">SAFETY PERIMETER</div>')
                safety_display = gr.Textbox(label="Status", value="---", interactive=False)
                step_counter = gr.Number(label="Step", value=0, interactive=False)
                status_bar = gr.Textbox(label="Summary", value="---", interactive=False)
                override_btn = gr.Button("EMERGENCY PID OVERRIDE", variant="stop", elem_classes="safety-btn")

        # === FOOTER ===
        gr.HTML("""
        <div style="text-align:center;padding:12px 0;color:#475569;font-size:0.75rem;border-top:1px solid #1e293b;margin-top:16px;">
            MS-AOS v0.2 | OpenEnv by Meta | PyTorch | MCP Protocol | 
            <a href="https://bhavneet1492.github.io/openenv-methanol-apc/" style="color:#00f5ff;">Documentation</a>
        </div>
        """)

        # === WIRING ===
        all_outputs = [
            plant_plot, reward_plot,
            tel_temp, tel_pres, tel_cat, tel_rate, tel_profit, tel_meoh,
            smr_status, syn_status, pur_status, sup_status,
            mcp_feed,
            safety_display, step_counter, status_bar,
        ]

        reset_btn.click(
            fn=reset_env,
            inputs=[task_select],
            outputs=all_outputs,
        )

        step_btn.click(
            fn=step_env,
            inputs=[h2_in, co_in, cool_in, comp_in, purge_in, recycle_in,
                    preheat_in, fuel_in, steam_in, reflux_in, reboiler_in, flare_in],
            outputs=all_outputs,
        )

        override_btn.click(
            fn=lambda: reset_env("optimization"),
            outputs=all_outputs,
        )

    return demo


# Allow standalone testing
if __name__ == "__main__":
    demo = build_msaos_ui()
    if demo:
        demo.launch(server_name="0.0.0.0", server_port=7860)
