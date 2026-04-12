"""Custom Gradio UI tabs for integration visualizations.

Adds an "Integrations" tab to the OpenEnv web interface with:
- Integration status dashboard (which tools are connected)
- Cantera: live reaction rate bar chart
- ChemSep: VLE phase diagram
- DWSIM: thermodynamic property table
- Azure DT: twin status panel

Usage:
    Pass `build_integration_ui` as the `gradio_builder` parameter
    to `create_app()` in app.py.
"""

from __future__ import annotations

try:
    import gradio as gr
except ImportError:
    gr = None


def build_integration_ui(
    web_manager, action_fields, metadata, is_chat_env, title, quick_start_md
):
    """Build custom Gradio Blocks with integration visualizations.

    This is called by OpenEnv's web interface to create the "Custom" tab
    in the TabbedInterface alongside the default "Playground" tab.
    """
    if gr is None:
        return None

    with gr.Blocks(title="Methanol APC - Integrations") as demo:
        gr.Markdown("# Integrations Dashboard")
        gr.Markdown("Visualize external tool connections and cross-validation data.")

        with gr.Tabs():
            # Tab 1: Status
            with gr.Tab("Status"):
                _build_status_tab()

            # Tab 2: Reaction Kinetics (Cantera)
            with gr.Tab("Reaction Kinetics"):
                _build_kinetics_tab()

            # Tab 3: VLE Diagram (ChemSep)
            with gr.Tab("VLE Diagram"):
                _build_vle_tab()

            # Tab 4: Thermodynamics (DWSIM)
            with gr.Tab("Thermodynamics"):
                _build_thermo_tab()

            # Tab 5: Azure Digital Twin
            with gr.Tab("Azure Digital Twin"):
                _build_adt_tab()

    return demo


def _build_status_tab():
    """Integration connection status dashboard."""

    def check_status():
        results = []
        # Check each integration
        try:
            from methanol_apc_env.integrations import DWSIMIntegration
            d = DWSIMIntegration()
            results.append(("DWSIM", "Connected" if d.is_available else "Not installed (using internal SRK)", d.is_available))
        except Exception:
            results.append(("DWSIM", "Import error", False))

        try:
            from methanol_apc_env.integrations import CanteraIntegration
            c = CanteraIntegration()
            results.append(("Cantera", "Connected" if c.is_available else "Not installed (using internal LHHW)", c.is_available))
        except Exception:
            results.append(("Cantera", "Import error", False))

        try:
            from methanol_apc_env.integrations import ChemSepIntegration
            cs = ChemSepIntegration()
            results.append(("ChemSep", "Connected" if cs.is_available else "Not installed (using Antoine/Margules)", cs.is_available))
        except Exception:
            results.append(("ChemSep", "Import error", False))

        try:
            from methanol_apc_env.integrations import AzureDigitalTwinIntegration
            a = AzureDigitalTwinIntegration()
            results.append(("Azure DT", "Connected" if a.is_available else "Not configured (using internal sim)", a.is_available))
        except Exception:
            results.append(("Azure DT", "Import error", False))

        try:
            from methanol_apc_env.integrations import OPCUABridge
            o = OPCUABridge()
            results.append(("OPC-UA", "Available" if o.is_available else "asyncua not installed", o.is_available))
        except Exception:
            results.append(("OPC-UA", "Import error", False))

        try:
            from methanol_apc_env.integrations import StateStore
            s = StateStore()
            results.append(("Redis", "Connected" if s.is_available else "Not running (using in-memory)", s.is_available))
        except Exception:
            results.append(("Redis", "Import error", False))

        lines = []
        for name, status, ok in results:
            icon = "+" if ok else "-"
            lines.append(f"[{icon}] {name}: {status}")
        return "\n".join(lines)

    status_display = gr.Textbox(label="Integration Status", lines=8, interactive=False)
    refresh_btn = gr.Button("Refresh Status")
    refresh_btn.click(fn=check_status, outputs=status_display)
    # Auto-load on page open
    demo = gr.Blocks()
    status_display.value = check_status()


def _build_kinetics_tab():
    """Cantera reaction rate comparison chart."""

    def compute_rates(temperature, pressure):
        T_K = temperature + 273.15
        P_Pa = pressure * 1e5
        X = {"CO": 0.10, "H2": 0.65, "CO2": 0.05, "CH3OH": 0.01, "H2O": 0.01}

        try:
            from methanol_apc_env.integrations import CanteraIntegration
            cantera = CanteraIntegration()
            result = cantera.get_reaction_rates(T=T_K, P=P_Pa, X=X)
            return {
                "Source": result.source,
                "R1 (CO + 2H2 = CH3OH)": f"{result.rate_co_hydrogenation:.4e} mol/s",
                "R2 (CO2 + 3H2 = CH3OH + H2O)": f"{result.rate_co2_hydrogenation:.4e} mol/s",
                "R3 (CO2 + H2 = CO + H2O)": f"{result.rate_rwgs:.4e} mol/s",
                "Equilibrium K": f"{result.equilibrium_constant:.6f}",
            }
        except Exception as e:
            return {"Error": str(e)}

    gr.Markdown("### Reaction Rate Calculator")
    gr.Markdown("Compare reaction rates at different conditions using Cantera or internal LHHW model.")
    with gr.Row():
        temp_input = gr.Slider(200, 300, value=250, step=5, label="Temperature (C)")
        pres_input = gr.Slider(30, 120, value=80, step=5, label="Pressure (bar)")
    calc_btn = gr.Button("Calculate Rates")
    rate_output = gr.JSON(label="Reaction Rates")
    calc_btn.click(fn=compute_rates, inputs=[temp_input, pres_input], outputs=rate_output)


def _build_vle_tab():
    """ChemSep VLE phase diagram data."""

    def compute_vle(temperature, pressure, x_meoh):
        T_K = temperature + 273.15
        P_Pa = pressure * 1e5
        x = {"CH3OH": x_meoh, "H2O": 1.0 - x_meoh}

        try:
            from methanol_apc_env.integrations import ChemSepIntegration
            chemsep = ChemSepIntegration()
            results = chemsep.get_vle(T=T_K, P=P_Pa, compounds=["CH3OH", "H2O"], x=x)
            data = {}
            for r in results:
                data[r.compound] = {
                    "P_sat (bar)": f"{r.p_sat_bar:.4f}",
                    "K value": f"{r.K_value:.4f}",
                    "Activity coeff": f"{r.activity_coefficient:.4f}",
                    "Source": r.source,
                }
            return data
        except Exception as e:
            return {"Error": str(e)}

    def compute_bubble_point(pressure, x_meoh):
        P_Pa = pressure * 1e5
        x = {"CH3OH": x_meoh, "H2O": 1.0 - x_meoh}
        try:
            from methanol_apc_env.integrations import ChemSepIntegration
            chemsep = ChemSepIntegration()
            T_bp = chemsep.get_bubble_point(P=P_Pa, x=x)
            return f"Bubble point: {T_bp:.2f} K ({T_bp - 273.15:.1f} C)"
        except Exception as e:
            return f"Error: {e}"

    gr.Markdown("### Vapor-Liquid Equilibrium (Methanol-Water)")
    gr.Markdown("VLE data for the distillation column using ChemSep or internal Antoine/Margules model.")
    with gr.Row():
        vle_temp = gr.Slider(60, 110, value=65, step=1, label="Temperature (C)")
        vle_pres = gr.Slider(0.5, 5.0, value=1.013, step=0.1, label="Pressure (bar)")
        vle_x = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Methanol mole fraction (liquid)")
    vle_btn = gr.Button("Calculate VLE")
    vle_output = gr.JSON(label="VLE Results")
    vle_btn.click(fn=compute_vle, inputs=[vle_temp, vle_pres, vle_x], outputs=vle_output)

    bp_btn = gr.Button("Calculate Bubble Point")
    bp_output = gr.Textbox(label="Bubble Point", interactive=False)
    bp_btn.click(fn=compute_bubble_point, inputs=[vle_pres, vle_x], outputs=bp_output)


def _build_thermo_tab():
    """DWSIM thermodynamic property calculator."""

    def compute_thermo(temperature, pressure):
        T_K = temperature + 273.15
        P_Pa = pressure * 1e5
        try:
            from methanol_apc_env.integrations import DWSIMIntegration
            dwsim = DWSIMIntegration()
            thermo = dwsim.get_thermodynamic_properties(T=T_K, P=P_Pa)
            return {
                "Source": thermo.source,
                "Temperature (K)": f"{thermo.temperature:.2f}",
                "Pressure (Pa)": f"{thermo.pressure:.0f}",
                "Compressibility Z": f"{thermo.compressibility_factor:.6f}",
                "Heat Capacity Cp (J/mol/K)": f"{thermo.heat_capacity_cp:.2f}",
                "Fugacity Coefficients": {
                    k: f"{v:.6f}" for k, v in thermo.fugacity_coefficients.items()
                },
            }
        except Exception as e:
            return {"Error": str(e)}

    gr.Markdown("### Thermodynamic Properties (SRK EOS)")
    gr.Markdown("Calculate fugacity coefficients and compressibility using DWSIM or internal SRK model.")
    with gr.Row():
        thermo_temp = gr.Slider(200, 350, value=250, step=5, label="Temperature (C)")
        thermo_pres = gr.Slider(20, 120, value=80, step=5, label="Pressure (bar)")
    thermo_btn = gr.Button("Calculate Properties")
    thermo_output = gr.JSON(label="Thermodynamic Properties")
    thermo_btn.click(fn=compute_thermo, inputs=[thermo_temp, thermo_pres], outputs=thermo_output)


def _build_adt_tab():
    """Azure Digital Twin status and controls."""

    def check_adt_status():
        try:
            from methanol_apc_env.integrations import AzureDigitalTwinIntegration
            adt = AzureDigitalTwinIntegration()
            if not adt.is_available:
                return {
                    "Status": "Not connected",
                    "Mode": "Using internal reactor_sim.py",
                    "Setup": "Set AZURE_DIGITAL_TWINS_URL in .env and run 'az login'",
                    "Docs": "See docs/integrations/azure-digital-twins.md",
                }
            twins = adt.list_twins()
            return {
                "Status": "Connected",
                "Endpoint": "Azure Digital Twins",
                "Twins found": len(twins),
                "Twin IDs": [t.get("twin_id", "?") for t in twins],
            }
        except Exception as e:
            return {"Error": str(e)}

    def get_twin_state(twin_id):
        try:
            from methanol_apc_env.integrations import AzureDigitalTwinIntegration
            adt = AzureDigitalTwinIntegration()
            if not adt.is_available:
                return {"Error": "Azure DT not connected"}
            state = adt.get_twin_state(twin_id)
            return state if state else {"Error": f"Twin '{twin_id}' not found"}
        except Exception as e:
            return {"Error": str(e)}

    def export_dtdl():
        try:
            from methanol_apc_env.integrations import AzureDigitalTwinIntegration
            return AzureDigitalTwinIntegration.export_dtdl_model()
        except Exception as e:
            return {"Error": str(e)}

    gr.Markdown("### Azure Digital Twins")
    gr.Markdown("Connect to your company's Azure DT instance to use your own plant model.")

    status_btn = gr.Button("Check Connection")
    adt_status = gr.JSON(label="Connection Status")
    status_btn.click(fn=check_adt_status, outputs=adt_status)

    gr.Markdown("---")
    gr.Markdown("#### Read Twin State")
    twin_id_input = gr.Textbox(value="methanol-reactor-001", label="Twin ID")
    read_btn = gr.Button("Read Twin")
    twin_state_output = gr.JSON(label="Twin Properties")
    read_btn.click(fn=get_twin_state, inputs=twin_id_input, outputs=twin_state_output)

    gr.Markdown("---")
    gr.Markdown("#### DTDL Model Definition")
    gr.Markdown("Upload this JSON to Azure DT with: `az dt model create --dt-name <name> --models <json>`")
    dtdl_btn = gr.Button("Export DTDL Model")
    dtdl_output = gr.JSON(label="DTDL v2 Model")
    dtdl_btn.click(fn=export_dtdl, outputs=dtdl_output)
