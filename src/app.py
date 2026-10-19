"""
Auto-Catalyst 2.0: 2026 SOTA Industrial Web Dashboard & Control Center
Built with Streamlit and Plotly.

Comprehensive Capabilities:
  - Tab 1: AI Discovery & BoTorch Bayesian Active Learning (Log q-NEHVI Pareto Frontier)
  - Tab 2: 3D Crystallography & Steric Molecular Confinement (FAU / MFI / BEA)
  - Tab 3: Petrochemical 10-Lump Kinetic Digital Twin (Non-Isothermal Riser)
  - Tab 4: FCC Regenerator Combustion & Multi-Cycle Hydrothermal Aging
  - Tab 5: Real-Time GC-MS Spectral Peak Deconvolution
  - Tab 6: Refinery Techno-Economics & Scope 1 CO2 Emissions Accounting
  - Tab 7: Executive LLM LabAgent Audit Report Viewer
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add source paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.extend([
    str(ROOT_DIR / "src"),
    str(ROOT_DIR / "src" / "models"),
    str(ROOT_DIR / "src" / "automation"),
    str(ROOT_DIR / "src" / "simulation"),
    str(ROOT_DIR / "src" / "analysis"),
])

from bayesian_active_optimizer import BayesianActiveCatalystOptimizer
from synthesis_planner import CatalystSynthesisPlanner
from petrochemical_10lump_twin import PetrochemicalFCC10LumpModel, BreckFlanigenZeolitePhysics
from industrial_10lump_conserved import IndustrialPetrochemical10LumpTwin, IndustrialBreckFlanigenManifold
from instrument_data_ingestion import RealInstrumentDataIngestionEngine
from regenerator_aging_loop import FCCRegeneratorAgingLoop
from gcms_analyzer import GCMSProductAnalyzer
from techno_economics import FCCTechnoEconomicAnalyzer
from zeolite_crystallography import ZeoliteCrystallographyEngine

st.set_page_config(
    page_title="Auto-Catalyst 2.0 | SOTA Petrochemical Platform",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Auto-Catalyst 2.0: Autonomous AI Petrochemical Discovery Platform")
st.markdown(
    "**End-to-End Closed-Loop Discovery, Crystallography, 10-Lump Non-Isothermal Kinetics, Regenerator Aging, and Refinery Economics.**"
)

tabs = st.tabs([
    "🎯 1. Bayesian Active Learning",
    "💎 2. 3D Crystallography",
    "⚙️ 3. 10-Lump Kinetic Twin",
    "🔥 4. Regenerator Aging",
    "🔬 5. GC-MS Analytics",
    "🧪 6. XRD / BET QC Gates",
    "💰 7. Economics & CO2",
    "📋 8. Executive LLM Audit"
])

# ==============================================================================
# TAB 1: Bayesian Active Learning
# ==============================================================================
with tabs[0]:
    st.header("BoTorch Bayesian Active Learning & Multi-Objective Pareto Frontier")
    st.markdown(r"Uses Gaussian Process Epistemic Uncertainty ($\mu, \sigma$) and Log $q$-NEHVI Acquisition to propose high-value catalyst batches.")

    mf_store_file = ROOT_DIR / "data" / "multi_fidelity_active_learning_history.json"
    legacy_store_file = ROOT_DIR / "data" / "active_learning_history.json"
    store_file = mf_store_file if mf_store_file.exists() else legacy_store_file

    if store_file.exists():
        with open(store_file, "r", encoding="utf-8") as f:
            hist_data = json.load(f)
        
        flat_hist = []
        for r in hist_data:
            p = r["parameters"]
            m = r["observed_metrics"]
            ucs_val = float(p.get("usy_ucs_angstrom", 24.26))
            # Strict Breck-Flanigen Framework Si/Al calculation
            n_al = max(0.1, (ucs_val - 24.191) / 0.00868)
            si_al = (192.0 - n_al) / n_al

            flat_hist.append({
                "Experiment_ID": r["experiment_id"],
                "Cycle": r.get("cycle", 0),
                "Fidelity_Tier": "High-Fidelity ACE Pilot (s=1.0)" if float(r.get("fidelity_s", 0.0)) >= 0.5 else "Low-Fidelity ODE Twin (s=0.0)",
                "Data_Source": r.get("data_source", "10-Lump ODE Simulation"),
                "ZSM5_pct": round(float(p["zsm5_fraction"]) * 100.0, 1),
                "USY_pct": round((1.0 - float(p["zsm5_fraction"])) * 100.0, 1),
                "UCS_Angstrom": round(ucs_val, 3),
                "Framework_Si_Al": round(si_al, 1),
                "Rare_Earth_wt": round(float(p.get("rare_earth_wt_pct", 0.15)), 2),
                "Octane_RON": float(m["octane_ron"]),
                "Gasoline_wt_pct": float(m["gasoline_yield_wt"]),
                "Coke_wt_pct": float(m["coke_wt_pct"]),
                "Propylene_wt_pct": float(m["propylene_yield_wt"])
            })
        df_hist = pd.DataFrame(flat_hist)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Multi-Fidelity Discovery Metrics")
            st.metric("Total Observations Evaluated", len(df_hist))
            n_hf = len(df_hist[df_hist["Fidelity_Tier"].str.contains("High-Fidelity")])
            st.metric("High-Fidelity ACE Pilot Runs", f"{n_hf} physical runs")
            st.metric("Max Discovered Octane", f"{df_hist['Octane_RON'].max():.1f} RON")
            st.metric("Max Propylene Yield", f"{df_hist['Propylene_wt_pct'].max():.1f} wt%")
            st.metric("Lowest Coke Deposit", f"{df_hist['Coke_wt_pct'].min():.1f} wt%")

        with col2:
            fig_3d = px.scatter_3d(
                df_hist,
                x="Octane_RON",
                y="Coke_wt_pct",
                z="Gasoline_wt_pct",
                color="Fidelity_Tier",
                size="Propylene_wt_pct",
                size_max=14,
                title="Multi-Fidelity Pareto Frontier: Octane (RON) vs Coke vs Gasoline",
                labels={"Propylene_wt_pct": "Propylene %", "Coke_wt_pct": "Coke wt%", "Octane_RON": "Octane (RON)"},
                color_discrete_map={"High-Fidelity ACE Pilot (s=1.0)": "#e31a1c", "Low-Fidelity ODE Twin (s=0.0)": "#1f78b4"},
                hover_data=["ZSM5_pct", "UCS_Angstrom", "Framework_Si_Al", "Experiment_ID"]
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("Multi-Fidelity Active Learning Knowledge Fabric")
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No active learning history found. Run `run_sota_autonomous_campaign.py` to seed data.")

# ==============================================================================
# TAB 2: 3D Crystallography & Molecular Confinement
# ==============================================================================
with tabs[1]:
    st.header("3D Zeolite Crystallography & Molecular Confinement Engine")
    st.markdown("Sub-nanometer pore dimension analysis based on International Zeolite Association (IZA) framework topologies.")

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        selected_fw = st.selectbox("Select Zeolite Framework Topology", ["FAU", "MFI", "BEA"], index=0)
        cif_info = ZeoliteCrystallographyEngine.evaluate_steric_accessibility(selected_fw)

        st.subheader(f"Framework: {cif_info['framework_name']} ({cif_info['framework_code']})")
        st.write(f"• **Pore Limiting Diameter (PLD):** `{cif_info['pore_window_angstrom']:.2f} Å`")
        st.write(f"• **Largest Internal Cavity (LCD):** `{cif_info['largest_cavity_angstrom']:.2f} Å`")

        st.subheader("Accessible Reactant Molecules (In-Pore Cracking)")
        st.dataframe(pd.DataFrame(cif_info["allowed_diffusing_reactants"]), use_container_width=True)

    with col_c2:
        st.subheader("Sterically Excluded Heavy Molecules (Matrix Cracking Required)")
        st.dataframe(pd.DataFrame(cif_info["sterically_excluded_reactants"]), use_container_width=True)

        cif_path = ROOT_DIR / "data" / f"{selected_fw.lower()}_zeolite_structure.cif"
        if cif_path.exists():
            with open(cif_path, "r", encoding="utf-8") as f:
                cif_text = f.read()
            st.download_button(
                label=f"📥 Download {selected_fw} Structure (.cif)",
                data=cif_text,
                file_name=f"{selected_fw.lower()}_zeolite.cif",
                mime="chemical/x-cif"
            )
            with st.expander("Inspect Crystallographic Information File (CIF)"):
                st.code(cif_text, language="text")

# ==============================================================================
# TAB 3: Petrochemical 10-Lump Kinetic Twin
# ==============================================================================
with tabs[2]:
    st.header("Petrochemical 10-Lump Non-Isothermal Kinetic Digital Twin")
    st.markdown("Resolves stiff ODE cracking kinetics coupled with Breck-Flanigen stoichiometry ($N_{Al}$) and non-isothermal endotherm.")

    col_k1, col_k2 = st.columns([1, 3])
    with col_k1:
        st.subheader("Reactor & Catalyst Levers")
        inlet_t = st.slider("Riser Inlet Temp (°C)", 500.0, 550.0, 530.0, step=1.0)
        c_to_o = st.slider("Cat-to-Oil Ratio", 4.5, 8.5, 6.5, step=0.1)
        zsm5_ratio = st.slider("ZSM-5 Additive Ratio (%)", 5.0, 35.0, 23.6, step=0.5) / 100.0
        ucs_input = st.number_input("Unit Cell Size (Å)", 24.22, 24.50, 24.265, format="%.3f")
        re_wt_input = st.slider("Rare Earth ($RE_2O_3$ wt%)", 0.0, 2.5, 0.15, step=0.05)

        st.subheader("Feed Contaminants & Poisons")
        ni_ppm_in = st.slider("Feed Nickel (Ni ppm)", 0.0, 5.0, 1.5, step=0.1)
        v_ppm_in = st.slider("Feed Vanadium (V ppm)", 0.0, 8.0, 3.0, step=0.2)
        basic_n_in = st.slider("Basic Nitrogen (ppm)", 100, 2000, 950, step=50)

        # Breck-Flanigen physics
        n_al, si_al_locked = IndustrialBreckFlanigenManifold.ucs_to_framework_stoichiometry(ucs_input)
        active_b, ht_idx = IndustrialBreckFlanigenManifold.calculate_active_sites(n_al, basic_n_in, re_wt_input)
        dai = BreckFlanigenZeolitePhysics.calculate_davison_attrition_index(25.0, 35.0)

        st.info(f"**Framework Al ($N_{{Al}}$):** `{n_al:.2f} Al/uc`\n\n**Framework Si/Al (Locked):** `{si_al_locked:.2f}`\n\n**Poisoned Active Sites:** `{active_b:.2f} / uc`\n\n**Hydrogen Transfer Index:** `{ht_idx:.2f}`\n\n**Davison Attrition (DAI):** `{dai:.2f}`")

    with col_k2:
        twin = IndustrialPetrochemical10LumpTwin(
            inlet_temp_c=inlet_t,
            cat_to_oil=c_to_o,
            usy_ucs_angstrom=ucs_input,
            zsm5_fraction=zsm5_ratio,
            rare_earth_wt_pct=re_wt_input,
            feed_ni_ppm=ni_ppm_in,
            feed_v_ppm=v_ppm_in,
            basic_nitrogen_ppm=basic_n_in
        )
        df_10lump = twin.simulate(t_span=(0.0, 3.0), num_points=60)
        pt_last = df_10lump.iloc[-1]

        # WGC compressor alert banner
        wgc_flow = pt_last['wgc_flow_nm3_h']
        if pt_last['wgc_trip_flag']:
            st.error(f"🚨 **WET GAS COMPRESSOR TRIP HAZARD:** Volumetric flow ({wgc_flow:.0f} Nm³/h) exceeds equipment limit (38,000 Nm³/h)! Reduce severity or additive.")
        else:
            st.success(f"✅ **WET GAS COMPRESSOR STATUS: NORMAL** ({wgc_flow:.0f} Nm³/h / 38,000 Nm³/h limit)")

        fig_10lump = go.Figure()
        fig_10lump.add_trace(go.Scatter(x=df_10lump["residence_time_sec"], y=df_10lump["Gasoline_wt_pct"], name="Gasoline Cut (wt%)", line=dict(color="#2ca02c", width=3)))
        fig_10lump.add_trace(go.Scatter(x=df_10lump["residence_time_sec"], y=df_10lump["Propylene_C3=_wt_pct"], name="Propylene Olefin (wt%)", line=dict(color="#ff7f0e", width=2.5)))
        fig_10lump.add_trace(go.Scatter(x=df_10lump["residence_time_sec"], y=df_10lump["Dry_Gas_wt_pct"], name="Dry Gas (H2/C1/C2) (wt%)", line=dict(color="#7f7f7f", width=2, dash="dash")))
        fig_10lump.add_trace(go.Scatter(x=df_10lump["residence_time_sec"], y=df_10lump["LCO_diesel_wt_pct"], name="LCO Diesel Cut (wt%)", line=dict(color="#9467bd", width=2)))
        fig_10lump.add_trace(go.Scatter(x=df_10lump["residence_time_sec"], y=df_10lump["Coke_wt_pct"], name="Coke Deposit (wt%)", line=dict(color="#d62728", width=2)))

        fig_10lump.update_layout(
            title=f"Industrial 10-Lump Conserved Cracking Profile (Inlet = {inlet_t}°C, C/O = {c_to_o:.1f}, Ni = {ni_ppm_in} ppm)",
            xaxis_title="Residence Time (seconds)",
            yaxis_title="Product Yield (wt%)",
            template="plotly_white"
        )
        st.plotly_chart(fig_10lump, use_container_width=True)

# ==============================================================================
# TAB 4: Regenerator Aging Loop
# ==============================================================================
with tabs[3]:
    st.header("FCC Regenerator Combustion & Multi-Cycle Hydrothermal Aging")
    st.markdown("Simulates spent catalyst coke combustion and long-term hydrothermal dealumination (ASTM D4463) towards Equilibrium Catalyst (E-Cat).")

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        coke_spent = st.slider("Spent Catalyst Coke Deposit (wt%)", 0.5, 1.5, 0.92, step=0.02)
        steam_temp = st.slider("Regenerator Steaming Temp (°C)", 720.0, 800.0, 760.0, step=5.0)
        cycles = st.slider("Steaming / Aging Cycles", 10, 80, 40, step=5)

        loop = FCCRegeneratorAgingLoop(regenerator_temp_c=steam_temp)
        reg_res = loop.calculate_regenerator_state(riser_exit_temp_c=520.0, coke_on_spent_cat_wt_pct=coke_spent)

        st.subheader("Combustion Heat Balance")
        st.metric("Regenerator Dense Bed Temp", f"{reg_res['regenerator_dense_bed_temp_c']:.1f} °C")
        st.metric("Exothermic Combustion Rise", f"+{reg_res['combustion_temperature_lift_c']:.1f} °C")
        st.metric("Flue Gas CO2 Generated", f"{reg_res['flue_gas_co2_kg_per_ton_cat']:.1f} kg/t cat")

    with col_r2:
        df_aging = FCCRegeneratorAgingLoop.simulate_hydrothermal_aging(
            fresh_ucs_angstrom=ucs_input,
            fresh_zeolite_sa_m2_g=280.0,
            num_cycles=cycles,
            steaming_temp_c=steam_temp
        )

        fig_aging = px.line(
            df_aging,
            x="aging_cycle",
            y="unit_cell_size_angstrom",
            title=f"Hydrothermal Dealumination: UCS Decay over {cycles} Steaming Cycles",
            labels={"aging_cycle": "Aging Cycles", "unit_cell_size_angstrom": "Unit Cell Size (Å)"}
        )
        st.plotly_chart(fig_aging, use_container_width=True)
        st.dataframe(df_aging.iloc[::5].reset_index(drop=True), use_container_width=True)

# ==============================================================================
# TAB 5: GC-MS Analytics
# ==============================================================================
with tabs[4]:
    st.header("Real-Time Online GC-MS Spectral Peak Deconvolution")
    st.markdown("Deconvolutes high-resolution FID chromatographic streams into ASTM D2887 PIONA fractions.")

    raw_gc = GCMSProductAnalyzer.simulate_chromatogram(zsm5_booster=True)
    results = GCMSProductAnalyzer.analyze_spectrum(raw_gc)

    fig_gc = px.line(
        raw_gc,
        x="retention_time_min",
        y="detector_current_pa",
        title="Online FID Chromatogram Detector Stream (0 - 24 minutes)",
        labels={"retention_time_min": "Retention Time (min)", "detector_current_pa": "Current (pA)"}
    )
    st.plotly_chart(fig_gc, use_container_width=True)

    y_data = results["resolved_yields_wt_pct"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calculated Octane (RON)", f"{results['gasoline_octane_ron']} RON")
    c2.metric("Gasoline Liquid Cut", f"{y_data['gasoline_cut']}%")
    c3.metric("Propylene Olefin", f"{y_data['propylene_olefin']}%")
    c4.metric("Total LPG Cut", f"{y_data['lpg_total']}%")

# ==============================================================================
# TAB 6: XRD / BET Analytical QC Gates
# ==============================================================================
with tabs[5]:
    st.header("Powder XRD & N2 Physisorption Analytical QC Gates")
    st.markdown("Verifies relative zeolite crystallinity (ASTM D3906) and BET surface area/mesoporosity (ASTM D3663) before cracking.")

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        st.subheader("Powder XRD Crystallinity Scan (ASTM D3906)")
        ref_intensity = st.slider("Reference Reflection Peak Intensity", 8000.0, 16000.0, 12500.0, step=500.0)
        two_theta_demo = np.linspace(10.0, 40.0, 1200)
        intensity_demo = 350.0 + np.random.normal(0, 25, len(two_theta_demo))
        for p in [15.4, 18.4, 20.1, 23.3, 26.7, 31.0]:
            intensity_demo += 12200.0 * np.exp(-0.5 * ((two_theta_demo - p) / 0.12) ** 2)

        xrd_out = RealInstrumentDataIngestionEngine.parse_xrd_crystallinity_gate(two_theta_demo, intensity_demo, reference_peak_intensity=ref_intensity)
        st.metric("Relative Crystallinity", f"{xrd_out['relative_crystallinity_pct']}%")
        if "PASS" in xrd_out["qc_status"]:
            st.success(f"Status: {xrd_out['qc_status']}")
        else:
            st.error(f"Status: {xrd_out['qc_status']}")

        fig_xrd = px.line(x=two_theta_demo, y=intensity_demo, title="Powder XRD Diffractogram (2-theta 10° - 40°)", labels={"x": "2-Theta (degrees)", "y": "Counts"})
        st.plotly_chart(fig_xrd, use_container_width=True)

    with col_q2:
        st.subheader("N2 Physisorption Textural Analysis (ASTM D3663)")
        p_p0_demo = np.linspace(0.02, 0.95, 30)
        v_ads_demo = 175.0 * (p_p0_demo / (1.0 - p_p0_demo + 0.05)) ** 0.18 + p_p0_demo * 75.0
        bet_out = RealInstrumentDataIngestionEngine.parse_bet_physisorption(p_p0_demo, v_ads_demo)

        st.metric("Total BET Surface Area", f"{bet_out['total_bet_surface_area_m2_g']} m²/g")
        st.metric("Matrix Mesopore Area", f"{bet_out['matrix_mesopore_area_m2_g']} m²/g")
        st.metric("Total Pore Volume", f"{bet_out['total_pore_volume_cm3_g']} cm³/g")

        fig_bet = px.scatter(x=p_p0_demo, y=v_ads_demo, title="N2 Adsorption Isotherm @ 77 K", labels={"x": "Relative Pressure (P/P0)", "y": "Quantity Adsorbed (cm³/g STP)"})
        fig_bet.update_traces(mode="lines+markers")
        st.plotly_chart(fig_bet, use_container_width=True)

# ==============================================================================
# TAB 7: Economics & CO2 Emissions
# ==============================================================================
with tabs[6]:
    st.header("Refinery Techno-Economics (TEA) & Scope 1 CO2 Emissions")
    st.markdown("Commercial margin analysis and carbon pricing impact on a 40,000 bpd FCC unit.")

    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        crude_price = st.slider("VGO Feedstock Price ($/bbl)", 60.0, 95.0, 72.0, step=1.0)
        co2_tax = st.slider("ETS Carbon Tax ($/metric ton CO2)", 20.0, 150.0, 65.0, step=5.0)

        FCCTechnoEconomicAnalyzer.PRICES_PER_KG["vgo_feed"] = crude_price / 142.0
        FCCTechnoEconomicAnalyzer.CO2_CARBON_TAX_PER_TON = co2_tax

        sample_yields = {
            "VGO_feed_wt_pct": float(df_10lump.iloc[-1]["VGO_feed_wt_pct"]),
            "Bottoms_HCO_wt_pct": float(df_10lump.iloc[-1]["Bottoms_HCO_wt_pct"]),
            "Gasoline_wt_pct": float(df_10lump.iloc[-1]["Gasoline_wt_pct"]),
            "Propylene_C3=_wt_pct": float(df_10lump.iloc[-1]["Propylene_C3=_wt_pct"]),
            "Propane_C3_wt_pct": float(df_10lump.iloc[-1]["Propane_C3_wt_pct"]),
            "Butylenes_C4=_wt_pct": float(df_10lump.iloc[-1]["Butylenes_C4=_wt_pct"]),
            "Isobutane_iC4_wt_pct": float(df_10lump.iloc[-1]["Isobutane_iC4_wt_pct"]),
            "LCO_diesel_wt_pct": float(df_10lump.iloc[-1]["LCO_diesel_wt_pct"]),
            "Dry_Gas_wt_pct": float(df_10lump.iloc[-1]["Dry_Gas_wt_pct"]),
            "Coke_wt_pct": float(df_10lump.iloc[-1]["Coke_wt_pct"])
        }
        cat_blend = {"usy_fraction": 1.0 - zsm5_ratio, "zsm5_fraction": zsm5_ratio, "rare_earth_wt_pct": re_wt_input}

        tea_res = FCCTechnoEconomicAnalyzer.evaluate_economics_and_emissions(
            product_slate_wt_pct=sample_yields,
            gasoline_ron=float(df_10lump.iloc[-1]["gasoline_ron"]),
            catalyst_blend=cat_blend
        )

        fin = tea_res["financial_metrics"]
        em = tea_res["emissions_metrics"]

        st.subheader("Financial Unit Margins")
        st.metric("Net Refining Margin", f"${fin['net_margin_per_bbl']:.2f} / bbl")
        st.metric("Annual Unit EBITDA", f"${fin['annual_unit_ebitda_usd_millions']:.1f} Million USD")
        st.metric("Carbon Tax Debit", f"-${fin['carbon_tax_debit_per_bbl']:.2f} / bbl")

    with col_e2:
        rev_df = pd.DataFrame(list(tea_res["revenue_breakdown_usd_per_bbl"].items()), columns=["Product Stream", "Revenue ($/bbl)"])
        fig_rev = px.bar(rev_df, x="Product Stream", y="Revenue ($/bbl)", color="Product Stream", title="Product Slate Revenue Contribution ($/bbl)")
        st.plotly_chart(fig_rev, use_container_width=True)

        st.subheader("Carbon Emissions Profile")
        st.write(f"• **Scope 1 CO2 Intensity:** `{em['scope_1_co2_intensity_kg_per_bbl']:.2f} kg CO2 / bbl feed`")
        st.write(f"• **Daily Flue Gas Volume:** `{em['daily_flue_gas_co2_metric_tons']:.1f} metric tons CO2 / day`")

# ==============================================================================
# TAB 8: Executive LLM Audit Report
# ==============================================================================
with tabs[7]:
    st.header("Executive LLM LabAgent Audit & Decision Directives")
    st.markdown("Automated industrial audit synthesizing chemistry, kinetics, economics, and robot instructions.")

    rep_path = ROOT_DIR / "data" / "executive_engineering_report.md"
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.markdown(report_text)
        st.download_button(
            label="📥 Download Executive Report (.md)",
            data=report_text,
            file_name="executive_engineering_report.md",
            mime="text/markdown"
        )
    else:
        st.info("Run `python src/analysis/llm_lab_agent.py` to generate the executive report.")
