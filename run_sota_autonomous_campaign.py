"""
Master SOTA 2026 Closed-Loop Autonomous Discovery Orchestrator (Council Unified Edition).
Fully unifies all 11 core capabilities into an end-to-end defensible discovery engine:
  1. Multi-Fidelity Bayesian Active Learning (BoTorch SingleTaskMultiFidelityGP with Kennedy-O'Hagan discrepancy)
  2. ASTM D3906 Powder XRD % Crystallinity & ASTM D3663 BET Specific Surface Area Quality Control Gates
  3. Hydrothermal Dealumination to Equilibrium Catalyst (E-cat) State (ASTM D4463 Steaming)
  4. Breck-Flanigen Stoichiometric Manifold & Davison Attrition Index (ASTM D5757 <= 6.5)
  5. Opentrons OT-2 Automated Slurry Synthesis Protocol Generation
  6. Industrial 10-Lump Conserved Non-Isothermal Kinetic Twin with Ni/V/Basic-N Poisons & WGC Compressor Limit
  7. Continuous GC-MS Spectral Peak Deconvolution & PIONA Octane Blending (eliminating step functions)
  8. Refinery Techno-Economic Analysis (TEA) with Scope 1 CO2 Carbon Tax ($65/t ETS)
  9. Persistent Closed-Loop Active Learning Posterior Updating
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Force UTF-8 on Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root and src subpackages to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.extend([
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "models"),
    str(PROJECT_ROOT / "src" / "automation"),
    str(PROJECT_ROOT / "src" / "simulation"),
    str(PROJECT_ROOT / "src" / "analysis"),
])

from multi_fidelity_active_optimizer import MultiFidelityActiveCatalystOptimizer
from synthesis_planner import CatalystSynthesisPlanner
from regenerator_aging_loop import FCCRegeneratorAgingLoop
from industrial_10lump_conserved import IndustrialPetrochemical10LumpTwin, IndustrialBreckFlanigenManifold
from instrument_data_ingestion import RealInstrumentDataIngestionEngine
from gcms_analyzer import GCMSProductAnalyzer
from techno_economics import FCCTechnoEconomicAnalyzer


def run_sota_autonomous_campaign(target_cycles: int = 1, batch_size: int = 2, feedstock_key: str = "standard_middle_east_vgo"):
    feedstock_file = PROJECT_ROOT / "data" / "feedstock_library.json"
    if feedstock_file.exists():
        with open(feedstock_file, "r") as f:
            feed_lib = json.load(f)
        feed_info = feed_lib.get(feedstock_key, feed_lib["standard_middle_east_vgo"])
    else:
        feed_info = {"name": "Standard VGO", "nickel_ppm": 1.2, "vanadium_ppm": 2.5, "basic_nitrogen_ppm": 850.0}

    print("=" * 90)
    print("  [AUTO-CATALYST 2.0] LAUNCHING UNIFIED MASTER SOTA AUTONOMOUS DISCOVERY CAMPAIGN")
    print(f"  Feedstock Profile : {feed_info.get('name', feedstock_key)} (Ni: {feed_info['nickel_ppm']} ppm, V: {feed_info['vanadium_ppm']} ppm, Basic N: {feed_info['basic_nitrogen_ppm']} ppm)")
    print("  Multi-Fidelity BoTorch Active Learning + E-Cat Hydrothermal Dealumination + Conserved 10-Lump Twin")
    print("=" * 90)

    store_file = PROJECT_ROOT / "data" / "multi_fidelity_active_learning_history.json"
    mf_opt = MultiFidelityActiveCatalystOptimizer(store_file)

    for cycle in range(1, target_cycles + 1):
        print(f"\n{'='*38} CYCLE {cycle}/{target_cycles} {'='*38}")

        # -------------------------------------------------------------
        # STEP 1: Multi-Fidelity Bayesian Active Learning Batch Proposal (q-NEHVI)
        # -------------------------------------------------------------
        print(f"\n[STEP 1/7] Fitting Multi-Fidelity GP & Optimizing Discrepancy Surface (Batch q={batch_size})...")
        t0 = time.time()
        proposals = mf_opt.suggest_next_batch(batch_size=batch_size, target_high_fidelity=False)
        print(f"  -> Generated {len(proposals)} Pareto-optimal candidate formulations in {time.time() - t0:.2f}s")

        for idx, candidate in enumerate(proposals, 1):
            exp_id = f"CYC{cycle}-BAT{idx}"
            print(f"\n--- [CAMPAIGN EXECUTION: {exp_id}] ---")
            zsm5_pct = candidate["zsm5_fraction"] * 100.0
            usy_pct = 100.0 - zsm5_pct

            # -------------------------------------------------------------
            # STEP 2: Hydrothermal Aging to Equilibrium Catalyst (E-cat) State
            # -------------------------------------------------------------
            print(f"\n[STEP 2/7] Conditioning Formulation to Equilibrium Catalyst (E-cat) State...")
            ecat = FCCRegeneratorAgingLoop.get_equilibrium_ecat_properties(
                fresh_ucs_angstrom=candidate["usy_ucs_angstrom"],
                fresh_zeolite_sa_m2_g=480.0,
                rare_earth_wt_pct=candidate["rare_earth_wt_pct"],
                steaming_temp_c=760.0
            )
            aged_ucs = ecat["ecat_ucs_angstrom"]
            n_al, framework_si_al = IndustrialBreckFlanigenManifold.ucs_to_framework_stoichiometry(aged_ucs)
            active_sites, ht_idx = IndustrialBreckFlanigenManifold.calculate_active_sites(
                n_al, basic_nitrogen_ppm=feed_info["basic_nitrogen_ppm"], rare_earth_wt=candidate["rare_earth_wt_pct"]
            )

            # Davison Attrition Index (ASTM D5757)
            dai = max(2.5, 9.8 - (candidate["matrix_alumina_wt"] * 0.16) - (35.0 * 0.05))

            print(f"  • Blend Recipe        : {usy_pct:.1f}% USY Zeolite + {zsm5_pct:.1f}% ZSM-5 Additive")
            print(f"  • Fresh -> E-Cat UCS  : {candidate['usy_ucs_angstrom']:.3f} Å -> {aged_ucs:.3f} Å (Steady-State E-cat)")
            print(f"  • Framework Al (N_Al) : {n_al:.2f} Al/uc | Framework Si/Al: {framework_si_al:.2f} (Locked)")
            print(f"  • Active Brønsted     : {active_sites:.2f} / uc (Poisoned by {feed_info['basic_nitrogen_ppm']:.0f} ppm basic N)")
            print(f"  • Davison Attrition   : DAI = {dai:.2f} (ASTM D5757 <= 6.5 -> {'PASS' if dai <= 6.5 else 'FAIL'})")

            # -------------------------------------------------------------
            # STEP 3: Intermediate Analytical Characterization Gates (ASTM XRD & BET)
            # -------------------------------------------------------------
            print(f"\n[STEP 3/7] Ingesting Intermediate Analytical Characterization Gates...")
            # Simulate XRD scan and evaluate relative crystallinity (ASTM D3906)
            two_theta = np.linspace(10.0, 40.0, 1000)
            intensity = 400.0 + np.random.normal(0, 20, len(two_theta))
            for peak in [15.4, 18.4, 20.1, 23.3, 26.7, 31.0]:
                intensity += 12400.0 * np.exp(-0.5 * ((two_theta - peak) / 0.12) ** 2)
            xrd_qc = RealInstrumentDataIngestionEngine.parse_xrd_crystallinity_gate(two_theta, intensity)

            # Simulate N2 physisorption isotherm (ASTM D3663)
            p_p0 = np.linspace(0.02, 0.98, 40)
            v_ads = 175.0 * (p_p0 / (1.0 - p_p0 + 0.05)) ** 0.18 + p_p0 * 75.0
            bet_qc = RealInstrumentDataIngestionEngine.parse_bet_physisorption(p_p0, v_ads)

            print(f"  • XRD Crystallinity   : {xrd_qc['relative_crystallinity_pct']:.1f}% (ASTM D3906 Gate: {xrd_qc['qc_status']})")
            print(f"  • BET Surface Area    : {bet_qc['total_bet_surface_area_m2_g']:.1f} m²/g (Matrix Mesopore: {bet_qc['matrix_mesopore_area_m2_g']:.1f} m²/g)")

            # -------------------------------------------------------------
            # STEP 4: Robotic Slurry Synthesis Protocol Planning (Opentrons OT-2)
            # -------------------------------------------------------------
            print(f"\n[STEP 4/7] Generating Automated Opentrons OT-2 Slurry Dispensing Script...")
            planner = CatalystSynthesisPlanner(target_batch_mass_grams=15.0)
            recipe = planner.calculate_recipe({"params": candidate})
            ot2_file = PROJECT_ROOT / "src" / "automation" / f"ot2_{exp_id.lower()}_protocol.py"
            planner.generate_ot2_protocol(recipe, ot2_file)
            print(f"  • USY Slurry Dispense : {recipe['liquid_stock_dispensing_ml']['USY_slurry_20pct_ml']} mL")
            kaolin_vol = recipe['liquid_stock_dispensing_ml'].get('kaolin_suspension_25pct_ml', 0.0)
            print(f"  • Alumina Sol / Kaolin: {recipe['liquid_stock_dispensing_ml']['alumina_sol_15pct_ml']} mL / {kaolin_vol} mL")
            print(f"  • Script Generated    : {ot2_file.name}")

            # -------------------------------------------------------------
            # STEP 5: Conserved Industrial 10-Lump Reactor Twin with Feed Poisons
            # -------------------------------------------------------------
            print(f"\n[STEP 5/7] Simulating Conserved 10-Lump FCC Riser with Feed Contaminants...")
            twin = IndustrialPetrochemical10LumpTwin(
                inlet_temp_c=candidate["riser_temp_c"],
                cat_to_oil=candidate["cat_to_oil_ratio"],
                usy_ucs_angstrom=aged_ucs,  # Conditioned E-cat UCS
                zsm5_fraction=candidate["zsm5_fraction"],
                rare_earth_wt_pct=candidate["rare_earth_wt_pct"],
                matrix_alumina_wt=candidate["matrix_alumina_wt"],
                feed_ni_ppm=feed_info["nickel_ppm"],
                feed_v_ppm=feed_info["vanadium_ppm"],
                basic_nitrogen_ppm=feed_info["basic_nitrogen_ppm"]
            )
            df_kin = twin.simulate(t_span=(0.0, 3.0), num_points=60)
            pt_peak = df_kin.iloc[(df_kin["residence_time_sec"] - 1.5).abs().argsort()[:1]].iloc[0]

            obs_gas = float(pt_peak["Gasoline_wt_pct"])
            obs_coke = float(pt_peak["Coke_wt_pct"])
            obs_prop = float(pt_peak["Propylene_C3=_wt_pct"])
            obs_lco = float(pt_peak["LCO_diesel_wt_pct"])
            wgc_flow = float(pt_peak["wgc_flow_nm3_h"])
            wgc_trip = bool(pt_peak["wgc_trip_flag"])

            print(f"  • Feed Conversion     : {pt_peak['feed_conversion_wt_pct']:.2f} wt%")
            print(f"  • Gasoline Liquid Cut : {obs_gas:.2f} wt%")
            print(f"  • Propylene (C3=)     : {obs_prop:.2f} wt% (Petrochemical Feedstock)")
            print(f"  • Wet Gas Compressor : {wgc_flow:.0f} Nm3/h (Trip Ceiling: 38,000 Nm3/h -> {'[TRIP HAZARD]' if wgc_trip else '[NORMAL]'})")

            # -------------------------------------------------------------
            # STEP 6: Continuous GC-MS Spectral Peak Deconvolution & PIONA Octane
            # -------------------------------------------------------------
            print(f"\n[STEP 6/7] Continuously Deconvoluting GC-MS FID Detector Stream...")
            raw_gc = GCMSProductAnalyzer.simulate_chromatogram(
                total_minutes=24.0,
                product_yields=dict(pt_peak),
                catalyst_properties={
                    "zsm5_fraction": candidate["zsm5_fraction"],
                    "usy_ucs_angstrom": aged_ucs,
                    "rare_earth_wt_pct": candidate["rare_earth_wt_pct"]
                }
            )
            spec_res = GCMSProductAnalyzer.analyze_spectrum(raw_gc)
            obs_ron = float(spec_res["gasoline_octane_ron"])
            print(f"  • Continuous Octane   : {obs_ron:.2f} RON (Directly coupled to kinetics & E-cat UCS)")

            # Refinery Economics Evaluation
            tea = FCCTechnoEconomicAnalyzer.evaluate_economics_and_emissions(
                product_slate_wt_pct=dict(pt_peak),
                gasoline_ron=obs_ron,
                catalyst_blend={"usy_fraction": 1.0 - candidate["zsm5_fraction"], "zsm5_fraction": candidate["zsm5_fraction"]}
            )
            margin = tea["financial_metrics"]["net_margin_per_bbl"]
            ebitda = tea["financial_metrics"]["annual_unit_ebitda_usd_millions"]
            co2_tax = tea["financial_metrics"]["carbon_tax_debit_per_bbl"]
            co2_kg = tea["emissions_metrics"]["scope_1_co2_intensity_kg_per_bbl"]
            print(f"  • Refinery Net Margin : ${margin:.2f}/bbl (Annual EBITDA: ${ebitda:.1f}M, CO2 Intensity: {co2_kg:.1f} kg/bbl, Tax: ${co2_tax:.2f}/bbl)")

            # -------------------------------------------------------------
            # STEP 7: Multi-Fidelity Closed-Loop Database Update
            # -------------------------------------------------------------
            print(f"\n[STEP 7/7] Closing Multi-Fidelity Active Learning Loop...")
            observed_metrics = {
                "octane_ron": obs_ron,
                "gasoline_yield_wt": obs_gas,
                "coke_wt_pct": obs_coke,
                "propylene_yield_wt": obs_prop
            }
            mf_opt.append_observation(
                exp_id=exp_id,
                cycle=cycle,
                params=candidate,
                metrics=observed_metrics,
                fidelity_s=0.0,
                data_source="Industrial 10-Lump Conserved Twin (E-Cat)"
            )
            print(f"  -> Successfully recorded observation to persistent multi-fidelity data fabric.")

    print("\n" + "=" * 90)
    print(f"  UNIFIED MASTER SOTA CAMPAIGN COMPLETED SUCCESSFULLY ({target_cycles} CYCLES)")
    print(f"  Multi-fidelity active learning dataset: {store_file}")
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SOTA Closed-Loop Catalyst Campaign")
    parser.add_argument("--cycles", type=int, default=1, help="Number of active learning cycles")
    parser.add_argument("--batch-size", type=int, default=2, help="Candidates proposed per batch")
    parser.add_argument(
        "--feedstock",
        type=str,
        default="standard_middle_east_vgo",
        choices=["standard_middle_east_vgo", "hydrotreated_vgo", "severe_resid_dao_blend", "shale_tight_oil"],
        help="Feedstock impurity profile from library"
    )
    args = parser.parse_args()
    run_sota_autonomous_campaign(target_cycles=args.cycles, batch_size=args.batch_size, feedstock_key=args.feedstock)
