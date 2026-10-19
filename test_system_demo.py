"""
One-Click Comprehensive System Verification & Demo Suite for Auto-Catalyst 2.0.
Verifies all modules end-to-end in < 20 seconds:
  1. 3D Crystallography & CIF Generation
  2. 10-Lump Petrochemical Kinetic Riser Simulation
  3. FCC Regenerator Dense-Bed Combustion & Aging
  4. Online GC-MS Spectral Peak Deconvolution
  5. Techno-Economics & Scope 1 CO2 Footprint
  6. BoTorch Bayesian Active Learning Integrity Check
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.extend([
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "models"),
    str(PROJECT_ROOT / "src" / "automation"),
    str(PROJECT_ROOT / "src" / "simulation"),
    str(PROJECT_ROOT / "src" / "analysis"),
])

def run_system_verification():
    print("=" * 80)
    print("  AUTO-CATALYST 2.0: FULL SYSTEM VERIFICATION & DEMO SUITE")
    print("=" * 80)
    start_total = time.time()

    # 1. Crystallography Engine
    t0 = time.time()
    from zeolite_crystallography import ZeoliteCrystallographyEngine
    fau_res = ZeoliteCrystallographyEngine.evaluate_steric_accessibility("FAU")
    assert len(fau_res["allowed_diffusing_reactants"]) > 0
    print(f"[1/6] 3D Zeolite Crystallography Engine   : PASS ({time.time() - t0:.2f}s) - {fau_res['framework_name']} PLD = {fau_res['pore_window_angstrom']} A")

    # 2. 10-Lump Petrochemical Kinetic Twin
    t0 = time.time()
    from petrochemical_10lump_twin import PetrochemicalFCC10LumpModel
    twin = PetrochemicalFCC10LumpModel(inlet_temp_c=530.0, cat_to_oil=6.5)
    df_kin = twin.simulate(t_span=(0.0, 1.5), num_points=20)
    assert df_kin["Gasoline_wt_pct"].iloc[-1] > 0.0
    print(f"[2/6] 10-Lump Petrochemical Kinetic Twin  : PASS ({time.time() - t0:.2f}s) - Peak Gasoline = {df_kin['Gasoline_wt_pct'].iloc[-1]:.2f} wt%")

    # 3. Regenerator Combustion & Steaming
    t0 = time.time()
    from regenerator_aging_loop import FCCRegeneratorAgingLoop
    loop = FCCRegeneratorAgingLoop()
    reg = loop.calculate_regenerator_state(520.0, 0.92)
    assert reg["regenerator_dense_bed_temp_c"] > 600.0
    print(f"[3/6] FCC Regenerator Combustion Loop     : PASS ({time.time() - t0:.2f}s) - Dense Bed Temp = {reg['regenerator_dense_bed_temp_c']:.1f} °C")

    # 4. GC-MS Spectral Deconvolution
    t0 = time.time()
    from gcms_analyzer import GCMSProductAnalyzer
    raw_gc = GCMSProductAnalyzer.simulate_chromatogram(total_minutes=5.0)
    gc_res = GCMSProductAnalyzer.analyze_spectrum(raw_gc)
    assert gc_res["gasoline_octane_ron"] > 90.0
    print(f"[4/6] GC-MS Peak Deconvolution Engine     : PASS ({time.time() - t0:.2f}s) - Resolved Octane = {gc_res['gasoline_octane_ron']:.2f} RON")

    # 5. Techno-Economics & Carbon Accounting
    t0 = time.time()
    from techno_economics import FCCTechnoEconomicAnalyzer
    full_slate = {
        "Gasoline_wt_pct": 45.0,
        "Propylene_C3=_wt_pct": 8.5,
        "Propane_C3_wt_pct": 2.0,
        "Butylenes_C4=_wt_pct": 6.8,
        "Isobutane_iC4_wt_pct": 3.5,
        "LCO_diesel_wt_pct": 8.5,
        "Dry_Gas_wt_pct": 2.2,
        "Coke_wt_pct": 4.5
    }
    tea = FCCTechnoEconomicAnalyzer.evaluate_economics_and_emissions(
        product_slate_wt_pct=full_slate,
        gasoline_ron=96.5,
        catalyst_blend={"usy_fraction": 0.75, "zsm5_fraction": 0.25}
    )
    assert tea["financial_metrics"]["gross_product_revenue_per_bbl"] > 0.0
    print(f"[5/6] Refinery Economics & CO2 Calculator : PASS ({time.time() - t0:.2f}s) - Net Margin = ${tea['financial_metrics']['net_margin_per_bbl']:.2f}/bbl")

    # 6. Active Learning Data Store Integrity
    t0 = time.time()
    from bayesian_active_optimizer import BayesianActiveCatalystOptimizer
    store_file = PROJECT_ROOT / "data" / "active_learning_history.json"
    bayes_opt = BayesianActiveCatalystOptimizer(store_file)
    x_hist, y_hist = bayes_opt.load_historical_data()
    assert len(x_hist) >= 12
    print(f"[6/6] BoTorch Active Learning Data Fabric : PASS ({time.time() - t0:.2f}s) - {len(x_hist)} Verified Observations Loaded")

    print("-" * 80)
    print(f"ALL SYSTEMS VERIFIED & READY FOR DEMO IN {time.time() - start_total:.2f} SECONDS (100% PASS RATE)")
    print("=" * 80)


if __name__ == "__main__":
    run_system_verification()
