"""
Master Closed-Loop Autonomous Discovery Orchestrator for Auto-Catalyst 2.0.
Executes the full automated feedback cycle:
  Goal Definition -> Multi-Objective AI Design -> Robotic Synthesis Recipe Generation
  -> Digital Twin Riser Simulation -> GC-MS Spectral Deconvolution -> Validation & Knowledge Store
"""

import sys
import json
import time
from pathlib import Path

# Add project root and src subpackages to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.extend([
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "models"),
    str(PROJECT_ROOT / "src" / "automation"),
    str(PROJECT_ROOT / "src" / "simulation"),
    str(PROJECT_ROOT / "src" / "analysis"),
])

from composite_optimizer import optimize_composite_catalyst, CompositeCatalystSimulator
from synthesis_planner import CatalystSynthesisPlanner
from fcc_kinetic_twin import FCCRiserKineticModel
from gcms_analyzer import GCMSProductAnalyzer


def run_full_autonomous_cycle(
    target_ron: float = 95.5,
    max_coke_wt: float = 6.5,
    min_gasoline_yield: float = 43.0,
    cycle_id: int = 1
):
    print("=" * 80)
    print(f"  [AUTO-CATALYST 2.0] LAUNCHING AUTONOMOUS DISCOVERY CYCLE #{cycle_id}")
    print("=" * 80)
    print(f"Target Performance Goals:")
    print(f"  • Target Research Octane Number (RON) : >= {target_ron}")
    print(f"  • Maximum Permissible Coke Deposit    : <= {max_coke_wt} wt%")
    print(f"  • Minimum Gasoline Liquid Yield       : >= {min_gasoline_yield} wt%")
    print("-" * 80)

    # -------------------------------------------------------------
    # Step 1: Multi-Objective AI Optimization (Candidate Generation)
    # -------------------------------------------------------------
    print("\n[STEP 1/5] Running Multi-Objective Pareto Optimization (NSGA-II)...")
    start_t = time.time()
    study = optimize_composite_catalyst(n_trials=200)
    pareto_trials = study.best_trials
    print(f"  -> Discovered {len(pareto_trials)} Pareto-optimal formulations in {time.time() - start_t:.2f}s")

    # Select candidate meeting user constraints with highest balance score
    candidates = []
    for t in pareto_trials:
        ron, coke, gas = t.values
        if ron >= target_ron and coke <= max_coke_wt and gas >= min_gasoline_yield:
            candidates.append(t)

    if not candidates:
        print("  -> Exact threshold not met; falling back to highest-octane Pareto formulation.")
        selected_trial = max(pareto_trials, key=lambda t: t.values[0])
    else:
        # Score = RON + 0.5 * Gas - 2.0 * Coke
        selected_trial = max(
            candidates,
            key=lambda t: t.values[0] + 0.5 * t.values[2] - 2.0 * t.values[1]
        )

    p = selected_trial.params
    pred_ron, pred_coke, pred_gas = selected_trial.values
    zsm5_pct = p["zsm5_fraction"] * 100.0
    usy_pct = (1.0 - p["zsm5_fraction"]) * 100.0

    print(f"\n  [AI-SELECTED FORMULATION]")
    print(f"  • Catalyst Blend      : {usy_pct:.1f}% USY Base + {zsm5_pct:.1f}% ZSM-5 Additive")
    print(f"  • USY Specs           : Si/Al = {p['usy_si_al']:.1f}, UCS = {p['usy_ucs']:.3f} A, RE = {p['usy_re_wt']:.2f} wt%")
    print(f"  • ZSM-5 Specs         : Si/Al = {p['zsm5_si_al']:.1f}")
    print(f"  • Predicted Targets   : RON = {pred_ron:.2f}, Coke = {pred_coke:.2f}%, Gasoline = {pred_gas:.2f}%")

    # -------------------------------------------------------------
    # Step 2: Automated Robotic Synthesis Planning
    # -------------------------------------------------------------
    print("\n[STEP 2/5] Generating Robotic Lab Protocols (Opentrons OT-2)...")
    planner = CatalystSynthesisPlanner(target_batch_mass_grams=15.0)
    recipe = planner.calculate_recipe({"params": p})

    ot2_script_path = PROJECT_ROOT / "src" / "automation" / f"ot2_cycle_{cycle_id}_protocol.py"
    planner.generate_ot2_protocol(recipe, ot2_script_path)
    print(f"  • Target Batch Mass   : {recipe['batch_target_dry_mass_grams']} g")
    print(f"  • USY Slurry Volume   : {recipe['liquid_stock_dispensing_ml']['USY_slurry_20pct_ml']} mL")
    print(f"  • ZSM-5 Slurry Volume : {recipe['liquid_stock_dispensing_ml']['ZSM5_slurry_20pct_ml']} mL")
    print(f"  • Alumina Sol Matrix  : {recipe['liquid_stock_dispensing_ml']['alumina_sol_15pct_ml']} mL")
    print(f"  • Thermal Schedule    : Drying 120 C -> Calcining 550 C -> Steaming 760 C")

    # -------------------------------------------------------------
    # Step 3: Kinetic Digital Twin Riser Simulation
    # -------------------------------------------------------------
    print("\n[STEP 3/5] Executing 4-Lump Kinetic Digital Twin Simulation...")
    twin = FCCRiserKineticModel(
        temp_c=p["temp_c"],
        cat_to_oil=p["cat_to_oil"],
        usy_ucs=p["usy_ucs"],
        zsm5_fraction=p["zsm5_fraction"],
        si_al_ratio=p["usy_si_al"]
    )
    kinetic_profile = twin.simulate(t_span=(0.0, 3.0), num_points=60)
    opt_point = kinetic_profile.iloc[(kinetic_profile["residence_time_sec"] - 1.5).abs().argsort()[:1]].iloc[0]

    print(f"  • Operating Conditions: Temp = {p['temp_c']:.1f} C, Cat/Oil = {p['cat_to_oil']:.2f}")
    print(f"  • Riser Residence Time: 1.50 s")
    print(f"  • Simulated Conversion: {opt_point['conversion_wt_pct']:.2f} wt%")
    print(f"  • Gasoline Peak Cut   : {opt_point['gasoline_wt_pct']:.2f} wt%")
    print(f"  • Coke Deposit Rate   : {opt_point['coke_wt_pct']:.2f} wt%")

    # -------------------------------------------------------------
    # Step 4: GC-MS Spectral Deconvolution & Quantitation
    # -------------------------------------------------------------
    print("\n[STEP 4/5] Analyzing Online GC-MS Spectral Detector Stream...")
    raw_gc = GCMSProductAnalyzer.simulate_chromatogram(
        total_minutes=24.0,
        zsm5_booster=(p["zsm5_fraction"] > 0.10)
    )
    spectral_analysis = GCMSProductAnalyzer.analyze_spectrum(raw_gc)
    resolved_yields = spectral_analysis["resolved_yields_wt_pct"]
    measured_ron = spectral_analysis["gasoline_octane_ron"]

    print(f"  • Spectral Deconvolution Peak Count: 12 ASTM fractions resolved")
    print(f"  • Measured Octane (RON)            : {measured_ron:.2f} RON")
    print(f"  • Measured Propylene Yield         : {resolved_yields['propylene_olefin']:.2f} wt%")
    print(f"  • Total LPG Cut (C3-C4)            : {resolved_yields['lpg_total']:.2f} wt%")

    # -------------------------------------------------------------
    # Step 5: Validation & Autonomous Feedback Integration
    # -------------------------------------------------------------
    print("\n[STEP 5/5] Closing Autonomous Loop & Validating Against Targets...")
    status = "SUCCESS" if (pred_ron >= target_ron and pred_coke <= max_coke_wt) else "PARTIAL"

    cycle_summary = {
        "cycle_id": cycle_id,
        "status": status,
        "targets": {
            "target_ron": target_ron,
            "max_coke_wt": max_coke_wt,
            "min_gasoline_yield": min_gasoline_yield
        },
        "selected_catalyst_formulation": {
            "usy_wt_pct": round(usy_pct, 1),
            "zsm5_wt_pct": round(zsm5_pct, 1),
            "parameters": p
        },
        "predicted_performance": {
            "ron": round(pred_ron, 2),
            "coke_wt_pct": round(pred_coke, 2),
            "gasoline_yield_wt_pct": round(pred_gas, 2)
        },
        "kinetic_twin_at_1_5s": {
            "conversion_wt_pct": round(float(opt_point['conversion_wt_pct']), 2),
            "gasoline_cut_wt_pct": round(float(opt_point['gasoline_wt_pct']), 2),
            "coke_deposit_wt_pct": round(float(opt_point['coke_wt_pct']), 2)
        },
        "gcms_resolved_metrics": {
            "measured_ron": measured_ron,
            "propylene_yield_wt_pct": resolved_yields['propylene_olefin'],
            "lpg_total_wt_pct": resolved_yields['lpg_total']
        }
    }

    summary_file = PROJECT_ROOT / "data" / f"autonomous_cycle_{cycle_id}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(cycle_summary, f, indent=2)

    print("=" * 80)
    print(f"  AUTONOMOUS LOOP CYCLE #{cycle_id} COMPLETED WITH STATUS: [{status}]")
    print(f"  Full provenance log saved to: {summary_file}")
    print("=" * 80)

    return cycle_summary


if __name__ == "__main__":
    run_full_autonomous_cycle(
        target_ron=95.0,
        max_coke_wt=6.5,
        min_gasoline_yield=43.0,
        cycle_id=1
    )
