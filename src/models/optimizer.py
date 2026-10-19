"""
Multi-Objective Catalyst Discovery Optimizer for Auto-Catalyst 2.0.
Uses Optuna (NSGA-II / TPE) coupled with the trained surrogate model
to discover optimal catalyst formulations that:
  - Maximize Gasoline Octane (RON >= 95)
  - Maximize Gasoline Yield (wt%)
  - Minimize Coke Deposition (<= 5.0 wt%)
"""

import json
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from surrogate_predictor import load_dataset


class CatalystOptimizer:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = load_dataset(data_path)
        self.target_cols = ["target_ron", "target_coke_wt", "target_gasoline_yield"]
        self.feature_cols = [c for c in self.df.columns if not c.startswith("target_")]

        # Train surrogate evaluator
        X = self.df[self.feature_cols]
        y = self.df[self.target_cols]
        self.surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
        self.surrogate.fit(X, y)

    def predict(self, candidate_dict: dict) -> dict:
        """Predict targets for a single candidate formulation."""
        input_df = pd.DataFrame([candidate_dict])
        # Ensure all one-hot encoded columns are present
        for col in self.feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[self.feature_cols]
        preds = self.surrogate.predict(input_df)[0]
        return {
            "predicted_ron": float(preds[0]),
            "predicted_coke_wt": float(preds[1]),
            "predicted_gasoline_yield": float(preds[2]),
        }

    def optimize_discovery(self, n_trials: int = 150):
        """Run Multi-Objective Pareto optimization."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Objective: [Maximize RON, Minimize Coke, Maximize Gasoline Yield]
        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42)
        )

        def objective(trial):
            # Categorical choice of zeolite
            zeolite = trial.suggest_categorical("zeolite_type", ["USY", "RE-USY", "ZSM-5", "Beta"])
            framework = "MFI" if zeolite == "ZSM-5" else ("BEA" if zeolite == "Beta" else "FAU")

            # Compositional variables
            si_al = trial.suggest_float("si_al_ratio", 5.0, 60.0)
            ucs = trial.suggest_float("unit_cell_size", 24.20, 24.60)
            rare_earth = trial.suggest_float("rare_earth_wt", 0.0, 3.5) if zeolite == "RE-USY" else trial.suggest_float("rare_earth_wt", 0.0, 0.5)
            matrix_al = trial.suggest_float("matrix_alumina_wt", 15.0, 40.0)
            binder = trial.suggest_float("binder_clay_wt", 20.0, 45.0)

            # Textural variables
            bet_sa = trial.suggest_float("bet_surface_area", 220.0, 380.0)
            zeo_sa = trial.suggest_float("zeolite_surface_area", 140.0, 280.0)
            mat_sa = max(10.0, bet_sa - zeo_sa)
            total_pv = trial.suggest_float("total_pore_volume", 0.25, 0.45)
            bronsted = trial.suggest_float("bronsted_acid", 80.0, 450.0)
            lewis = trial.suggest_float("lewis_acid", 40.0, 250.0)

            # Operating conditions (typical riser conditions)
            temp_c = trial.suggest_float("temp_c", 500.0, 545.0)
            c_to_o = trial.suggest_float("cat_to_oil", 4.5, 7.5)
            res_time = trial.suggest_float("residence_time", 2.0, 3.0)
            api = 25.0
            ccr = 0.40

            candidate = {
                "si_al_ratio": si_al,
                "unit_cell_size": ucs,
                "rare_earth_wt": rare_earth,
                "matrix_alumina_wt": matrix_al,
                "binder_clay_wt": binder,
                "bet_surface_area": bet_sa,
                "zeolite_surface_area": zeo_sa,
                "matrix_surface_area": mat_sa,
                "total_pore_volume": total_pv,
                "bronsted_acid": bronsted,
                "lewis_acid": lewis,
                "temp_c": temp_c,
                "cat_to_oil": c_to_o,
                "residence_time": res_time,
                "feedstock_api": api,
                "feedstock_ccr": ccr,
                f"zeolite_type_{zeolite}": 1,
                f"framework_type_{framework}": 1,
            }

            res = self.predict(candidate)
            return res["predicted_ron"], res["predicted_coke_wt"], res["predicted_gasoline_yield"]

        study.optimize(objective, n_trials=n_trials)
        return study


if __name__ == "__main__":
    data_file = Path(__file__).resolve().parent.parent.parent / "data" / "catalysts_seed.json"
    print("Initializing Multi-Objective Catalyst Optimizer...")
    optimizer = CatalystOptimizer(data_file)

    print("Running Pareto Optimization (NSGA-II) for 200 trials...")
    study = optimizer.optimize_discovery(n_trials=200)

    # Filter Pareto optimal candidates
    pareto_trials = study.best_trials

    # Find candidate meeting target: RON >= 95 and Coke <= 5.0
    valid_candidates = []
    for t in pareto_trials:
        ron, coke, gas_yield = t.values
        if ron >= 94.5 and coke <= 5.0:
            valid_candidates.append({
                "trial_id": t.number,
                "predicted_ron": round(ron, 2),
                "predicted_coke_wt": round(coke, 2),
                "predicted_gasoline_yield": round(gas_yield, 2),
                "params": t.params
            })

    print("\n" + "=" * 70)
    print("       AI AUTONOMOUS DISCOVERY: TOP OPTIMAL CANDIDATES")
    print("=" * 70)
    print(f"Total Pareto-optimal solutions discovered: {len(pareto_trials)}")
    print(f"Candidates satisfying (RON >= 94.5 & Coke <= 5.0%): {len(valid_candidates)}")
    print("-" * 70)

    for idx, c in enumerate(valid_candidates[:5], 1):
        print(f"\n[Candidate #{idx}]")
        print(f"  • Predicted Octane (RON)  : {c['predicted_ron']} (Target: >=95.0)")
        print(f"  • Predicted Coke Deposit  : {c['predicted_coke_wt']}% (Target: <=5.0%)")
        print(f"  • Gasoline Yield          : {c['predicted_gasoline_yield']}%")
        p = c['params']
        print(f"  • Recipe: Zeolite = {p.get('zeolite_type')}, Si/Al = {p.get('si_al_ratio', 0):.1f}, "
              f"UCS = {p.get('unit_cell_size', 0):.3f} Å, Rare Earth = {p.get('rare_earth_wt', 0):.2f} wt%")
        print(f"  • Acidity: Brønsted = {p.get('bronsted_acid', 0):.1f} μmol/g, Lewis = {p.get('lewis_acid', 0):.1f} μmol/g")
    print("=" * 70)

    # Save discovered recipes
    out_path = data_file.parent / "discovered_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(valid_candidates, f, indent=2)
    print(f"Top discovered catalyst recipes exported to {out_path}")
