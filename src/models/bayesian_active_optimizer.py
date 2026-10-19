"""
Bayesian Active Learning Multi-Objective Optimizer for Auto-Catalyst 2.0.
Powered by BoTorch and GPyTorch.

Replaces static heuristic optimization with true Probabilistic Active Learning:
  1. Multi-Output Gaussian Process (MOGP) modeling:
     - Mean performance: mu(x)
     - Epistemic Model Uncertainty: sigma(x)
  2. q-Noisy Expected Hypervolume Improvement (q-NEHVI):
     - Balances Exploitation (high predicted RON & gasoline) vs Exploration (high model uncertainty)
     - Proposes parallel batches (q candidates) for high-throughput robotic execution
  3. Dynamic Knowledge Store & Retraining Loop:
     - Automatically accumulates true physical/simulated observations
     - Retrains Gaussian Process posterior on every cycle (TRUE CLOSED LOOP)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize


class BayesianActiveCatalystOptimizer:
    """
    SOTA 2026 Self-Driving Laboratory Bayesian Active Learning Engine.
    """

    FEATURE_NAMES = [
        "zsm5_fraction",        # [0.05, 0.35]
        "usy_ucs_angstrom",     # [24.22, 24.50]
        "usy_si_al_ratio",      # [5.0, 45.0]
        "rare_earth_wt_pct",    # [0.0, 2.5]
        "matrix_alumina_wt",    # [18.0, 35.0]
        "riser_temp_c",         # [510.0, 545.0]
        "cat_to_oil_ratio",     # [4.5, 8.5]
    ]

    TARGET_NAMES = [
        "octane_ron",           # Maximize
        "gasoline_yield_wt",    # Maximize
        "neg_coke_wt",          # Maximize (-Coke is minimized)
        "propylene_yield_wt"    # Maximize
    ]

    def __init__(self, data_store_path: Path):
        self.store_path = data_store_path
        self.device = torch.device("cpu")
        self.dtype = torch.double

        # Define parameter search bounds
        self.bounds = torch.tensor([
            [0.05, 24.22, 5.0, 0.0, 18.0, 510.0, 4.5],  # Lower
            [0.35, 24.50, 45.0, 2.5, 35.0, 545.0, 8.5], # Upper
        ], device=self.device, dtype=self.dtype)

        # Multi-objective Pareto reference point [RON_min, Gas_min, -Coke_max, Propylene_min]
        # In BoTorch, all objectives are framed as MAXIMIZATION
        self.ref_point = torch.tensor([92.0, 35.0, -7.5, 3.0], device=self.device, dtype=self.dtype)

        self.model = None

    def load_historical_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads accumulated experimental runs from the Parquet/JSON database."""
        if not self.store_path.exists():
            # Seed initial cold-start dataset if none exists
            self._create_cold_start_seed()

        with open(self.store_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        x_list, y_list = [], []
        for r in records:
            p = r["parameters"]
            x_vec = [
                p["zsm5_fraction"],
                p["usy_ucs_angstrom"],
                p["usy_si_al_ratio"],
                p["rare_earth_wt_pct"],
                p["matrix_alumina_wt"],
                p["riser_temp_c"],
                p["cat_to_oil_ratio"],
            ]
            obs = r["observed_metrics"]
            y_vec = [
                obs["octane_ron"],
                obs["gasoline_yield_wt"],
                -obs["coke_wt_pct"],  # Negative for maximization
                obs["propylene_yield_wt"]
            ]
            x_list.append(x_vec)
            y_list.append(y_vec)

        x_tensor = torch.tensor(x_list, device=self.device, dtype=self.dtype)
        y_tensor = torch.tensor(y_list, device=self.device, dtype=self.dtype)
        return x_tensor, y_tensor

    def _create_cold_start_seed(self, n_seed: int = 12):
        """Initial Latin Hypercube / Sobol sampling for cold-start active learning."""
        np.random.seed(42)
        records = []
        for i in range(n_seed):
            zsm5 = float(np.random.uniform(0.08, 0.30))
            ucs = float(np.random.uniform(24.24, 24.45))
            si_al = float(np.random.uniform(10.0, 40.0))
            re = float(np.random.uniform(0.05, 1.8))
            al = float(np.random.uniform(20.0, 30.0))
            temp = float(np.random.uniform(515.0, 535.0))
            co = float(np.random.uniform(5.5, 7.5))

            # Initial synthetic calibration values
            ron = 91.0 + (zsm5 * 18.0) + (24.45 - ucs) * 12.0
            gas = 48.0 - (zsm5 * 15.0) + (ucs - 24.25) * 8.0
            coke = 3.5 + (co * 0.25) + (re * 0.8)
            prop = 4.0 + (zsm5 * 14.0)

            records.append({
                "experiment_id": f"SEED-{i+1:03d}",
                "cycle": 0,
                "parameters": {
                    "zsm5_fraction": round(zsm5, 4),
                    "usy_ucs_angstrom": round(ucs, 4),
                    "usy_si_al_ratio": round(si_al, 2),
                    "rare_earth_wt_pct": round(re, 3),
                    "matrix_alumina_wt": round(al, 2),
                    "riser_temp_c": round(temp, 1),
                    "cat_to_oil_ratio": round(co, 2),
                },
                "observed_metrics": {
                    "octane_ron": round(float(ron), 2),
                    "gasoline_yield_wt": round(float(gas), 2),
                    "coke_wt_pct": round(float(coke), 2),
                    "propylene_yield_wt": round(float(prop), 2)
                }
            })

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def fit_gaussian_process(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """Fits Exact Multi-Task GP with Standardized Outcomes."""
        norm_x = normalize(train_x, self.bounds)
        self.model = SingleTaskGP(norm_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def suggest_next_batch(self, batch_size: int = 2) -> List[Dict[str, float]]:
        """
        Uses q-NEHVI acquisition function to propose a batch of formulations
        that simultaneously maximize the Pareto hypervolume and reduce epistemic uncertainty.
        """
        train_x, train_y = self.load_historical_data()
        self.fit_gaussian_process(train_x, train_y)

        norm_x = normalize(train_x, self.bounds)

        # Modern Log-scale q-Noisy Expected Hypervolume Improvement acquisition
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.ref_point,
            X_baseline=norm_x,
            prune_baseline=True
        )

        candidates_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(self.bounds.shape[1], dtype=self.dtype),
                                torch.ones(self.bounds.shape[1], dtype=self.dtype)]),
            q=batch_size,
            num_restarts=8,
            raw_samples=256
        )

        candidates = unnormalize(candidates_norm, self.bounds)
        candidate_dicts = []
        for i in range(batch_size):
            row = candidates[i].detach().cpu().numpy()
            candidate_dicts.append({
                "zsm5_fraction": float(row[0]),
                "usy_ucs_angstrom": float(row[1]),
                "usy_si_al_ratio": float(row[2]),
                "rare_earth_wt_pct": float(row[3]),
                "matrix_alumina_wt": float(row[4]),
                "riser_temp_c": float(row[5]),
                "cat_to_oil_ratio": float(row[6]),
            })

        return candidate_dicts

    def append_observation(self, exp_id: str, cycle: int, params: Dict[str, float], metrics: Dict[str, float]):
        """Persists newly acquired experimental observations to close the active learning loop."""
        with open(self.store_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        records.append({
            "experiment_id": exp_id,
            "cycle": cycle,
            "parameters": {k: round(v, 4) for k, v in params.items()},
            "observed_metrics": {k: round(v, 2) for k, v in metrics.items()}
        })

        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)


if __name__ == "__main__":
    print("=" * 80)
    print("  BAYESIAN ACTIVE LEARNING (BoTorch q-NEHVI) DISCOVERY ENGINE")
    print("=" * 80)

    store_file = Path(__file__).resolve().parent.parent.parent / "data" / "active_learning_history.json"
    bayes_opt = BayesianActiveCatalystOptimizer(store_file)

    print("Fitting Multi-Output Gaussian Process surrogate model...")
    x_hist, y_hist = bayes_opt.load_historical_data()
    print(f"Loaded {len(x_hist)} historical observations.")

    print("\nOptimizing q-NEHVI acquisition surface for batch size q = 2...")
    proposals = bayes_opt.suggest_next_batch(batch_size=2)

    print("\n[AI PROPOSED ACTIVE LEARNING EXPERIMENTAL BATCH]")
    for idx, prop in enumerate(proposals, 1):
        zsm5_pct = prop["zsm5_fraction"] * 100.0
        usy_pct = 100.0 - zsm5_pct
        print(f"Candidate #{idx}:")
        print(f"  • Formulation: {usy_pct:.1f}% USY + {zsm5_pct:.1f}% ZSM-5 Additive")
        print(f"  • USY Specs  : UCS = {prop['usy_ucs_angstrom']:.3f} A, Si/Al = {prop['usy_si_al_ratio']:.1f}, RE = {prop['rare_earth_wt_pct']:.2f}%")
        print(f"  • Riser Run  : Temp = {prop['riser_temp_c']:.1f} °C, Cat/Oil = {prop['cat_to_oil_ratio']:.2f}")
        print("-" * 80)

    print(f"Active learning history dynamically managed at: {store_file}")
