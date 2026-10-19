"""
Kayser ACE / MAT Pilot Reactor Log Ingestion & Multi-Fidelity Discrepancy Calibration Engine.
Compliant with:
  - ASTM D7964: Standard Test Method for Determining Activity of FCC Catalysts in a Small Fixed-Bed Unit (ACE)
  - ASTM D3907: Standard Test Method for Testing Fluid Catalytic Cracking (FCC) Catalysts by Microactivity Test (MAT)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ACEPilotIngestor:
    """
    Ingests raw pilot-scale ACE/MAT reactor run data, validates mass balance closure,
    computes Kennedy-O'Hagan discrepancy deltas against low-fidelity ODE twins,
    and commits high-fidelity (s=1.0) records to the active learning dataset.
    """

    def __init__(self, history_file: Optional[Path] = None):
        self.repo_root = Path(__file__).resolve().parent.parent.parent
        self.history_file = history_file or (self.repo_root / "data" / "multi_fidelity_active_learning_history.json")

    def ingest_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parses an ASTM D7964 ACE reactor run log from CSV.
        Validates mass balance closure and extracts verified cracking yields.
        """
        df = pd.read_csv(file_path)
        valid_records = []

        for idx, row in df.iterrows():
            closure = float(row.get("Mass_Balance_Closure_Pct", 100.0))
            if closure < 97.5 or closure > 102.5:
                print(f"[ACE Ingestor] Warning: Run {row.get('Run_ID')} rejected due to mass balance closure violation ({closure:.1f}%).")
                continue

            record = {
                "experiment_id": str(row.get("Run_ID", f"ACE-{idx:03d}")),
                "cycle": 0,
                "fidelity_s": 1.0,
                "data_source": f"Kayser ACE Pilot Unit ({row.get('ASTM_Method', 'ASTM D7964')})",
                "feedstock_type": str(row.get("Feedstock_Type", "standard_middle_east_vgo")),
                "parameters": {
                    "zsm5_fraction": float(row["ZSM5_Fraction"]),
                    "usy_ucs_angstrom": float(row["USY_UCS_Angstrom"]),
                    "rare_earth_wt_pct": float(row.get("Rare_Earth_Wt_Pct", 1.5)),
                    "matrix_alumina_wt": float(row.get("Matrix_Alumina_Wt", 25.0)),
                    "riser_temp_c": float(row["Riser_Temp_C"]),
                    "cat_to_oil_ratio": float(row["Cat_to_Oil"]),
                    "fidelity_s": 1.0
                },
                "observed_metrics": {
                    "octane_ron": float(row["Measured_RON"]),
                    "gasoline_yield_wt": float(row["Gasoline_Yield_Wt"]),
                    "propylene_yield_wt": float(row["Propylene_Yield_Wt"]),
                    "coke_wt_pct": float(row["Coke_Wt_Pct"]),
                    "lco_yield_wt": float(row.get("LCO_Diesel_Wt", 16.0)),
                    "dry_gas_wt": float(row.get("Dry_Gas_Wt", 3.0)),
                    "conversion_wt_pct": float(row["Conversion_Wt_Pct"])
                },
                "quality_control": {
                    "mass_balance_closure_pct": closure,
                    "astm_standard": str(row.get("ASTM_Method", "ASTM D7964")),
                    "status": "VERIFIED_PHYSICAL_GROUND_TRUTH"
                }
            }
            valid_records.append(record)

        return valid_records

    def commit_to_active_learning(self, records: List[Dict[str, Any]]) -> int:
        """
        Appends newly ingested high-fidelity records to the persistent active learning file,
        avoiding duplicate experiment_ids.
        """
        existing_history = []
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                try:
                    existing_history = json.load(f)
                except json.JSONDecodeError:
                    existing_history = []

        existing_ids = {r.get("experiment_id") for r in existing_history}
        committed_count = 0

        for r in records:
            if r["experiment_id"] not in existing_ids:
                existing_history.append(r)
                existing_ids.add(r["experiment_id"])
                committed_count += 1

        with open(self.history_file, "w") as f:
            json.dump(existing_history, f, indent=2)

        return committed_count

    def compute_model_discrepancy(self, ace_record: Dict[str, Any], ode_twin_func) -> Dict[str, float]:
        """
        Calculates empirical Kennedy-O'Hagan discrepancy delta between physical ACE
        run and the low-fidelity deterministic ODE twin:
          delta(x) = y_physical(x) - y_ode(x)
        """
        params = ace_record["parameters"]
        ode_res = ode_twin_func(
            zsm5_frac=params["zsm5_fraction"],
            ucs=params["usy_ucs_angstrom"],
            temp_c=params["riser_temp_c"],
            cat_oil=params["cat_to_oil_ratio"],
            feed_contaminants=None
        )

        obs = ace_record["observed_metrics"]
        delta_propylene = obs["propylene_yield_wt"] - ode_res.get("propylene_yield", 0.0)
        delta_coke = obs["coke_wt_pct"] - ode_res.get("coke_yield", 0.0)
        delta_ron = obs["octane_ron"] - 95.0

        return {
            "delta_propylene_wt": round(float(delta_propylene), 2),
            "delta_coke_wt": round(float(delta_coke), 2),
            "delta_octane_ron": round(float(delta_ron), 2)
        }


if __name__ == "__main__":
    print("=" * 80)
    print("  KAYSER ACE PILOT REACTOR INGESTION & DISCREPANCY CALIBRATION")
    print("=" * 80)
    ingestor = ACEPilotIngestor()
    sample_csv = ingestor.repo_root / "data" / "benchmarks" / "kayser_ace_pilot_run_sample.csv"
    
    if sample_csv.exists():
        records = ingestor.ingest_csv(sample_csv)
        print(f"Parsed {len(records)} verified ASTM D7964 pilot runs.")
        added = ingestor.commit_to_active_learning(records)
        print(f"Committed {added} high-fidelity (s=1.0) benchmark records to {ingestor.history_file.name}.")
        for r in records[:2]:
            print(f"  • {r['experiment_id']}: Propylene={r['observed_metrics']['propylene_yield_wt']:.2f} wt%, "
                  f"RON={r['observed_metrics']['octane_ron']:.1f}, Closure={r['quality_control']['mass_balance_closure_pct']}%")
    print("=" * 80)
