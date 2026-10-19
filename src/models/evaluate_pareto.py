"""
Evaluate and rank Pareto-optimal catalyst candidates discovered by the multi-objective optimizer.
"""
from pathlib import Path
import json
from optimizer import CatalystOptimizer

def inspect_solutions():
    data_file = Path(__file__).resolve().parent.parent.parent / "data" / "catalysts_seed.json"
    opt = CatalystOptimizer(data_file)
    study = opt.optimize_discovery(n_trials=200)

    # Sort trials by high octane and low coke
    trials = study.best_trials
    sorted_trials = sorted(trials, key=lambda t: t.values[0] - (t.values[1] * 0.5), reverse=True)

    print("=" * 75)
    print("        TOP MULTI-OBJECTIVE PARETO CATALYST FORMULATIONS")
    print("=" * 75)
    records = []
    for idx, t in enumerate(sorted_trials[:10], 1):
        ron, coke, gas = t.values
        p = t.params
        print(f"[{idx}] Octane (RON): {ron:5.2f} | Coke: {coke:4.2f}% | Gasoline: {gas:5.2f}%")
        print(f"    Zeolite: {p.get('zeolite_type')} | Si/Al: {p.get('si_al_ratio', 0):.1f} | UCS: {p.get('unit_cell_size', 0):.3f} A | RE: {p.get('rare_earth_wt', 0):.2f}%")
        print(f"    Bronsted Acidity: {p.get('bronsted_acid', 0):.1f} umol/g | Lewis Acidity: {p.get('lewis_acid', 0):.1f} umol/g")
        print("-" * 75)
        records.append({
            "rank": idx,
            "predicted_ron": round(ron, 2),
            "predicted_coke_wt": round(coke, 2),
            "predicted_gasoline_yield": round(gas, 2),
            "params": p
        })

    out_file = data_file.parent / "pareto_top_candidates.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved top ranked formulations to {out_file}")

if __name__ == "__main__":
    inspect_solutions()
