"""
Inspect the full Pareto front distribution of the composite catalyst optimizer.
"""
from pathlib import Path
import json
from composite_optimizer import optimize_composite_catalyst

def analyze_trade_offs():
    study = optimize_composite_catalyst(n_trials=250)
    trials = study.best_trials

    print(f"Total Pareto-optimal solutions discovered: {len(trials)}\n")
    print(f"{'Rank':<5} | {'RON':<7} | {'Coke wt%':<9} | {'Gasoline%':<10} | {'ZSM-5%':<7} | {'USY UCS':<8} | {'RE wt%'}")
    print("-" * 65)

    # Sort by Octane descending
    sorted_by_ron = sorted(trials, key=lambda t: t.values[0], reverse=True)
    for idx, t in enumerate(sorted_by_ron[:12], 1):
        ron, coke, gas = t.values
        p = t.params
        zsm5_pct = p["zsm5_fraction"] * 100.0
        ucs = p["usy_ucs"]
        re_wt = p["usy_re_wt"]
        print(f"#{idx:<4} | {ron:6.2f}  | {coke:6.2f}%   | {gas:6.2f}%    | {zsm5_pct:5.1f}% | {ucs:6.3f} A | {re_wt:5.2f}%")

    print("\n" + "=" * 65)
    # Find candidates with the lowest coke that maintain RON >= 94.0
    low_coke_cand = [t for t in trials if t.values[0] >= 94.0]
    low_coke_cand.sort(key=lambda t: t.values[1])
    print("Lowest Coke Formulations achieving Octane >= 94.0:")
    for idx, t in enumerate(low_coke_cand[:5], 1):
        ron, coke, gas = t.values
        p = t.params
        print(f"  • Option #{idx}: RON = {ron:.2f}, Coke = {coke:.2f}%, Gasoline = {gas:.2f}%, Blend = {100-p['zsm5_fraction']*100:.1f}% USY + {p['zsm5_fraction']*100:.1f}% ZSM-5")

if __name__ == "__main__":
    analyze_trade_offs()
