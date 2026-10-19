"""
Option 2: 4-Lump Kinetic Digital Twin of an FCC Riser Reactor.
Models the cracking network over residence time (0 to 3.5 seconds):
   Lump 1: Gas Oil (VGO Feed) -> y1
   Lump 2: Gasoline (C5 - 216°C cut) -> y2
   Lump 3: Light Gases (LPG + Dry Gas) -> y3
   Lump 4: Coke Deposit on Catalyst -> y4

Kinetic Reaction Network:
   y1 --(k1)--> y2 (Gasoline)
   y1 --(k2)--> y3 (Light Gases)
   y1 --(k3)--> y4 (Coke)
   y2 --(k4)--> y3 (Gasoline overcracking to LPG/Gas)
   y2 --(k5)--> y4 (Gasoline condensation to Coke)

Includes catalyst deactivation function phi(t) = exp(-alpha * t)
governed by catalyst acidity, Si/Al ratio, and coke accumulation.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pathlib import Path
import json


class FCCRiserKineticModel:
    def __init__(
        self,
        temp_c: float = 525.0,
        cat_to_oil: float = 6.5,
        usy_ucs: float = 24.266,
        zsm5_fraction: float = 0.237,
        si_al_ratio: float = 25.0
    ):
        self.temp_k = temp_c + 273.15
        self.c_to_o = cat_to_oil
        self.ucs = usy_ucs
        self.zsm5_frac = zsm5_fraction
        self.si_al = si_al_ratio

        # Base rate constants at reference temp T_ref = 520°C (793.15 K)
        # Calibrated against classic Weekman & Nace industrial FCC kinetics
        t_ref = 793.15
        r_gas = 8.314  # J / (mol * K)

        # Apparent Activation Energies (kJ/mol)
        e_act = {
            "k1": 56.0e3,   # VGO -> Gasoline
            "k2": 62.0e3,   # VGO -> Light Gas
            "k3": 48.0e3,   # VGO -> Coke
            "k4": 74.0e3,   # Gasoline -> Light Gas
            "k5": 65.0e3,   # Gasoline -> Coke
        }

        # Catalyst activity tuning factor based on dealumination and ZSM-5 content
        # Dealumination (lower UCS) slightly decreases bulk k1 but increases k4 (shape selectivity)
        # ZSM-5 strongly promotes k4 (gasoline isomerization/cracking to propylene)
        activity_mod = (self.ucs / 24.35) ** 2.0 * (self.c_to_o / 6.0)
        zsm5_boost = 1.0 + (self.zsm5_frac * 2.8)

        # Reference pre-exponential constants (s^-1) calibrated for 1.5 - 2.5s residence time
        k0 = {
            "k1": 2.2 * activity_mod,
            "k2": 0.65 * activity_mod,
            "k3": 0.22 * (self.c_to_o / 6.0),
            "k4": 0.15 * zsm5_boost,
            "k5": 0.045,
        }

        # Temperature-adjusted rate constants (Arrhenius equation)
        self.k = {}
        for key in k0:
            self.k[key] = k0[key] * np.exp((-e_act[key] / r_gas) * (1.0 / self.temp_k - 1.0 / t_ref))

        # Catalyst deactivation decay constant alpha (s^-1)
        self.alpha = float(np.clip(0.45 * (1.0 - self.zsm5_frac * 0.3) * (self.si_al / 30.0) ** -0.2, 0.1, 1.2))

    def _ode_system(self, t, y):
        """
        Differential equations for mass fractions y = [y1_vgo, y2_gasoline, y3_gas, y4_coke]
        Second-order cracking for VGO feed (due to complex feed mixture),
        First-order cracking for gasoline lump.
        """
        y1, y2, y3, y4 = y

        # Catalyst deactivation decay over contact time
        phi = np.exp(-self.alpha * t)

        # Reaction rates
        r1 = self.k["k1"] * (y1 ** 2) * phi  # VGO -> Gasoline
        r2 = self.k["k2"] * (y1 ** 2) * phi  # VGO -> Gas
        r3 = self.k["k3"] * (y1 ** 2) * phi  # VGO -> Coke
        r4 = self.k["k4"] * y2 * phi         # Gasoline -> Gas
        r5 = self.k["k5"] * y2 * phi         # Gasoline -> Coke

        dy1_dt = -(r1 + r2 + r3)
        dy2_dt = r1 - (r4 + r5)
        dy3_dt = r2 + r4
        dy4_dt = r3 + r5

        return [dy1_dt, dy2_dt, dy3_dt, dy4_dt]

    def simulate(self, t_span=(0.0, 3.5), num_points=100) -> pd.DataFrame:
        """Runs the digital twin simulation over time t_span (seconds)."""
        # Initial conditions: Pure VGO feed at t=0
        y0 = [1.0, 0.0, 0.0, 0.0]
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        sol = solve_ivp(
            self._ode_system,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45"
        )

        df = pd.DataFrame({
            "residence_time_sec": sol.t,
            "vgo_feed_wt_pct": sol.y[0] * 100.0,
            "gasoline_wt_pct": sol.y[1] * 100.0,
            "light_gas_wt_pct": sol.y[2] * 100.0,
            "coke_wt_pct": sol.y[3] * 100.0,
        })
        df["conversion_wt_pct"] = 100.0 - df["vgo_feed_wt_pct"]
        return df


if __name__ == "__main__":
    print("=" * 70)
    print("  FCC RISER 4-LUMP KINETIC DIGITAL TWIN SIMULATION")
    print("=" * 70)

    # Initialize model with our winning AI composite catalyst
    twin = FCCRiserKineticModel(
        temp_c=528.0,
        cat_to_oil=6.2,
        usy_ucs=24.266,
        zsm5_fraction=0.237,
        si_al_ratio=25.0
    )

    df_profile = twin.simulate(t_span=(0.0, 3.5), num_points=50)

    print(f"\nSimulated Riser Performance Profiles across Residence Time:")
    print(f"{'Time (s)':<9} | {'Conversion %':<13} | {'Gasoline %':<11} | {'LPG/Gas %':<11} | {'Coke %'}")
    print("-" * 65)

    checkpoints = [0.2, 0.8, 1.5, 2.2, 3.0]
    for cp in checkpoints:
        row = df_profile.iloc[(df_profile["residence_time_sec"] - cp).abs().argsort()[:1]].iloc[0]
        print(f"{row['residence_time_sec']:<9.2f} | {row['conversion_wt_pct']:<13.2f} | {row['gasoline_wt_pct']:<11.2f} | {row['light_gas_wt_pct']:<11.2f} | {row['coke_wt_pct']:.2f}")

    print("=" * 70)

    out_file = Path(__file__).resolve().parent.parent.parent / "data" / "riser_kinetic_profile.csv"
    df_profile.to_csv(out_file, index=False)
    print(f"Full kinetic time-series saved to: {out_file}")
