"""
Module 2: FCC Regenerator Fluidized-Bed Combustion & Hydrothermal Aging Loop.
Models the continuous industrial FCC closed loop:
   [ Riser Cracking (510 - 540°C) ]  ---> Spent Catalyst with Coke Deposit (Cc_spent ~ 0.8 - 1.2 wt%)
               ^                                   |
               | Regenerated Catalyst              v
   [ Regenerator Combustion (680 - 760°C) ] <--- Stripper (Steam removes entrained HC)

Physics Modeled:
  1. Dense-bed Coke Combustion Kinetics (Carbon burn & CO/CO2 ratio):
       C + 0.5 O2 -> CO   (Partial combustion)
       CO + 0.5 O2 -> CO2 (Afterburning in dilute phase)
       C + O2 -> CO2      (Direct combustion)
  2. Regenerator Exothermic Heat Balance:
       T_reg = T_spent + [ Delta_H_combustion * (Cc_spent - Cc_regen) ] / (C_p * C/O)
  3. Multi-Cycle Hydrothermal Zeolite Dealumination Aging (Steaming at 740 - 780°C):
       - Loss of Unit Cell Size (UCS) over equilibrium cycles:
           UCS(N) = UCS_inf + (UCS_fresh - UCS_inf) * exp(-k_steam * N_cycles)
       - Loss of Framework Micropore Surface Area (Zeolite Dealumination)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import json


class FCCRegeneratorAgingLoop:
    """
    Simulates coke combustion in the regenerator dense bed and multi-cycle hydrothermal steaming.
    """

    def __init__(
        self,
        regenerator_temp_c: float = 720.0,
        air_blower_rate_kg_s: float = 45.0,
        steam_partial_pressure_bar: float = 0.25,
        target_residual_coke_wt: float = 0.05
    ):
        self.t_reg_c = regenerator_temp_c
        self.t_reg_k = regenerator_temp_c + 273.15
        self.air_rate = air_blower_rate_kg_s
        self.p_steam = steam_partial_pressure_bar
        self.coke_regen_target = target_residual_coke_wt

        # Exothermic Heat of Combustion (J / kg coke burned)
        # Average CO2/CO molar ratio ~ 3.5: Heat release ~ 32.5 MJ / kg coke
        self.delta_h_combustion = 32.5e6  # J / kg coke
        self.catalyst_heat_capacity = 1150.0 # J / (kg_cat * K)

    def calculate_regenerator_state(
        self,
        riser_exit_temp_c: float,
        coke_on_spent_cat_wt_pct: float,
        cat_to_oil_ratio: float = 6.5
    ) -> Dict[str, Any]:
        """
        Computes dense bed temperature rise from burning spent coke deposit.
        """
        # Coke burned per kg of circulating catalyst
        delta_coke_fraction = max(0.001, (coke_on_spent_cat_wt_pct - self.coke_regen_target) / 100.0)

        # Thermal heat release per kg of catalyst
        heat_released = delta_coke_fraction * self.delta_h_combustion

        # Temperature lift in regenerator: Delta_T = Q / Cp_cat
        delta_t_reg = heat_released / self.catalyst_heat_capacity
        steady_state_t_reg = riser_exit_temp_c + delta_t_reg

        # Flue Gas Composition (CO2, CO, O2, N2)
        # Flue gas volume based on air stoichiometry
        o2_consumed_kg = delta_coke_fraction * 2.45
        co2_generated_kg = delta_coke_fraction * 3.42
        co_generated_kg = delta_coke_fraction * 0.24

        return {
            "spent_catalyst_coke_wt_pct": round(coke_on_spent_cat_wt_pct, 3),
            "regenerated_catalyst_coke_wt_pct": round(self.coke_regen_target, 3),
            "regenerator_dense_bed_temp_c": round(steady_state_t_reg, 1),
            "combustion_temperature_lift_c": round(delta_t_reg, 1),
            "flue_gas_co2_kg_per_ton_cat": round(co2_generated_kg * 1000.0, 2),
            "co2_to_co_ratio": 14.2,
        }

    @classmethod
    def simulate_hydrothermal_aging(
        cls,
        fresh_ucs_angstrom: float,
        fresh_zeolite_sa_m2_g: float,
        num_cycles: int = 50,
        steaming_temp_c: float = 760.0,
        rare_earth_wt_pct: float = 0.0
    ) -> pd.DataFrame:
        """
        Simulates long-term hydrothermal dealumination towards Equilibrium Catalyst (E-cat).
        FAU framework aluminum is knocked out of tetrahedral T-sites by high-temperature steam.
        Rare Earth (RE) promoters electrostatically lock bridged Al-O-Si bonds, retarding dealumination.
        """
        # Asymptotic limits for equilibrium catalyst (E-cat) stabilized by Lanthanum / Rare Earths
        ucs_inf = 24.238 + min(0.045, 0.016 * rare_earth_wt_pct)
        sa_retention_inf = 0.55 + min(0.15, 0.05 * rare_earth_wt_pct)

        # Steaming rate constant governed by Arrhenius temperature
        # Higher regenerator steaming temp accelerates dealumination
        t_k = steaming_temp_c + 273.15
        k_steam = 0.045 * np.exp(-45.0e3 / 8.314 * (1.0 / t_k - 1.0 / 1033.15))

        history = []
        for cycle in range(num_cycles + 1):
            # Exponential decay of Unit Cell Size
            decay = np.exp(-k_steam * cycle)
            current_ucs = ucs_inf + (fresh_ucs_angstrom - ucs_inf) * decay
            current_sa = fresh_zeolite_sa_m2_g * (sa_retention_inf + (1.0 - sa_retention_inf) * decay)

            # Framework aluminum (Breck-Flanigen)
            n_al = (current_ucs - 24.191) / 0.00868

            history.append({
                "aging_cycle": cycle,
                "unit_cell_size_angstrom": round(float(current_ucs), 3),
                "framework_al_per_uc": round(float(n_al), 2),
                "zeolite_surface_area_m2_g": round(float(current_sa), 1),
                "activity_retention_pct": round(float(decay * 100.0), 1)
            })

        return pd.DataFrame(history)

    @classmethod
    def get_equilibrium_ecat_properties(
        cls,
        fresh_ucs_angstrom: float,
        fresh_zeolite_sa_m2_g: float = 480.0,
        rare_earth_wt_pct: float = 0.15,
        steaming_temp_c: float = 760.0
    ) -> Dict[str, float]:
        """
        Returns steady-state Equilibrium Catalyst (E-cat) properties after 50 cycles of steaming.
        """
        df_aged = cls.simulate_hydrothermal_aging(
            fresh_ucs_angstrom=fresh_ucs_angstrom,
            fresh_zeolite_sa_m2_g=fresh_zeolite_sa_m2_g,
            num_cycles=50,
            steaming_temp_c=steaming_temp_c,
            rare_earth_wt_pct=rare_earth_wt_pct
        )
        ecat = df_aged.iloc[-1]
        return {
            "ecat_ucs_angstrom": float(ecat["unit_cell_size_angstrom"]),
            "ecat_surface_area_m2_g": float(ecat["zeolite_surface_area_m2_g"]),
            "ecat_framework_al": float(ecat["framework_al_per_uc"]),
            "activity_retention_pct": float(ecat["activity_retention_pct"])
        }


if __name__ == "__main__":
    print("=" * 80)
    print("  FCC REGENERATOR FLUIDIZED COMBUSTION & HYDROTHERMAL AGING")
    print("=" * 80)

    loop = FCCRegeneratorAgingLoop(regenerator_temp_c=720.0)

    reg_state = loop.calculate_regenerator_state(
        riser_exit_temp_c=520.0,
        coke_on_spent_cat_wt_pct=0.92,
        cat_to_oil_ratio=6.5
    )

    print(f"Regenerator Fluidized Combustion Performance:")
    print(f"  • Spent Cat Coke Deposit      : {reg_state['spent_catalyst_coke_wt_pct']} wt%")
    print(f"  • Regenerated Cat Carbon      : {reg_state['regenerated_catalyst_coke_wt_pct']} wt%")
    print(f"  • Heat Release Temperature Rise: +{reg_state['combustion_temperature_lift_c']} °C")
    print(f"  • Steady-State Bed Temperature: {reg_state['regenerator_dense_bed_temp_c']} °C")
    print(f"  • Flue Gas CO2 Generated      : {reg_state['flue_gas_co2_kg_per_ton_cat']} kg CO2 / ton catalyst")

    print(f"\nSimulating 50-Cycle Hydrothermal Steaming Aging (ASTM D4463 @ 760 °C):")
    df_aging = FCCRegeneratorAgingLoop.simulate_hydrothermal_aging(
        fresh_ucs_angstrom=24.320,
        fresh_zeolite_sa_m2_g=280.0,
        num_cycles=40,
        steaming_temp_c=760.0
    )

    print(f"{'Cycle':<7} | {'UCS (Å)':<9} | {'Framework N_Al':<16} | {'Zeolite SA (m²/g)':<18} | {'Activity Retention'}")
    print("-" * 75)
    for c in [0, 5, 15, 30, 40]:
        row = df_aging[df_aging["aging_cycle"] == c].iloc[0]
        print(f"#{row['aging_cycle']:<6} | {row['unit_cell_size_angstrom']:<9.3f} | {row['framework_al_per_uc']:<16.2f} | {row['zeolite_surface_area_m2_g']:<18.1f} | {row['activity_retention_pct']:.1f}%")

    print("=" * 80)

    out_file = Path(__file__).resolve().parent.parent.parent / "data" / "catalyst_hydrothermal_aging_profile.csv"
    df_aging.to_csv(out_file, index=False)
    print(f"Saved hydrothermal aging profile to: {out_file}")
