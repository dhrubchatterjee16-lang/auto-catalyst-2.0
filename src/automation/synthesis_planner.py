"""
Autonomous Synthesis Protocol Planner for Auto-Catalyst 2.0.
Translates optimized catalyst formulations (e.g. USY + ZSM-5, Si/Al, Rare-Earth, Matrix Alumina)
into precise physical laboratory synthesis recipes and Opentrons OT-2 executable Python protocols.

Handles:
  1. Stoichiometric mass & volume calculations for:
     - Zeolite USY & ZSM-5 slurry suspensions
     - Pseudoboehmite active alumina sol (peptized with HNO3)
     - Kaolin clay matrix slurry
     - Lanthanum / Rare-Earth nitrate promoter precursor solutions
  2. Sequential synthesis workflow:
     - Wet slurry blending & homogenizing
     - Ultrasonic peptization & sol-gel aging
     - Spray-drying / droplet micro-pelletizing simulation
     - Programmed thermal calcination ramp profile
  3. Generation of runnable Opentrons OT-2 Python protocol script (API v2).
"""

import json
from pathlib import Path
from typing import Dict, Any


class CatalystSynthesisPlanner:
    """
    Translates an AI-designed catalyst recipe into wet-chemistry instructions
    and Opentrons robotic liquid-dispensing actions.
    """

    def __init__(self, target_batch_mass_grams: float = 10.0):
        self.batch_mass_g = target_batch_mass_grams

    def calculate_recipe(self, formulation: Dict[str, Any]) -> Dict[str, Any]:
        params = formulation.get("params", formulation)

        # Zeolite fractions
        zsm5_frac = params.get("zsm5_fraction", 0.237)
        usy_frac = 1.0 - zsm5_frac

        # Catalyst solid composition (typical industrial split)
        # Total active zeolites: 40 wt% of final catalyst particle
        # Active matrix alumina: 25 wt%
        # Kaolin clay binder: 35 wt% - Rare Earth wt%
        zeolite_total_mass = self.batch_mass_g * 0.40
        usy_solid_g = zeolite_total_mass * usy_frac
        zsm5_solid_g = zeolite_total_mass * zsm5_frac

        re_wt_pct = params.get("usy_re_wt", 0.1)
        re_mass_g = self.batch_mass_g * (re_wt_pct / 100.0)

        matrix_alumina_g = self.batch_mass_g * 0.25
        kaolin_clay_g = self.batch_mass_g * 0.35 - re_mass_g

        # Stock slurry concentrations (g/mL for automated pipetting on OT-2)
        # USY slurry: 0.20 g/mL in deionized H2O
        # ZSM-5 slurry: 0.20 g/mL in deionized H2O
        # Alumina sol (peptized with 0.1M HNO3): 0.15 g/mL
        # Kaolin clay suspension: 0.25 g/mL
        # La(NO3)3 stock solution: 0.05 g La/mL
        usy_slurry_vol_ml = usy_solid_g / 0.20
        zsm5_slurry_vol_ml = zsm5_solid_g / 0.20
        alumina_sol_vol_ml = matrix_alumina_g / 0.15
        kaolin_slurry_vol_ml = kaolin_clay_g / 0.25
        re_stock_vol_ml = max(0.05, re_mass_g / 0.05) if re_mass_g > 0 else 0.0

        # Thermal calcination profile
        # Ramp 1: Desorption of moisture & templates at 120°C for 2h
        # Ramp 2: Dehydroxylation and binder crystallization at 550°C for 4h
        calcination_profile = {
            "stage_1_drying": {"target_temp_c": 120, "ramp_rate_c_min": 2.5, "hold_time_hours": 2.0, "gas_env": "Air"},
            "stage_2_calcination": {"target_temp_c": 550, "ramp_rate_c_min": 5.0, "hold_time_hours": 4.0, "gas_env": "Air / N2 sweep"},
            "stage_3_steaming_activation": {"target_temp_c": 760, "steam_partial_pressure_bar": 1.0, "duration_hours": 2.0, "purpose": "Hydrothermal dealumination to reach target UCS"}
        }

        recipe = {
            "batch_target_dry_mass_grams": self.batch_mass_g,
            "solid_components_grams": {
                "USY_zeolite_dry_g": round(usy_solid_g, 3),
                "ZSM5_zeolite_dry_g": round(zsm5_solid_g, 3),
                "alumina_matrix_dry_g": round(matrix_alumina_g, 3),
                "kaolin_clay_dry_g": round(kaolin_clay_g, 3),
                "lanthanum_promoter_g": round(re_mass_g, 4),
            },
            "liquid_stock_dispensing_ml": {
                "USY_slurry_20pct_ml": round(usy_slurry_vol_ml, 2),
                "ZSM5_slurry_20pct_ml": round(zsm5_slurry_vol_ml, 2),
                "alumina_sol_15pct_ml": round(alumina_sol_vol_ml, 2),
                "kaolin_suspension_25pct_ml": round(kaolin_slurry_vol_ml, 2),
                "rare_earth_solution_ml": round(re_stock_vol_ml, 2),
            },
            "thermal_treatment_profile": calcination_profile,
        }
        return recipe

    def generate_ot2_protocol(self, recipe: Dict[str, Any], output_py_path: Path):
        """Generates a standalone Opentrons Python protocol (API v2)."""
        vols = recipe["liquid_stock_dispensing_ml"]

        protocol_code = f'''# -*- coding: utf-8 -*-
"""
Opentrons OT-2 Automated Synthesis Protocol for Auto-Catalyst 2.0
Batch: {recipe["batch_target_dry_mass_grams"]}g Composite Catalyst (USY + ZSM-5)
Target Liquid Dispensing:
  - USY Slurry: {vols["USY_slurry_20pct_ml"]} mL
  - ZSM-5 Slurry: {vols["ZSM5_slurry_20pct_ml"]} mL
  - Alumina Sol: {vols["alumina_sol_15pct_ml"]} mL
  - Kaolin Clay: {vols["kaolin_suspension_25pct_ml"]} mL
  - RE Promoter: {vols["rare_earth_solution_ml"]} mL
"""

metadata = {{
    'protocolName': 'Auto-Catalyst 2.0: FCC Dual-Zeolite Composite Slurry Prep',
    'author': 'Antigravity Autonomous Lab System',
    'description': 'Automated dispensing and homogenization of dual-zeolite FCC catalyst precursors',
    'apiLevel': '2.14'
}}

def run(protocol):
    # Labware setup
    # Slot 1: Reservoir containing reagent stock solutions (15mL Falcon tubes)
    reservoir = protocol.load_labware('opentrons_6_tuberack_falcon_50ml', '1')
    # Slot 2: Mixing/reaction vessel (temperature-controlled or shaker module)
    reaction_vessel = protocol.load_labware('opentrons_6_tuberack_falcon_50ml', '2')
    # Slot 3: Tip rack
    tiprack_1000 = protocol.load_labware('opentrons_96_tiprack_1000ul', '3')

    # Pipette setup (1000 uL single channel)
    p1000 = protocol.load_instrument('p1000_single_gen2', 'right', tip_racks=[tiprack_1000])

    # Reagent stock tube positions
    usy_stock = reservoir.wells_by_name()['A1']
    zsm5_stock = reservoir.wells_by_name()['A2']
    alumina_stock = reservoir.wells_by_name()['A3']
    kaolin_stock = reservoir.wells_by_name()['B1']
    re_stock = reservoir.wells_by_name()['B2']

    # Target synthesis reactor vessel
    target_reactor = reaction_vessel.wells_by_name()['A1']

    # Step 1: Dispense Alumina Sol Matrix (Base peptized binder)
    protocol.comment("Step 1: Dispensing Alumina Sol Matrix...")
    p1000.pick_up_tip()
    alumina_ul = {int(vols["alumina_sol_15pct_ml"] * 1000)}
    while alumina_ul > 0:
        transfer_vol = min(alumina_ul, 1000)
        p1000.aspirate(transfer_vol, alumina_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        alumina_ul -= transfer_vol
    p1000.drop_tip()

    # Step 2: Dispense Kaolin Clay Suspension
    protocol.comment("Step 2: Dispensing Kaolin Clay Suspension...")
    p1000.pick_up_tip()
    kaolin_ul = {int(vols["kaolin_suspension_25pct_ml"] * 1000)}
    while kaolin_ul > 0:
        transfer_vol = min(kaolin_ul, 1000)
        p1000.aspirate(transfer_vol, kaolin_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        kaolin_ul -= transfer_vol
    p1000.drop_tip()

    # Step 3: Dispense USY Zeolite Slurry
    protocol.comment("Step 3: Dispensing USY Zeolite Slurry...")
    p1000.pick_up_tip()
    p1000.mix(3, 800, usy_stock.bottom(5))  # Re-suspend zeolite before transfer
    usy_ul = {int(vols["USY_slurry_20pct_ml"] * 1000)}
    while usy_ul > 0:
        transfer_vol = min(usy_ul, 1000)
        p1000.aspirate(transfer_vol, usy_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        usy_ul -= transfer_vol
    p1000.drop_tip()

    # Step 4: Dispense ZSM-5 Additive Slurry
    protocol.comment("Step 4: Dispensing ZSM-5 Additive Slurry...")
    p1000.pick_up_tip()
    p1000.mix(3, 800, zsm5_stock.bottom(5))
    zsm5_ul = {int(vols["ZSM5_slurry_20pct_ml"] * 1000)}
    while zsm5_ul > 0:
        transfer_vol = min(zsm5_ul, 1000)
        p1000.aspirate(transfer_vol, zsm5_stock.bottom(2))
        p1000.dispense(transfer_vol, target_reactor.top(-5))
        zsm5_ul -= transfer_vol
    p1000.drop_tip()

    # Step 5: Dispense Rare-Earth Promoter
    re_ul = {int(vols["rare_earth_solution_ml"] * 1000)}
    if re_ul > 0:
        protocol.comment("Step 5: Dispensing Rare-Earth Promoter...")
        p1000.pick_up_tip()
        while re_ul > 0:
            transfer_vol = min(re_ul, 1000)
            p1000.aspirate(transfer_vol, re_stock.bottom(2))
            p1000.dispense(transfer_vol, target_reactor.top(-5))
            re_ul -= transfer_vol
        p1000.drop_tip()

    # Final Homogenization mixing
    protocol.comment("Step 6: Performing Final Pre-Gel Homogenization...")
    p1000.pick_up_tip()
    p1000.mix(5, 900, target_reactor.bottom(10))
    p1000.drop_tip()

    protocol.comment("Automated composite slurry preparation complete. Transfer to aging and spray-drying.")
'''
        with open(output_py_path, "w", encoding="utf-8") as f:
            f.write(protocol_code)
        print(f"Opentrons OT-2 executable Python protocol generated at: {output_py_path}")


if __name__ == "__main__":
    planner = CatalystSynthesisPlanner(target_batch_mass_grams=10.0)

    # Use winning formulation discovered by our composite optimizer
    best_formulation = {
        "params": {
            "zsm5_fraction": 0.237,
            "usy_si_al": 20.0,
            "usy_ucs": 24.266,
            "usy_re_wt": 0.15,
            "matrix_alumina_wt": 25.0
        }
    }

    recipe = planner.calculate_recipe(best_formulation)

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    protocols_dir = Path(__file__).resolve().parent.parent.parent / "src" / "automation"
    protocols_dir.mkdir(parents=True, exist_ok=True)

    recipe_out = data_dir / "synthesis_recipe_execution.json"
    with open(recipe_out, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
    print(f"Synthesis recipe JSON saved to: {recipe_out}")

    ot2_script = protocols_dir / "ot2_synthesis_protocol.py"
    planner.generate_ot2_protocol(recipe, ot2_script)
