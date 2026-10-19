"""
Seed dataset generator for FCC catalyst performance modeling.
Generates realistic, physically consistent catalyst samples covering
various zeolite frameworks (USY, ZSM-5, Beta), Si/Al ratios, acidities, and yields.
"""

import json
import random
import numpy as np
from pathlib import Path


def generate_benchmark_catalysts(num_samples: int = 250, seed: int = 42) -> list:
    np.random.seed(seed)
    random.seed(seed)

    records = []
    frameworks = [
        {"type": "FAU", "zeolite": "USY", "base_ucs": 24.30, "base_ron": 91.5},
        {"type": "FAU", "zeolite": "RE-USY", "base_ucs": 24.50, "base_ron": 90.0},
        {"type": "MFI", "zeolite": "ZSM-5", "base_ucs": 24.25, "base_ron": 96.0},
        {"type": "BEA", "zeolite": "Beta", "base_ucs": 24.35, "base_ron": 93.5},
    ]

    for i in range(num_samples):
        fw = random.choice(frameworks)
        si_al = float(np.random.uniform(5.0, 60.0))
        re_wt = float(np.random.uniform(0.0, 3.5)) if fw["zeolite"] == "RE-USY" else float(np.random.uniform(0.0, 0.5))
        matrix_al = float(np.random.uniform(15.0, 40.0))
        binder_clay = float(np.random.uniform(20.0, 45.0))
        ucs = float(fw["base_ucs"] + np.random.normal(0, 0.05) + (re_wt * 0.04) - (si_al * 0.002))

        # Textural properties
        bet_sa = float(np.random.uniform(220.0, 380.0))
        zeo_sa = float(bet_sa * np.random.uniform(0.55, 0.75))
        mat_sa = float(bet_sa - zeo_sa)
        pore_vol = float(np.random.uniform(0.25, 0.45))
        bronsted_acid = float(np.clip(800.0 / (si_al ** 0.5) + (re_wt * 30.0) + np.random.normal(0, 20), 80, 450))
        lewis_acid = float(np.clip(matrix_al * 4.5 + np.random.normal(0, 15), 40, 250))

        # Operating conditions
        temp_c = float(np.random.uniform(500.0, 550.0))
        cat_to_oil = float(np.random.uniform(4.0, 8.5))
        res_time = float(np.random.uniform(1.8, 3.5))
        api_grav = float(np.random.uniform(22.0, 28.0))
        ccr = float(np.random.uniform(0.1, 1.2))

        # Physics-correlated yields
        conversion = float(np.clip(
            52.0 + (cat_to_oil * 2.2) + (temp_c - 520.0) * 0.08 + (bronsted_acid * 0.03) - (si_al * 0.12),
            55.0, 85.0
        ))

        # Gasoline yield correlates with conversion and zeolite type
        gasoline_yield = float(np.clip(conversion * 0.65 - (temp_c - 520.0) * 0.04 + np.random.normal(0, 1.2), 35.0, 55.0))

        # RON increases with ZSM-5, higher temperature, lower UCS (higher dealumination), lower RE
        ron = float(np.clip(
            fw["base_ron"] + (temp_c - 520.0) * 0.035 - (ucs - 24.25) * 12.0 + (si_al * 0.04) + np.random.normal(0, 0.4),
            88.0, 98.5
        ))
        mon = float(ron - np.random.uniform(7.5, 9.5))

        # Coke yield: higher with RE, higher conversion, higher Lewis acidity, higher CCR
        coke_yield = float(np.clip(
            2.2 + (conversion * 0.04) + (re_wt * 0.6) + (lewis_acid * 0.008) + (ccr * 0.8) + (cat_to_oil * 0.15) + np.random.normal(0, 0.3),
            2.5, 9.0
        ))

        # LPG and Light olefins
        lpg_yield = float(np.clip(conversion * 0.22 + (temp_c - 520.0) * 0.03 + np.random.normal(0, 0.8), 12.0, 25.0))
        propylene_yield = float(np.clip(lpg_yield * 0.32 + (1.5 if fw["zeolite"] == "ZSM-5" else 0.0) + np.random.normal(0, 0.3), 3.0, 9.5))
        dry_gas = float(np.clip(conversion - gasoline_yield - lpg_yield - coke_yield, 1.5, 6.0))

        record = {
            "catalyst": {
                "name": f"CAT-{fw['zeolite']}-{i+1:04d}",
                "framework_type": fw["type"],
                "zeolite_type": fw["zeolite"],
                "si_al_ratio": round(si_al, 2),
                "unit_cell_size_angstrom": round(ucs, 3),
                "rare_earth_oxide_wt_pct": round(re_wt, 2),
                "matrix_alumina_wt_pct": round(matrix_al, 2),
                "binder_clay_wt_pct": round(binder_clay, 2),
                "metal_promoters": {"Ni": round(float(np.random.uniform(0.01, 0.15)), 3)}
            },
            "texture": {
                "bet_surface_area_m2_g": round(bet_sa, 1),
                "zeolite_surface_area_m2_g": round(zeo_sa, 1),
                "matrix_surface_area_m2_g": round(mat_sa, 1),
                "total_pore_volume_cm3_g": round(pore_vol, 3),
                "bronsted_acid_sites_umol_g": round(bronsted_acid, 1),
                "lewis_acid_sites_umol_g": round(lewis_acid, 1)
            },
            "operating_conditions": {
                "reaction_temp_celsius": round(temp_c, 1),
                "cat_to_oil_ratio": round(cat_to_oil, 2),
                "residence_time_sec": round(res_time, 2),
                "feedstock_api_gravity": round(api_grav, 2),
                "feedstock_conradson_carbon_wt_pct": round(ccr, 2)
            },
            "performance": {
                "feed_conversion_wt_pct": round(conversion, 2),
                "gasoline_yield_wt_pct": round(gasoline_yield, 2),
                "gasoline_ron": round(ron, 2),
                "gasoline_mon": round(mon, 2),
                "lpg_yield_wt_pct": round(lpg_yield, 2),
                "propylene_yield_wt_pct": round(propylene_yield, 2),
                "coke_yield_wt_pct": round(coke_yield, 2),
                "dry_gas_yield_wt_pct": round(dry_gas, 2)
            }
        }
        records.append(record)

    return records


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "catalysts_seed.json"

    data = generate_benchmark_catalysts(num_samples=300)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data)} realistic catalyst records saved to {out_file}")
