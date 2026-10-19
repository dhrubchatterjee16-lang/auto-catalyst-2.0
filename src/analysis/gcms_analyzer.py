"""
Option 3: Real-Time GC-MS Spectral Analyzer and Quantitative Yield Resolver.
Processes raw gas chromatography (FID / MS) retention-time intensity streams
and translates spectral peak distributions into ASTM D2887 simulated distillation fractions,
hydrocarbon PIONA cuts (Paraffins, Isoparaffins, Olefins, Naphthenes, Aromatics),
and calculated Research Octane Number (RON).

Provides the closed-loop feedback mechanism:
  Raw GC Detector Stream -> Peak Deconvolution -> Component Quantitation -> Performance Feedback to Optimizer
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, Any, List
from pathlib import Path
import json


class GCMSProductAnalyzer:
    """
    Automated spectral deconvolution and hydrocarbon cut quantitation engine.
    """

    # Reference retention times (minutes) for simulated gasoline boiling curve
    HYDROCARBON_WINDOWS = {
        "Dry_Gas_C1_C2": (0.0, 1.2),
        "Propylene_C3=": (1.2, 1.8),
        "Propane_C3": (1.8, 2.3),
        "Isobutane_iC4": (2.3, 2.9),
        "Butenes_C4=": (2.9, 3.6),
        "n_Butane_nC4": (3.6, 4.2),
        "Isopentane_iC5": (4.2, 5.2),
        "n_Pentane_nC5": (5.2, 6.0),
        "Isooctane_Isoparaffins": (6.0, 8.5),
        "Benzene_Toluene_Aromatics": (8.5, 12.0),
        "Heavy_Aromatics_C9_C10": (12.0, 16.5),
        "Diesel_LCO_Cut": (16.5, 22.0)
    }

    # Pure component Research Octane Numbers (RON)
    PURE_RON_LOOKUP = {
        "Dry_Gas_C1_C2": 0.0,
        "Propylene_C3=": 102.0,
        "Propane_C3": 112.0,
        "Isobutane_iC4": 102.0,
        "Butenes_C4=": 100.0,
        "n_Butane_nC4": 94.0,
        "Isopentane_iC5": 92.3,
        "n_Pentane_nC5": 61.7,
        "Isooctane_Isoparaffins": 100.0,
        "Benzene_Toluene_Aromatics": 118.0,
        "Heavy_Aromatics_C9_C10": 108.0,
        "Diesel_LCO_Cut": 30.0
    }

    @classmethod
    def simulate_chromatogram(
        cls,
        total_minutes: float = 24.0,
        sampling_rate_hz: float = 10.0,
        zsm5_booster: bool = True,
        noise_level: float = 0.015,
        seed: int = 42,
        product_yields: Dict[str, float] = None,
        catalyst_properties: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Simulates raw high-resolution FID chromatogram detector signal (pA vs Retention Time).
        Continuously scales peak distributions from actual 10-lump ODE product yields
        and crystallographic catalyst properties (eliminating unphysical step functions).
        """
        np.random.seed(seed)
        t = np.linspace(0.0, total_minutes, int(total_minutes * 60 * sampling_rate_hz))
        signal = np.zeros_like(t)

        # Baseline drift
        baseline = 0.05 + 0.02 * (t / total_minutes) ** 1.5

        if product_yields is not None:
            # CONTINUOUS MULTI-SCALE COUPLING:
            # Map ODE lump mass fractions (wt%) directly to FID peak areas
            w_gas = max(10.0, float(product_yields.get("Gasoline_wt_pct", 45.0)))
            w_prop = max(0.5, float(product_yields.get("Propylene_C3=_wt_pct", 5.0)))
            w_propane = max(0.2, float(product_yields.get("Propane_C3_wt_pct", 1.8)))
            w_butenes = max(0.5, float(product_yields.get("Butylenes_C4=_wt_pct", 5.5)))
            w_isobutane = max(0.5, float(product_yields.get("Isobutane_iC4_wt_pct", 3.2)))
            w_drygas = max(0.2, float(product_yields.get("Dry_Gas_wt_pct", 1.8)))
            w_lco = max(1.0, float(product_yields.get("LCO_diesel_wt_pct", 8.5)))

            # Continuous catalyst framework modulation
            cat = catalyst_properties or {}
            zsm5_frac = float(cat.get("zsm5_fraction", 0.20 if zsm5_booster else 0.05))
            ucs = float(cat.get("usy_ucs_angstrom", 24.265))
            re_wt = float(cat.get("rare_earth_wt_pct", 0.15))

            # Gas baseline scaling factor (relative to 45 wt% nominal gasoline)
            gas_scale = w_gas / 45.0

            # Dealumination & ZSM-5 booster impact on gasoline PIONA cuts:
            # 1. ZSM-5 cracks linear n-paraffins (nC5) to light olefins
            h_nc5 = max(0.2, 2.4 * gas_scale * (1.0 - 1.2 * min(0.35, zsm5_frac)))
            # 2. Lower UCS and higher ZSM-5 boost high-octane aromatics (BTX)
            # Low UCS (lower HT index) prevents over-saturation of naphthenes into low-octane paraffins
            ucs_octane_factor = max(0.6, 1.0 + (24.40 - ucs) * 3.5 - (re_wt * 0.15))
            h_btx = (4.8 + 9.5 * zsm5_frac) * gas_scale * ucs_octane_factor
            h_heavy_aro = (3.8 + 3.0 * zsm5_frac) * gas_scale * ucs_octane_factor
            h_isoparaffins = 6.5 * gas_scale
            h_ic5 = 4.0 * gas_scale

            peaks = [
                (0.8, w_drygas * 0.65, 0.06),              # C1-C2 Dry gas
                (1.5, w_prop * 0.75, 0.07),                # Propylene C3=
                (2.0, w_propane * 1.05, 0.08),             # Propane C3
                (2.6, w_isobutane * 0.90, 0.09),           # Isobutane iC4
                (3.2, w_butenes * 0.45, 0.09),             # Butenes C4=
                (3.9, 1.4 * gas_scale, 0.10),              # nC4
                (4.7, h_ic5, 0.12),                        # Isopentane iC5
                (5.6, h_nc5, 0.12),                        # n-Pentane nC5
                (7.2, h_isoparaffins, 0.35),               # Isooctane / Isoparaffins
                (10.2, h_btx, 0.40),                       # BTX Aromatics
                (14.0, h_heavy_aro, 0.50),                 # Heavy Aromatics C9-C10
                (18.5, w_lco * 0.26, 0.70),                # LCO Diesel cut
            ]
        else:
            # Fallback legacy baseline definition
            peaks = [
                (0.8, 1.2, 0.06),   # C1-C2
                (1.5, 3.8 if zsm5_booster else 1.8, 0.07),  # Propylene (boosted by ZSM-5)
                (2.0, 2.1, 0.08),   # Propane
                (2.6, 2.9, 0.09),   # iC4
                (3.2, 2.4 if zsm5_booster else 1.2, 0.09),  # C4=
                (3.9, 1.4, 0.10),   # nC4
                (4.7, 4.2, 0.12),   # iC5
                (5.6, 1.1 if zsm5_booster else 2.5, 0.12),  # nC5 (cracked down by ZSM-5)
                (7.2, 6.8, 0.35),   # Isoparaffins / Alkylates
                (10.2, 7.5 if zsm5_booster else 5.2, 0.40), # Aromatics (BTX)
                (14.0, 4.5, 0.50),  # Heavy Aromatics
                (18.5, 2.2, 0.70),  # LCO
            ]

        for center, height, width in peaks:
            # Gaussian chromatographic peak shape
            signal += height * np.exp(-0.5 * ((t - center) / width) ** 2)

        # Add detector Gaussian white noise
        noise = np.random.normal(0, noise_level, size=len(t))
        total_signal = signal + baseline + noise

        return pd.DataFrame({"retention_time_min": t, "detector_current_pa": total_signal})

    @classmethod
    def analyze_spectrum(cls, chromatogram_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs baseline subtraction, peak integration, and PIONA octane blending.
        """
        t = chromatogram_df["retention_time_min"].values
        sig = chromatogram_df["detector_current_pa"].values

        # Baseline estimation (rolling minimum filter)
        window = int(len(t) * 0.03)
        baseline = pd.Series(sig).rolling(window, min_periods=1, center=True).quantile(0.05).values
        net_signal = np.maximum(0, sig - baseline)

        # Integrate peak areas per hydrocarbon window
        area_per_cut = {}
        dt = t[1] - t[0]

        for cut_name, (t_start, t_end) in cls.HYDROCARBON_WINDOWS.items():
            mask = (t >= t_start) & (t < t_end)
            area = float(np.sum(net_signal[mask]) * dt)
            area_per_cut[cut_name] = max(0.0, area)

        total_area = sum(area_per_cut.values())
        mass_fractions = {k: (v / total_area) * 100.0 for k, v in area_per_cut.items()}

        # Group into industrial product slates
        dry_gas = mass_fractions["Dry_Gas_C1_C2"]
        lpg = (
            mass_fractions["Propylene_C3="]
            + mass_fractions["Propane_C3"]
            + mass_fractions["Isobutane_iC4"]
            + mass_fractions["Butenes_C4="]
            + mass_fractions["n_Butane_nC4"]
        )
        propylene = mass_fractions["Propylene_C3="]

        # Gasoline cut: C5 to C10 aromatics
        gasoline_cuts = [
            "Isopentane_iC5",
            "n_Pentane_nC5",
            "Isooctane_Isoparaffins",
            "Benzene_Toluene_Aromatics",
            "Heavy_Aromatics_C9_C10",
        ]
        gasoline_yield = sum(mass_fractions[c] for c in gasoline_cuts)
        diesel_lco = mass_fractions["Diesel_LCO_Cut"]

        # Calculate Octane (RON) using linear-by-volume blending on gasoline lump
        total_gasoline_area = sum(area_per_cut[c] for c in gasoline_cuts)
        calculated_ron = sum(
            (area_per_cut[c] / total_gasoline_area) * cls.PURE_RON_LOOKUP[c]
            for c in gasoline_cuts
        )

        return {
            "resolved_yields_wt_pct": {
                "dry_gas": round(dry_gas, 2),
                "lpg_total": round(lpg, 2),
                "propylene_olefin": round(propylene, 2),
                "gasoline_cut": round(gasoline_yield, 2),
                "lco_diesel_cut": round(diesel_lco, 2),
            },
            "gasoline_octane_ron": round(calculated_ron, 2),
            "piona_component_distribution": {k: round(v, 2) for k, v in mass_fractions.items()}
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  GC-MS SPECTRAL ANALYZER & YIELD RESOLUTION ENGINE")
    print("=" * 70)

    # Simulate raw detector stream from our ZSM-5 boosted composite catalyst
    print("Simulating raw 24-minute high-resolution FID chromatogram...")
    raw_gc = GCMSProductAnalyzer.simulate_chromatogram(zsm5_booster=True)

    print("Running automated peak deconvolution and PIONA integration...")
    results = GCMSProductAnalyzer.analyze_spectrum(raw_gc)

    yields = results["resolved_yields_wt_pct"]
    print("\n[RESOLVED PRODUCT SLATE METRICS]")
    print(f"  • Gasoline Liquid Yield   : {yields['gasoline_cut']}%")
    print(f"  • Gasoline Octane (RON)   : {results['gasoline_octane_ron']} RON")
    print(f"  • Propylene Petrochemical : {yields['propylene_olefin']}%")
    print(f"  • Total LPG (C3-C4)       : {yields['lpg_total']}%")
    print(f"  • Dry Gas (C1-C2)         : {yields['dry_gas']}%")
    print(f"  • LCO Diesel Cut          : {yields['lco_diesel_cut']}%")

    print("\n[KEY HYDROCARBON FRACTIONS]")
    for comp, pct in list(results["piona_component_distribution"].items())[1:8]:
        print(f"  - {comp:<26}: {pct:5.2f}%")

    print("=" * 70)

    # Save spectral output data
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    raw_gc.to_csv(out_dir / "simulated_chromatogram_stream.csv", index=False)
    with open(out_dir / "gcms_resolved_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Raw chromatogram stream & resolved report exported to {out_dir}")
