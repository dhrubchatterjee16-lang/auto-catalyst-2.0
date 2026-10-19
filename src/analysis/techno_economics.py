"""
Module 1: Techno-Economic Analysis (TEA) & Life-Cycle CO2 Carbon Emissions Calculator.
Translates 10-lump product slates and kinetic yields into commercial refinery gross margins,
catalyst unit OPEX, and Scope 1 & 2 carbon footprint per barrel of feed.

Commercial Pricing Benchmarks (2026 Petrochemical Spot Market):
  - VGO Feedstock Cost: ~$72.00 / bbl ($530 / metric ton)
  - Gasoline Cut (RON 92 - 98 tiered pricing): $85.00 - $98.00 / bbl
  - Propylene (Polymer Grade): ~$980 / metric ton (~$0.98 / kg)
  - Butylenes / Alkylate feed: ~$820 / metric ton
  - LCO Diesel: ~$90.00 / bbl
  - Fuel / Dry Gas: $4.50 / MMBtu ($0.22 / kg equivalent)
  - Coke Burn (Credit for steam generation, debit for CO2 ETS tax)

Scope 1 Emissions:
  - Direct CO2 from catalyst coke combustion in FCC regenerator (C + O2 -> CO2)
  - Flue gas carbon tax / ETS credit debit ($65 / ton CO2)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import json


class FCCTechnoEconomicAnalyzer:
    """
    Evaluates unit economics ($/barrel) and Scope 1 & 2 CO2 intensity for FCC catalyst runs.
    """

    # Commercial spot prices ($/kg)
    PRICES_PER_KG = {
        "vgo_feed": 0.53,           # ~$72 / bbl baseline
        "gasoline_base": 0.72,      # Baseline 92 RON ($85 / bbl)
        "gasoline_ron_premium": 0.015, # +$0.015/kg per RON point above 92.0
        "propylene_c3": 0.98,       # High-value polymer grade olefin
        "propane_c3": 0.58,         # Saturated LPG
        "butylenes_c4": 0.82,       # Alkylation / petrochem feed
        "isobutane_c4": 0.65,       # Alkylate feedstock
        "lco_diesel": 0.68,         # Diesel blending pool
        "dry_gas": 0.22,            # Internal fuel gas credit
        "steam_credit_from_coke": 0.035, # High-pressure steam export from CO boiler
    }

    # Carbon tax & emissions constants
    CO2_CARBON_TAX_PER_TON = 65.0   # European ETS / Global benchmark ($/metric ton CO2)
    KG_CO2_PER_KG_COKE = 3.664      # Stoichiometric combustion: 12g C + 32g O2 -> 44g CO2 (44/12 = 3.664)

    # Catalyst manufacturing and replacement OPEX ($/kg catalyst)
    CATALYST_BASE_OPEX_PER_KG = 3.80   # USY / Matrix / Binder cost
    ZSM5_ADDITIVE_OPEX_PER_KG = 7.50   # Specialty MFI crystallization cost
    RARE_EARTH_OPEX_PER_KG = 18.00     # La2O3 / CeO2 raw material cost

    @classmethod
    def evaluate_economics_and_emissions(
        cls,
        product_slate_wt_pct: Dict[str, float],
        gasoline_ron: float,
        catalyst_blend: Dict[str, float],
        feed_throughput_bpd: float = 40000.0,
        fresh_cat_addition_rate_ton_day: float = 4.5
    ) -> Dict[str, Any]:
        """
        Calculates refinery profit margin ($/bbl) and net carbon emissions (kg CO2/bbl).
        """
        # Densities: 1 bbl VGO ~= 142 kg
        KG_PER_BBL = 142.0
        FEED_BBL_PER_DAY = feed_throughput_bpd
        TOTAL_FEED_KG_DAY = FEED_BBL_PER_DAY * KG_PER_BBL

        # 1. Product Revenue Calculations ($ per kg feed)
        gas_ron = max(90.0, float(gasoline_ron))
        gas_price_kg = cls.PRICES_PER_KG["gasoline_base"] + max(0.0, gas_ron - 92.0) * cls.PRICES_PER_KG["gasoline_ron_premium"]

        w = product_slate_wt_pct
        rev_gasoline = (w.get("Gasoline_wt_pct", 40.0) / 100.0) * gas_price_kg
        rev_propylene = (w.get("Propylene_C3=_wt_pct", 6.0) / 100.0) * cls.PRICES_PER_KG["propylene_c3"]
        rev_propane = (w.get("Propane_C3_wt_pct", 2.0) / 100.0) * cls.PRICES_PER_KG["propane_c3"]
        rev_butylenes = (w.get("Butylenes_C4=_wt_pct", 5.0) / 100.0) * cls.PRICES_PER_KG["butylenes_c4"]
        rev_isobutane = (w.get("Isobutane_iC4_wt_pct", 3.0) / 100.0) * cls.PRICES_PER_KG["isobutane_c4"]
        rev_lco = (w.get("LCO_diesel_wt_pct", 8.0) / 100.0) * cls.PRICES_PER_KG["lco_diesel"]
        rev_drygas = (w.get("Dry_Gas_wt_pct", 2.0) / 100.0) * cls.PRICES_PER_KG["dry_gas"]

        # In industrial FCC, unconverted VGO (HCO/slurry) is recycled to extinction or valued as heavy fuel oil (0.48 $/kg)
        unconverted_vgo_wt = w.get("VGO_feed_wt_pct", 0.0) + w.get("Bottoms_HCO_wt_pct", 0.0)
        rev_unconverted = (unconverted_vgo_wt / 100.0) * 0.49

        coke_wt_pct = w.get("Coke_wt_pct", 4.5)
        coke_steam_credit = (coke_wt_pct / 100.0) * cls.PRICES_PER_KG["steam_credit_from_coke"]

        total_revenue_per_kg = (
            rev_gasoline + rev_propylene + rev_propane + rev_butylenes +
            rev_isobutane + rev_lco + rev_drygas + rev_unconverted + coke_steam_credit
        )
        total_revenue_per_bbl = total_revenue_per_kg * KG_PER_BBL

        # 2. Feedstock Cost ($ per bbl)
        feedstock_cost_per_bbl = cls.PRICES_PER_KG["vgo_feed"] * KG_PER_BBL

        # 3. Catalyst Replacement OPEX ($ per bbl)
        usy_frac = catalyst_blend.get("usy_fraction", 0.75)
        zsm5_frac = catalyst_blend.get("zsm5_fraction", 0.25)
        re_wt = catalyst_blend.get("rare_earth_wt_pct", 0.15)

        blended_cat_cost_per_kg = (
            usy_frac * cls.CATALYST_BASE_OPEX_PER_KG +
            zsm5_frac * cls.ZSM5_ADDITIVE_OPEX_PER_KG +
            (re_wt / 100.0) * cls.RARE_EARTH_OPEX_PER_KG
        )
        daily_catalyst_spend = (fresh_cat_addition_rate_ton_day * 1000.0) * blended_cat_cost_per_kg
        catalyst_cost_per_bbl = daily_catalyst_spend / FEED_BBL_PER_DAY

        # 4. Scope 1 CO2 Emissions & Carbon Tax Debit
        # Direct combustion of coke deposit in FCC regenerator
        kg_coke_per_bbl = (coke_wt_pct / 100.0) * KG_PER_BBL
        kg_co2_per_bbl = kg_coke_per_bbl * cls.KG_CO2_PER_KG_COKE
        carbon_tax_cost_per_bbl = (kg_co2_per_bbl / 1000.0) * cls.CO2_CARBON_TAX_PER_TON

        # 5. Net FCC Refining Margin ($ / bbl)
        gross_margin_per_bbl = total_revenue_per_bbl - feedstock_cost_per_bbl
        net_margin_per_bbl = gross_margin_per_bbl - catalyst_cost_per_bbl - carbon_tax_cost_per_bbl
        annual_ebitda_millions = (net_margin_per_bbl * FEED_BBL_PER_DAY * 350.0) / 1e6

        return {
            "financial_metrics": {
                "gross_product_revenue_per_bbl": round(total_revenue_per_bbl, 2),
                "feedstock_cost_per_bbl": round(feedstock_cost_per_bbl, 2),
                "catalyst_opex_per_bbl": round(catalyst_cost_per_bbl, 3),
                "carbon_tax_debit_per_bbl": round(carbon_tax_cost_per_bbl, 2),
                "net_margin_per_bbl": round(net_margin_per_bbl, 2),
                "annual_unit_ebitda_usd_millions": round(annual_ebitda_millions, 2)
            },
            "emissions_metrics": {
                "scope_1_co2_intensity_kg_per_bbl": round(kg_co2_per_bbl, 2),
                "daily_flue_gas_co2_metric_tons": round((kg_co2_per_bbl * FEED_BBL_PER_DAY) / 1000.0, 1),
                "effective_carbon_tax_usd_ton": cls.CO2_CARBON_TAX_PER_TON
            },
            "revenue_breakdown_usd_per_bbl": {
                "Gasoline_pool": round(rev_gasoline * KG_PER_BBL, 2),
                "Propylene_petrochem": round(rev_propylene * KG_PER_BBL, 2),
                "Butylenes_alkylation": round(rev_butylenes * KG_PER_BBL, 2),
                "LCO_diesel": round(rev_lco * KG_PER_BBL, 2),
                "LPG_fuel_gas": round((rev_propane + rev_isobutane + rev_drygas) * KG_PER_BBL, 2)
            }
        }


if __name__ == "__main__":
    print("=" * 80)
    print("  FCC TECHNO-ECONOMIC ANALYSIS (TEA) & SCOPE 1 CO2 CALCULATOR")
    print("=" * 80)

    # Test with typical 10-lump results
    sample_yields = {
        "Gasoline_wt_pct": 42.5,
        "Propylene_C3=_wt_pct": 8.2,
        "Propane_C3_wt_pct": 1.8,
        "Butylenes_C4=_wt_pct": 6.5,
        "Isobutane_iC4_wt_pct": 3.4,
        "LCO_diesel_wt_pct": 8.0,
        "Dry_Gas_wt_pct": 1.8,
        "Coke_wt_pct": 4.6
    }

    cat_blend = {
        "usy_fraction": 0.764,
        "zsm5_fraction": 0.236,
        "rare_earth_wt_pct": 0.15
    }

    tea = FCCTechnoEconomicAnalyzer.evaluate_economics_and_emissions(
        product_slate_wt_pct=sample_yields,
        gasoline_ron=96.8,
        catalyst_blend=cat_blend,
        feed_throughput_bpd=40000.0
    )

    fin = tea["financial_metrics"]
    em = tea["emissions_metrics"]
    rev = tea["revenue_breakdown_usd_per_bbl"]

    print(f"Refinery Unit Economics (40,000 bpd FCC Unit):")
    print(f"  • Gross Product Revenue       : ${fin['gross_product_revenue_per_bbl']:.2f} / bbl")
    print(f"  • Feedstock Cost              : ${fin['feedstock_cost_per_bbl']:.2f} / bbl")
    print(f"  • Catalyst Replacement OPEX   : ${fin['catalyst_opex_per_bbl']:.3f} / bbl")
    print(f"  • Carbon Tax Debit ($65/t CO2): ${fin['carbon_tax_debit_per_bbl']:.2f} / bbl")
    print(f"  -------------------------------------------------------------")
    print(f"  • NET REFINING MARGIN         : ${fin['net_margin_per_bbl']:.2f} / bbl")
    print(f"  • ANNUAL ESTIMATED EBITDA     : ${fin['annual_unit_ebitda_usd_millions']:.2f} Million USD / year")

    print(f"\nCarbon Emissions & Environmental Impact:")
    print(f"  • Scope 1 CO2 Intensity       : {em['scope_1_co2_intensity_kg_per_bbl']:.2f} kg CO2 / bbl feed")
    print(f"  • Daily Flue Gas CO2 Volume   : {em['daily_flue_gas_co2_metric_tons']:.1f} metric tons CO2 / day")

    print(f"\nKey Revenue Value Drivers:")
    print(f"  - Gasoline Pool (96.8 RON)    : ${rev['Gasoline_pool']:.2f} / bbl")
    print(f"  - Propylene (Polymer Grade)   : ${rev['Propylene_petrochem']:.2f} / bbl (High-value petrochemical co-product)")
    print(f"  - Butylenes Alkylation Feed   : ${rev['Butylenes_alkylation']:.2f} / bbl")
    print("=" * 80)
