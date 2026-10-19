"""
Data schema for Fluid Catalytic Cracking (FCC) and Hydroprocessing catalysts.
Defines chemical formulations, textural properties, operating conditions, and performance metrics.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CatalystComposition(BaseModel):
    name: str = Field(..., description="Unique identifier or code for the catalyst")
    framework_type: str = Field(..., description="Zeolite framework type (e.g., FAU, MFI, BEA)")
    zeolite_type: str = Field(..., description="Specific zeolite form (e.g., USY, ZSM-5, Beta)")
    si_al_ratio: float = Field(..., gt=0, description="Silicon to Aluminum molar ratio (SiO2/Al2O3)")
    unit_cell_size_angstrom: float = Field(..., description="Unit cell size in Angstroms (e.g., 24.20 - 24.60)")
    rare_earth_oxide_wt_pct: float = Field(0.0, ge=0.0, description="Rare-earth oxide promoter wt% (e.g., La2O3, CeO2)")
    matrix_alumina_wt_pct: float = Field(..., ge=0.0, description="Active matrix alumina content wt%")
    binder_clay_wt_pct: float = Field(..., ge=0.0, description="Inert/binder kaolin clay wt%")
    metal_promoters: Dict[str, float] = Field(
        default_factory=dict,
        description="Dopant/promoter metal wt% (e.g., {'Ni': 0.1, 'V': 0.2, 'Pt': 0.05})"
    )


class TexturalProperties(BaseModel):
    bet_surface_area_m2_g: float = Field(..., gt=0, description="BET specific surface area in m^2/g")
    zeolite_surface_area_m2_g: float = Field(..., gt=0, description="Micropore/zeolite surface area in m^2/g")
    matrix_surface_area_m2_g: float = Field(..., ge=0, description="Mesopore/matrix surface area in m^2/g")
    total_pore_volume_cm3_g: float = Field(..., gt=0, description="Total pore volume in cm^3/g")
    bronsted_acid_sites_umol_g: float = Field(..., ge=0, description="Brønsted acid site density in umol/g")
    lewis_acid_sites_umol_g: float = Field(..., ge=0, description="Lewis acid site density in umol/g")


class OperatingConditions(BaseModel):
    reaction_temp_celsius: float = Field(..., description="Riser cracking temperature in °C (e.g., 500 - 560)")
    cat_to_oil_ratio: float = Field(..., gt=0, description="Catalyst to oil mass ratio (C/O)")
    residence_time_sec: float = Field(..., gt=0, description="Contact time in seconds (e.g., 1.5 - 4.0)")
    feedstock_api_gravity: float = Field(..., description="Feedstock API gravity (e.g., 20 - 28 for VGO)")
    feedstock_conradson_carbon_wt_pct: float = Field(0.0, ge=0.0, description="Feedstock Conradson Carbon Residue (CCR)")


class PerformanceYields(BaseModel):
    feed_conversion_wt_pct: float = Field(..., ge=0, le=100, description="Total feed conversion wt%")
    gasoline_yield_wt_pct: float = Field(..., ge=0, le=100, description="Gasoline cut (C5 - 216°C) yield wt%")
    gasoline_ron: float = Field(..., ge=80, le=110, description="Research Octane Number of gasoline")
    gasoline_mon: float = Field(..., ge=70, le=100, description="Motor Octane Number of gasoline")
    lpg_yield_wt_pct: float = Field(..., ge=0, le=100, description="Liquefied petroleum gas (C3-C4) yield wt%")
    propylene_yield_wt_pct: float = Field(..., ge=0, le=100, description="Light olefin propylene yield wt%")
    coke_yield_wt_pct: float = Field(..., ge=0, le=100, description="Coke deposit yield wt%")
    dry_gas_yield_wt_pct: float = Field(..., ge=0, le=100, description="Dry gas (H2, C1, C2) yield wt%")


class CatalystRecord(BaseModel):
    catalyst: CatalystComposition
    texture: TexturalProperties
    operating_conditions: OperatingConditions
    performance: PerformanceYields
