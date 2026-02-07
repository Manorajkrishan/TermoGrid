"""
Carbon Cost & ESG Layer for TermoGrid AI.
carbon_cost = energy_kwh * grid_carbon_intensity * carbon_tax_rate
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CarbonConfig:
    """Carbon/ESG parameters."""
    grid_carbon_intensity_kg_co2_per_kwh: float = 0.35  # UK grid ~0.35, EU avg ~0.4
    carbon_tax_gbp_per_tonne: float = 85.0  # UK 2026 ETS ~£85/tonne
    electricity_price_gbp_per_kwh: float = 0.12


def energy_kwh_from_pue(it_power_mw: float, pue: float, hours: float = 1.0) -> float:
    """Total facility energy (IT + cooling) in kWh."""
    total_mw = it_power_mw * pue
    return total_mw * 1000 * hours


def carbon_cost_gbp(
    energy_kwh: float,
    carbon_intensity: float = 0.35,
    carbon_tax_gbp_per_tonne: float = 85.0,
) -> float:
    """Carbon tax cost in GBP."""
    co2_kg = energy_kwh * carbon_intensity
    co2_tonnes = co2_kg / 1000
    return co2_tonnes * carbon_tax_gbp_per_tonne


def co2_kg(energy_kwh: float, carbon_intensity: float = 0.35) -> float:
    """CO₂ emitted in kg."""
    return energy_kwh * carbon_intensity


def savings_vs_baseline(
    baseline_pue: float,
    optimized_pue: float,
    it_power_mw: float,
    hours: float,
    config: Optional[CarbonConfig] = None,
) -> dict:
    """Compute £ saved and CO₂ avoided vs baseline PUE."""
    config = config or CarbonConfig()
    e_base = energy_kwh_from_pue(it_power_mw, baseline_pue, hours)
    e_opt = energy_kwh_from_pue(it_power_mw, optimized_pue, hours)
    e_saved = e_base - e_opt
    co2_saved_kg = co2_kg(e_saved, config.grid_carbon_intensity_kg_co2_per_kwh)
    carbon_tax_saved = carbon_cost_gbp(
        e_saved, config.grid_carbon_intensity_kg_co2_per_kwh, config.carbon_tax_gbp_per_tonne
    )
    electricity_saved = e_saved * config.electricity_price_gbp_per_kwh
    return {
        "energy_saved_kwh": e_saved,
        "co2_avoided_kg": co2_saved_kg,
        "carbon_tax_saved_gbp": carbon_tax_saved,
        "electricity_saved_gbp": electricity_saved,
        "total_saved_gbp": carbon_tax_saved + electricity_saved,
    }
