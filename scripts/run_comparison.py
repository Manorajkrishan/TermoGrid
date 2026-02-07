#!/usr/bin/env python3
"""Run controller comparison: Static vs Rule-based vs RL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.comparison import run_comparison, compute_savings_and_carbon
from src.simulation.carbon_esg import CarbonConfig

if __name__ == "__main__":
    models_dir = Path(__file__).resolve().parents[1] / "models"
    rl_path = models_dir / "ppo_datacenter_final.zip"
    if not rl_path.exists():
        rl_path = models_dir / "best" / "best_model.zip"

    print("Running controller comparison (5 episodes each)...")
    comp = run_comparison(n_episodes=5, max_steps=200, rl_model_path=rl_path)
    config = CarbonConfig()
    it_mw = 15.0
    hours = 200 * 0.25 / 60
    comp = compute_savings_and_carbon(comp, "Static (10°C)", it_mw, hours, config)

    print("\nResults:")
    for name, m in comp.items():
        print(f"  {name}: PUE={m['avg_pue']:.3f}, violations={m['avg_violations']:.1f}, "
              f"£ saved vs Static=£{m.get('total_saved_gbp', 0):.2f}, CO₂ avoided={m.get('co2_avoided_kg', 0):.0f} kg")
