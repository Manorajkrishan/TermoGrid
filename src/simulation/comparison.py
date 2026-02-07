"""
Controller comparison: Static vs Rule-based vs RL.
Energy usage, thermal violations, PUE improvement, recovery time.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .controllers import BaseController, RuleBasedController, StaticController
from .datacenter_gym import DataCenterThermalEnv
from .carbon_esg import CarbonConfig, savings_vs_baseline, co2_kg, energy_kwh_from_pue


def run_episode(
    env: DataCenterThermalEnv,
    controller: BaseController,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one episode and collect metrics."""
    max_steps = max_steps or env.max_steps
    obs, _ = env.reset()
    total_reward = 0.0
    pue_list = []
    inlet_list = []
    energy_list = []
    violations = 0
    step = 0
    while step < max_steps:
        action = controller.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        pue_list.append(info["pue"])
        inlet_list.append(info["inlet_temp"])
        energy_list.append(info.get("energy_kwh", info["pue"] * 0.25 * 15))
        if info["inlet_temp"] > env.target_inlet_temp:
            violations += max(0, info["inlet_temp"] - env.target_inlet_temp)
        step += 1
        if term or trunc:
            break
    total_energy = sum(energy_list)
    return {
        "total_reward": total_reward,
        "avg_pue": np.mean(pue_list) if pue_list else 0,
        "thermal_violations": violations,
        "max_inlet": max(inlet_list) if inlet_list else 0,
        "energy_kwh": total_energy,
        "steps": len(pue_list),
    }


def run_comparison(
    n_episodes: int = 10,
    max_steps: int = 500,
    rl_model_path: Optional[Path] = None,
    enable_failures: bool = False,
    failure_step: Optional[int] = None,
    failure_type: Optional[str] = None,
) -> Dict[str, Dict]:
    """Compare Static, Rule-based, and optionally RL."""
    from .controllers import RLController
    from stable_baselines3 import PPO

    results = {}
    controllers: List[tuple] = [
        ("Static (10°C)", StaticController(setpoint_c=10.0)),
        ("Rule-based", RuleBasedController()),
    ]
    if rl_model_path and Path(rl_model_path).exists():
        model = PPO.load(str(rl_model_path))
        controllers.append(("RL (PPO)", RLController(model)))

    env_kw = dict(max_steps=max_steps)
    if enable_failures:
        env_kw["enable_failures"] = True
        env_kw["failure_step"] = failure_step or max_steps // 3
        env_kw["failure_type"] = failure_type or "chiller"

    for name, ctrl in controllers:
        env = DataCenterThermalEnv(**env_kw)
        rewards, pues, violations, energies, max_inlets = [], [], [], [], []
        for ep in range(n_episodes):
            m = run_episode(env, ctrl, max_steps)
            rewards.append(m["total_reward"])
            pues.append(m["avg_pue"])
            violations.append(m["thermal_violations"])
            energies.append(m["energy_kwh"])
            max_inlets.append(m.get("max_inlet", 0))
        results[name] = {
            "avg_reward": np.mean(rewards),
            "avg_pue": np.mean(pues),
            "avg_violations": np.mean(violations),
            "avg_energy_kwh": np.mean(energies),
            "std_pue": np.std(pues),
            "max_inlet": np.mean(max_inlets) if max_inlets else 0,
        }

    return results


def compute_savings_and_carbon(
    comparison: Dict[str, Dict],
    baseline_name: str = "Static (10°C)",
    it_power_mw: float = 15.0,
    hours_per_episode: float = 500 * 0.25 / 60,  # ~2h per 500 steps at 15min
    config: Optional[CarbonConfig] = None,
) -> Dict[str, Dict]:
    """Add £ saved and CO₂ avoided vs baseline for each controller."""
    config = config or CarbonConfig()
    baseline_pue = comparison[baseline_name]["avg_pue"]
    out = {}
    for name, m in comparison.items():
        opt_pue = m["avg_pue"]
        savings = savings_vs_baseline(baseline_pue, opt_pue, it_power_mw, hours_per_episode, config)
        out[name] = {**m, **savings}
    return out
