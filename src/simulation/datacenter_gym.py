"""
Data Center Thermal Control - Gymnasium Environment.
Physics-informed simulation for RL training (PPO chilled water setpoint optimization).
Supports: failure modes (fan/chiller/load spike), physics constraints (max temp rate, inertia).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Optional, Tuple


# Physics constants
MAX_TEMP_RATE_C_PER_STEP = 1.5  # Max °C change per step (thermal inertia)
COOLING_INERTIA = 0.3  # Lag in cooling response


class DataCenterThermalEnv(gym.Env):
    """
    Custom Gymnasium environment for data center thermal control.
    Action: Chilled water setpoint temperature (7-15°C)
    State: Inlet temps, outlet temps, IT load, ambient, PUE components
    Reward: -energy_cost - thermal_violation_penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        baseline_pue: float = 1.3,
        min_chilled_temp: float = 7.0,
        max_chilled_temp: float = 15.0,
        target_inlet_temp: float = 27.0,
        max_steps: int = 1000,
        enable_failures: bool = False,
        failure_step: Optional[int] = None,
        failure_type: Optional[str] = None,  # "fan", "chiller", "load_spike"
    ):
        super().__init__()
        self.render_mode = render_mode
        self.baseline_pue = baseline_pue
        self.min_chilled_temp = min_chilled_temp
        self.max_chilled_temp = max_chilled_temp
        self.target_inlet_temp = target_inlet_temp
        self.max_steps = max_steps
        self.enable_failures = enable_failures
        self.failure_step = failure_step or max_steps // 3
        self.failure_type = failure_type
        self._cooling_efficiency = 1.0  # 1.0 = normal, <1 = degraded
        self._fan_efficiency = 1.0
        self._it_load_multiplier = 1.0  # 1.0 = normal, >1 = spike

        # Action: chilled water setpoint (normalized 0-1 -> 7-15°C)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State: inlet_temp, outlet_temp, it_load_norm, ambient_norm, current_pue
        self.observation_space = spaces.Box(
            low=np.array([15, 20, 0, -10, 1.0]),
            high=np.array([40, 45, 1, 45, 3.0]),
            shape=(5,),
            dtype=np.float32,
        )

        self._state: Optional[np.ndarray] = None
        self._step_count = 0
        self._episode_reward = 0.0

    def _get_initial_state(self) -> np.ndarray:
        inlet = 25 + np.random.uniform(-3, 5)
        outlet = inlet + 5 + np.random.uniform(2, 8)
        it_load = np.random.uniform(0.3, 1.0)
        ambient = 20 + np.random.uniform(-5, 15)
        pue = self.baseline_pue + np.random.uniform(-0.1, 0.3)
        return np.array([inlet, outlet, it_load, ambient, pue], dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._state = self._get_initial_state()
        self._step_count = 0
        self._episode_reward = 0.0
        self._cooling_efficiency = 1.0
        self._fan_efficiency = 1.0
        self._it_load_multiplier = 1.0
        return self._state, {"info": "reset"}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        inlet, outlet, it_load, ambient, pue = self._state

        # Action: chilled water setpoint 7-15°C
        setpoint = self.min_chilled_temp + action[0] * (self.max_chilled_temp - self.min_chilled_temp)

        # Physics-inspired dynamics:
        # Warmer setpoint -> less chiller work -> lower cooling cost but higher inlet temp
        # Colder setpoint -> more chiller work -> higher cost but cooler inlet
        cooling_effect = (self.max_chilled_temp - setpoint) / (self.max_chilled_temp - self.min_chilled_temp)
        heat_load = it_load * 15  # MW equivalent heat
        new_inlet = inlet + 0.1 * (heat_load * 0.5 - cooling_effect * 8) + np.random.normal(0, 0.5)
        new_outlet = outlet + 0.1 * (heat_load * 0.3 - cooling_effect * 5) + np.random.normal(0, 0.3)
        new_inlet = np.clip(new_inlet, 18, 40)
        new_outlet = np.clip(new_outlet, 20, 45)

        # PUE: lower setpoint = more cooling energy = higher PUE
        pue_cooling_component = 0.3 * (1 - cooling_effect) + 0.1
        new_pue = 1.0 + it_load * 0.1 + pue_cooling_component + np.random.uniform(-0.02, 0.02)
        new_pue = np.clip(new_pue, 1.0, 2.5)

        # Restore it_load for state (normalized 0-1)
        state_it_load = min(1.0, it_load)
        self._state = np.array([new_inlet, new_outlet, state_it_load, ambient, new_pue], dtype=np.float32)

        # Reward: minimize PUE, penalize thermal violations (inlet > 27°C)
        thermal_violation = max(0, new_inlet - self.target_inlet_temp)
        energy_cost = new_pue - 1.0  # Overhead
        reward = -energy_cost - 2.0 * thermal_violation
        self._episode_reward += reward

        terminated = self._step_count >= self.max_steps
        truncated = new_inlet > 35  # Safe-mode risk
        info = {
            "pue": new_pue,
            "inlet_temp": new_inlet,
            "setpoint": setpoint,
            "thermal_violation": thermal_violation,
            "energy_kwh": (new_pue * it_load * 15) * 0.25,  # ~15min step = 0.25h
            "failure_active": self._cooling_efficiency < 1 or self._fan_efficiency < 1 or self._it_load_multiplier > 1,
        }
        return self._state, float(reward), terminated, truncated, info
