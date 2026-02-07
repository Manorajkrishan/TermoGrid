"""
Baseline controllers for data center thermal control.
Static, Rule-based, and RL wrapper for comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseController(ABC):
    """Base class for thermal controllers."""

    name: str = "Base"

    @abstractmethod
    def act(self, state: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        """Return action (normalized 0-1 -> chilled water setpoint 7-15°C)."""
        pass


class StaticController(BaseController):
    """Always set chilled water = 10°C (mid-range)."""

    name = "Static (10°C)"

    def __init__(self, setpoint_c: float = 10.0, min_temp: float = 7.0, max_temp: float = 15.0):
        self.setpoint_c = setpoint_c
        self.min_temp = min_temp
        self.max_temp = max_temp

    def act(self, state: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        # Normalize setpoint to 0-1
        action = np.clip(
            (self.setpoint_c - self.min_temp) / (self.max_temp - self.min_temp), 0.0, 1.0
        )
        return np.array([action], dtype=np.float32)


class RuleBasedController(BaseController):
    """
    Rule-based: increase cooling when inlet > threshold, decrease when below.
    Inlet > 27°C -> colder setpoint (more cooling)
    Inlet < 24°C -> warmer setpoint (less cooling)
    """

    name = "Rule-based"

    def __init__(
        self,
        target_inlet: float = 27.0,
        deadband_low: float = 24.0,
        deadband_high: float = 27.0,
        step_size: float = 0.15,
        min_temp: float = 7.0,
        max_temp: float = 15.0,
    ):
        self.target_inlet = target_inlet
        self.deadband_low = deadband_low
        self.deadband_high = deadband_high
        self.step_size = step_size
        self.min_temp = min_temp
        self.max_temp = max_temp
        self._last_action = 0.5  # Start at 11°C

    def act(self, state: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        inlet_temp = state[0]
        action = self._last_action
        if inlet_temp > self.deadband_high:
            action = min(1.0, action + self.step_size)  # More cooling
        elif inlet_temp < self.deadband_low:
            action = max(0.0, action - self.step_size)  # Less cooling
        self._last_action = action
        return np.array([action], dtype=np.float32)


class RLController(BaseController):
    """Wrapper for RL agent (PPO)."""

    name = "RL (PPO)"

    def __init__(self, model):
        self.model = model

    def act(self, state: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        action, _ = self.model.predict(state, deterministic=True)
        return np.atleast_1d(np.array(action, dtype=np.float32))
