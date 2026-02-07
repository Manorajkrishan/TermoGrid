#!/usr/bin/env python3
"""Train PPO RL agent for chilled water setpoint control."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.rl_agent import train_ppo_agent

if __name__ == "__main__":
    models_dir = Path(__file__).resolve().parents[1] / "models"
    print("Training PPO agent (this may take a few minutes)...")
    model = train_ppo_agent(
        total_timesteps=50_000,  # Use 100_000 for better results
        save_path=models_dir,
    )
    print(f"Model saved to {models_dir}")
