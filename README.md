# TermoGrid AI

**Physics-Informed Reinforcement Learning for Dynamic Data Center Thermal Control**

Modern GPU clusters generate heat in bursts. Traditional cooling is *reactive*—it waits for the room to get hot before turning up fans, leading to **thermal oscillations** and wasted energy. TermoGrid AI turns cooling into a **predictive** system.

## Features

1. **PUE Forecaster** – LSTM/XGBoost model predicting Power Usage Effectiveness 15 minutes ahead
2. **Thermal Digital Twin** – 3D visualization of "Hot Alleys" in the data center floor
3. **RL Control Agent** – PPO algorithm optimizing Chilled Water Setpoint (7–15°C)
4. **Baseline Controller Comparison** – Static (10°C) vs Rule-based vs RL; energy, violations, PUE
5. **Carbon Cost & ESG** – CO₂ avoided (kg/day), £ carbon tax avoided, Net-Zero 2026 positioning
6. **Failure Mode Simulation** – Fan failure, chiller failure, IT load spike; RL vs rule-based recovery
7. **Physics-Informed Constraints** – Max temp rate, cooling inertia, thermal mass
8. **Executive Dashboard** – £ saved today/month, CO₂ avoided, controller toggle, failure simulation

## Quick Start

```bash
# 1. Create environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (generates synthetic data if external sources fail)
python scripts/download_data.py

# 4. Launch dashboard
python scripts/run_dashboard.py

# 5. (Optional) Train RL agent
python scripts/train_rl_agent.py
```

## Dataset Sources

| Data Type | Source | Fallback |
|-----------|--------|----------|
| HPC Waste Heat | [ORNL Frontier Figshare](https://figshare.com/articles/dataset/Dataset_of_Frontier_supercomputer/24391240) | Synthetic (8–30 MW, 30–38°C) |
| Cooling Control | [Kaggle DC Cold Source](https://www.kaggle.com/datasets/programmer3/data-center-cold-source-control-dataset) | Synthetic inlet/outlet, chiller/AHU power |
| Weather | Open-Meteo API | Synthetic humidity, dew point |

**Kaggle:** Place `kaggle.json` in `~/.kaggle/` for Kaggle dataset downloads.

## Controller Comparison

```bash
python scripts/run_comparison.py
```

## Project Structure

```
Chip/
├── config.yaml           # Configuration
├── requirements.txt
├── scripts/
│   ├── download_data.py  # Download/generate datasets
│   ├── train_rl_agent.py # Train PPO agent
│   └── run_dashboard.py  # Launch Streamlit
├── src/
│   ├── data/             # Download, loaders
│   ├── models/           # PUE forecaster, RL agent
│   ├── simulation/       # Gymnasium env
│   ├── visualization/    # Thermal twin
│   └── dashboard/        # Streamlit app
└── data/                 # Downloaded/generated data
```

## The 2026 Pitch

- **Buyer:** Data Center Operations (Equinix, Digital Realty, Google)
- **ROI:** Reduce cooling-related electricity costs by eliminating over-cooling buffers
- **Net-Zero:** Frame as AI for carbon tax avoidance—governments are taxing data center emissions

## License

MIT
