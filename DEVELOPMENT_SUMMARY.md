# TermoGrid AI – Development Summary

## What We Have Developed So Far

### Production-Ready Additions (v0.2)

| Feature | Description |
|--------|-------------|
| **Baseline Comparison** | Static (10°C), Rule-based, RL controllers; PUE, violations, energy metrics |
| **Carbon & ESG** | `carbon_cost = energy_kwh * grid_intensity * carbon_tax`; £ saved, CO₂ avoided |
| **Failure Simulation** | Fan failure, chiller failure, IT load spike; recovery comparison |
| **Physics Constraints** | Max temp rate (1.5°C/step), cooling inertia, thermal mass |
| **Executive Dashboard** | £ saved today/month, CO₂ avoided, controller toggle, failure mode |

---

### 1. **Data Pipeline** ✅

| Dataset | Source | Status |
|--------|--------|--------|
| **Frontier (HPC Waste Heat)** | Figshare API → xlsx | ✅ Downloaded & parsed into `frontier_waste_heat.csv` (49,747 rows of real data) |
| **DC Cold Source** | Kaggle API | ⚠️ Synthetic fallback (requires `kaggle.json` in `~/.kaggle/`) |
| **Weather** | Open-Meteo API | ✅ Real data (humidity, temperature, dew point) |

- **Xlsx Parser**: Converts ORNL Frontier xlsx into our schema (power_mw, waste_heat_temp_c, pump_speed, water temps)
- **Robust Loaders**: Handle mixed datetime formats, resample to 15min, merge into a PUE dataset
- **Unified PUE Dataset**: Combines Frontier + Cold Source + Weather for model training

---

### 2. **PUE Forecaster** ✅

- **XGBoost model**: Predicts PUE 15 minutes ahead using lookback window (e.g. 48–96 steps)
- **LSTM model** (optional): Deep learning alternative
- **Features**: power_mw, waste_heat_temp_c, pump_speed, inlet/outlet temps, chiller/AHU power, ambient/humidity
- **Output**: PUE prediction with MAE metric

---

### 3. **Thermal Digital Twin** ✅

- **3D visualization**: Plotly-based hot-alleys view
- **Floor plan heatmap**: 2D thermal map at rack height
- **3D surface**: Temperature distribution
- **Physics-inspired model**: Simplified diffusion from power (heat) and cooling distribution

---

### 4. **RL Control Agent** ✅

- **Environment**: Custom Gymnasium `DataCenterThermalEnv`
- **Action**: Chilled water setpoint (7–15°C)
- **State**: Inlet/outlet temps, IT load, ambient, PUE
- **Reward**: Minimize energy cost + thermal violation penalties
- **Algorithm**: PPO via Stable-Baselines3
- **Training**: `python scripts/train_rl_agent.py`

---

### 5. **Streamlit Dashboard** ✅

- **Overview**: PUE metrics, IT load, time series
- **Thermal Digital Twin**: 3D hot alleys visualization
- **PUE Forecaster**: Actual vs predicted, MAE
- **RL Agent**: Status and training instructions

---

## Project Structure

```
Chip/
├── config.yaml           # PUE, RL, thermal parameters
├── requirements.txt      # Dependencies (incl. openpyxl for xlsx)
├── README.md
├── DEVELOPMENT_SUMMARY.md
├── scripts/
│   ├── download_data.py   # Download + parse all datasets
│   ├── train_rl_agent.py  # Train PPO agent
│   └── run_dashboard.py   # Launch Streamlit
├── src/
│   ├── data/
│   │   ├── download_datasets.py  # Frontier/Kaggle/Weather + xlsx parser
│   │   └── loaders.py            # PUE dataset builder
│   ├── models/
│   │   ├── pue_forecaster.py     # XGBoost & LSTM
│   │   └── rl_agent.py           # PPO agent
│   ├── simulation/
│   │   └── datacenter_gym.py     # Gymnasium env
│   ├── visualization/
│   │   └── thermal_twin.py       # 3D thermal viz
│   └── dashboard/
│       └── app.py                # Streamlit app
└── data/
    ├── frontier/          # frontier_waste_heat.csv (from xlsx)
    ├── kaggle_cold_source/# cold_source_control.csv (synthetic if no Kaggle)
    ├── weather/           # weather.csv (Open-Meteo)
    └── synthetic/         # Extra RL training data
```

---

## How to Run

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Download data (parses Frontier xlsx, fetches weather, generates cold source if needed)
python scripts/download_data.py

# 3. Dashboard
python scripts/run_dashboard.py

# 4. Train RL agent (optional)
python scripts/train_rl_agent.py
```

---

## Kaggle: Getting Real Cold Source Data

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
2. Save `kaggle.json` to `~/.kaggle/` (or `C:\Users\<you>\.kaggle\` on Windows)
3. Re-run `python scripts/download_data.py`

---

## Tech Stack

- **Python 3.11+**
- **ML**: PyTorch, XGBoost, scikit-learn
- **RL**: Gymnasium, Stable-Baselines3 (PPO)
- **Data**: Pandas, Dask, openpyxl
- **Viz**: Plotly, Streamlit
