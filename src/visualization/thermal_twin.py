"""
Thermal Digital Twin - 3D visualization of "Hot Alleys" in data center floor.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def compute_thermal_grid(
    power_map: np.ndarray,
    cooling_map: np.ndarray,
    dimensions: Tuple[int, int, int] = (30, 20, 5),
    grid_res: int = 20,
) -> np.ndarray:
    """
    Compute 3D temperature field from power (heat source) and cooling distribution.
    Simplified diffusion: temp rises with power, decreases with cooling.
    """
    nx, ny, nz = grid_res, grid_res // 2, grid_res // 4
    temp = np.ones((nx, ny, nz)) * 25
    pw = np.zeros((nx, ny, nz))
    cl = np.zeros((nx, ny, nz))
    if power_map.size > 0:
        pw_2d = np.array(power_map, dtype=float)
        if pw_2d.ndim == 1:
            side = int(np.sqrt(len(pw_2d))) or 1
            pw_2d = np.resize(pw_2d, side * side).reshape(side, side)
        h, w = pw_2d.shape
        for i in range(nx):
            for j in range(ny):
                pw[i, j, -1] = pw_2d[min(int(i * h / nx), h - 1), min(int(j * w / ny), w - 1)]
    if cooling_map.size > 0:
        cl_2d = np.array(cooling_map, dtype=float)
        if cl_2d.ndim == 1:
            side = int(np.sqrt(len(cl_2d))) or 1
            cl_2d = np.resize(cl_2d, side * side).reshape(side, side)
        h, w = cl_2d.shape
        for i in range(nx):
            for j in range(ny):
                cl[i, j, 0] = cl_2d[min(int(i * h / nx), h - 1), min(int(j * w / ny), w - 1)]
    # Simple steady-state: temp += power_effect - cooling_effect
    for _ in range(5):
        temp = temp + 0.2 * pw - 0.3 * cl
        temp = np.clip(temp, 18, 45)
    return temp


def create_thermal_twin_figure(
    power_map: Optional[np.ndarray] = None,
    cooling_map: Optional[np.ndarray] = None,
    grid_res: int = 20,
) -> go.Figure:
    """
    Create interactive 3D Plotly figure showing thermal distribution (Hot Alleys).
    """
    if power_map is None:
        # Synthetic rack heat distribution (hot aisle / cold aisle pattern)
        n = grid_res
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, y)
        power_map = 0.5 + 0.5 * np.sin(4 * np.pi * xx) * np.cos(2 * np.pi * yy)
    if cooling_map is None:
        cooling_map = 0.3 + 0.2 * np.random.rand(*power_map.shape)

    temp = compute_thermal_grid(power_map, cooling_map, grid_res=grid_res)

    # Flatten for volume/mesh
    nx, ny, nz = temp.shape
    x = np.linspace(0, 30, nx)
    y = np.linspace(0, 20, ny)
    z = np.linspace(0, 5, nz)

    # Slice at rack height (z=2.5m) for top-down "hot alley" view
    z_idx = nz // 2
    slice_2d = temp[:, :, z_idx]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Floor Plan - Thermal Map (Hot Alleys)", "3D Thermal Distribution"),
        specs=[[{"type": "heatmap"}, {"type": "surface"}]],
    )
    fig.add_trace(
        go.Heatmap(z=slice_2d, x=x, y=y, colorscale="Hot", colorbar=dict(title="°C")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Surface(z=slice_2d, x=x, y=y, colorscale="Hot", colorbar=dict(title="°C")),
        row=1, col=2,
    )
    fig.update_layout(
        title="TermoGrid AI - Thermal Digital Twin",
        height=500,
        template="plotly_dark",
    )
    return fig


def create_pue_forecast_chart(
    timestamps: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> go.Figure:
    """Create PUE forecast vs actual chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=actual, name="Actual PUE", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=timestamps, y=predicted, name="Predicted PUE", line=dict(color="orange", dash="dash")))
    fig.update_layout(
        title="PUE Forecaster - 15 min Horizon",
        xaxis_title="Time",
        yaxis_title="PUE",
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig
