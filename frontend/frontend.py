 # %% Modules

import numpy as np
import matplotlib.pyplot as plt

from data.functions import polynomial
from data.loader import DEFAULT_COEFFICIENTS
from data.projection import find_projection
from src.inference import load_model, infer

# %% Load model

model = load_model()

# %% Setup figure

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title("Move mouse to see predicted projection and expert assignment")

# %% Plot expert polynomials

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
x_curve = np.linspace(-3.0, 3.0, 300)

for j, coeff in enumerate(DEFAULT_COEFFICIENTS):
    y_curve = polynomial(x_curve, coeff)
    ax.plot(x_curve, y_curve, color=colors[j], linewidth=2, label=f"Expert {j}")

# %% Plot initial point

point_drawing, = ax.plot([], [], "ko", markersize=10, zorder=5, label="point")

# %% Model projection

model_proj_drawing, = ax.plot([], [], "s", markersize=8, zorder=5, label="model projection")
model_proj_line, = ax.plot([], [], "--", alpha=0.5, linewidth=1)

# %% Analytic projection

analytic_proj_drawing, = ax.plot([], [], "^", markersize=8, zorder=5, label="analytic projection")
analytic_proj_line, = ax.plot([], [], ":", alpha=0.5, linewidth=1)

ax.legend(loc="upper right")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# %% Update event

analytic_guess = {}

def on_move(event):
    """
    Function to call when mouse moves
    """
    if event.inaxes != ax:
        return

    point = (event.xdata, event.ydata)

    # Model inference

    projections, expert_indices = infer(model, [point])

    model_proj = projections[0]
    expert = expert_indices[0]
    color = colors[expert]

    point_drawing.set_data([point[0]], [point[1]])
    model_proj_drawing.set_data([model_proj[0]], [model_proj[1]])
    model_proj_drawing.set_color(color)
    model_proj_line.set_data([point[0], model_proj[0]], [point[1], model_proj[1]])
    model_proj_line.set_color(color)

    # Analytic projection on the same expert's polynomial

    coeff = DEFAULT_COEFFICIENTS[expert]
    guess = analytic_guess.get(expert, point[0])
    ax_proj, ay_proj = find_projection(point, coeff, guess=guess)
    analytic_guess[expert] = ax_proj

    analytic_proj_drawing.set_data([ax_proj], [ay_proj])
    analytic_proj_drawing.set_color(color)
    analytic_proj_line.set_data([point[0], ax_proj], [point[1], ay_proj])
    analytic_proj_line.set_color(color)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.show()
