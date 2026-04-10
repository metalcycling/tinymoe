 # %% Modules

import numpy as np
import matplotlib.pyplot as plt

from data.functions import polynomial
from data.projection import find_projection

# %% Setup figure

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title("Move mouse to update the point and show the projection to the curve")

# %% Define polynomial function

coeff = np.array([0.5, 0.5, 0.0, 0.0])

x = np.linspace(-2.0, 2.0)
y = polynomial(x, coeff)

ax.plot(x, y, "b-", linewidth=2, label="polynomial")

# %% Plot initial point

point_drawing, = ax.plot([], [], "ro", markersize=10, zorder=5, label="point")

# %% Projection point

projection_drawing, = ax.plot([], [], "gs", markersize=8, zorder=5, label="projection")
projection = (0.0, 0.0)

# %% Line from point to projection

projection_line, = ax.plot([], [], "g--", alpha=0.5, linewidth=1)

ax.legend(loc="upper right")
ax.set_xlim(xmin=x[0], xmax=x[-1])

# %% Update event

def on_move(event):
    """
    Function to call when mouse moves
    """
    if event.inaxes != ax:
        return

    point = event.xdata, event.ydata

    # Update the red point

    point_drawing.set_data([point[0]], [point[1]])

    # Recompute projection point

    global projection

    projection = find_projection(point, coeff, guess=projection[0])
    projection_drawing.set_data([projection[0]], [projection[1]])

    # Update lines from point to projection

    projection_line.set_data([point[0], projection[0]], [point[1], projection[1]])

    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.show()
