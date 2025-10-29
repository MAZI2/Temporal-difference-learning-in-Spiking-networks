import argparse
import datetime
import gzip
import logging
import os
import pickle
import sys
import time
import io
import re
from contextlib import contextmanager
import random

import nest
import numpy as np
import matplotlib.pyplot as plt

grid_size = (3, 3)
start = (0, 0)
goal = (2, 2)


def plot_policy(input_to_motor, input_to_striatum, input_map, motor_map, grid_size=(3,3)):
    """
    Plot gridworld policy based on input->motor weights and color squares based on input->striatum weights.

    input_to_motor : dict with keys 'source', 'target', 'weight'
    input_to_striatum : dict with keys 'source', 'target', 'weight'
    input_map : dict mapping global input neuron IDs to local indices
    motor_map : dict mapping global motor neuron IDs to local indices
    grid_size : (rows, cols)
    """

    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(cols, rows))

    # --- Convert global IDs to local indices ---
    sources_motor = np.array([input_map[s] for s in input_to_motor["source"] if s in input_map])
    targets_motor = np.array([motor_map[t] for t in input_to_motor["target"] if t in motor_map])
    weights_motor = np.array([w for s, t, w in zip(input_to_motor["source"], input_to_motor["target"], input_to_motor["weight"])
                              if s in input_map and t in motor_map])

    sources_str = np.array([input_map[s] for s in input_to_striatum["source"] if s in input_map])
    weights_str = np.array([w for s, w in zip(input_to_striatum["source"], input_to_striatum["weight"]) if s in input_map])

    # --- Compute average input→striatum weight (for color) ---
    avg_str_weights = {src: np.mean(weights_str[sources_str == src]) for src in np.unique(sources_str)}
    min_w, max_w = np.min(list(avg_str_weights.values())), np.max(list(avg_str_weights.values()))
    if max_w == min_w:  # Avoid divide by zero
        max_w += 1e-6

    # --- Motor direction mapping ---
    motor_dxdy = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}

    # --- Plot ---
    for i in range(rows):
        for j in range(cols):
            input_idx = i * cols + j

            # Background color from striatum weight
            w_norm = (avg_str_weights.get(input_idx, min_w) - min_w) / (max_w - min_w)
            ax.add_patch(plt.Rectangle((j, rows-i-1), 1, 1, color=plt.cm.Blues(w_norm), alpha=0.7))

            # Compute directional weights to motors
            max_arrow_length = 0.3
            arrow_dx, arrow_dy = 0.0, 0.0
            for m in range(4):
                mask = (sources_motor == input_idx) & (targets_motor == m)
                if np.any(mask):
                    avg_w = np.mean(weights_motor[mask])
                    dx, dy = motor_dxdy[m]
                    arrow_dx += dx * avg_w
                    arrow_dy += dy * avg_w

            length = np.sqrt(arrow_dx**2 + arrow_dy**2)
            if length > 0:
                arrow_dx /= length
                arrow_dy /= length
                ax.arrow(j+0.5, rows-i-0.5, arrow_dx*max_arrow_length, arrow_dy*max_arrow_length,
                         head_width=0.05, head_length=0.05, width=0.005, fc='k', ec='k')

               # --- Grid styling ---
    for x in range(cols+1):
        ax.axvline(x, color="k", lw=1)
    for y in range(rows+1):
        ax.axhline(y, color="k", lw=1)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()


#game = gridworld.GridWorld(size=grid_size, start=start, goal=goal)
#state = game.reset()
#player = GridWorldAC(False)

with open("connections.pkl", "rb") as f:
    connections_data = pickle.load(f)

input_to_motor = connections_data["input_to_motor"]
input_to_striatum = connections_data["input_to_striatum"]

sources = np.array([c for c in input_to_motor["source"]])
targets = np.array([c for c in input_to_motor["target"]])
weights = np.array([c for c in input_to_motor["weight"]])

unique_sources = np.unique(sources)
avg_weights_per_input = {}

for src in unique_sources:
    mask = sources == src
    avg_weight = np.mean(weights[mask])
    avg_weights_per_input[src] = avg_weight

# Print results
for src, avg_w in avg_weights_per_input.items():
    print(f"Input neuron {src}: average weight to all motor neurons = {avg_w:.3f}")

# 3x3
motor_map = {27: 0, 28: 1, 29: 2, 30: 3}
input_map = {
    18: 0, 19: 1, 20: 2, 21: 3, 22: 4,
    23: 5, 24: 6, 25: 7, 26: 8
}
"""
# 4x4
motor_map = {
    33: 0,
    34: 1,
    35: 2,
    36: 3
}

input_map = {
    17: 0,
    18: 1,
    19: 2,
    20: 3,
    21: 4,
    22: 5,
    23: 6,
    24: 7,
    25: 8,
    26: 9,
    27: 10,
    28: 11,
    29: 12,
    30: 13,
    31: 14,
    32: 15
}
"""
"""
#5x5
motor_map = {
    51: 0,
    52: 1,
    53: 2,
    54: 3
}

input_map = {
    26: 0,
    27: 1,
    28: 2,
    29: 3,
    30: 4,
    31: 5,
    32: 6,
    33: 7,
    34: 8,
    35: 9,
    36: 10,
    37: 11,
    38: 12,
    39: 13,
    40: 14,
    41: 15,
    42: 16,
    43: 17,
    44: 18,
    45: 19,
    46: 20,
    47: 21,
    48: 22,
    49: 23,
    50: 24
}
"""

# --- Compute and print average input→striatum weights ---
sources_str = np.array(connections_data["input_to_striatum"]["source"])
targets_str = np.array(connections_data["input_to_striatum"]["target"])
weights_str = np.array(connections_data["input_to_striatum"]["weight"])

valid_mask_str = np.isin(sources_str, list(input_map.keys()))
sources_str = sources_str[valid_mask_str]
weights_str = weights_str[valid_mask_str]

unique_sources_str = np.unique(sources_str)
avg_weights_per_input_str = {}

for src in unique_sources_str:
    mask = sources_str == src
    avg_weight = np.mean(weights_str[mask])
    avg_weights_per_input_str[src] = avg_weight

print("\nAverage input→striatum weights:\n")
for src_global, avg_w in avg_weights_per_input_str.items():
    src_local = input_map.get(src_global, None)
    if src_local is not None:
        print(f"Input neuron {src_global} (local {src_local}): average weight to striatum = {avg_w:.3f}")




# Filter only connections that exist in these maps
valid_mask = np.isin(sources, list(input_map.keys())) & np.isin(targets, list(motor_map.keys()))
sources = sources[valid_mask]
targets = targets[valid_mask]
weights = weights[valid_mask]

# Unique local indices
unique_inputs = sorted(set(input_map[g] for g in sources))
unique_motors = sorted(set(motor_map[g] for g in targets))

# Initialize matrix (rows = inputs, columns = motors)
avg_weight_matrix = np.zeros((len(unique_inputs), len(unique_motors)))

# Compute average weights
for src_global, src_local in input_map.items():
    for tgt_global, tgt_local in motor_map.items():
        mask = (sources == src_global) & (targets == tgt_global)
        if np.any(mask):
            avg_weight_matrix[src_local, tgt_local] = np.mean(weights[mask])
        else:
            avg_weight_matrix[src_local, tgt_local] = np.nan

# Print formatted table with local indices
print("\nAverage weights (Input local index → Motor local index):\n")
header = "Input\\Motor | " + "  ".join([f"{j:>8}" for j in unique_motors])
print(header)
print("-" * len(header))

for i, src_local in enumerate(unique_inputs):
    row_vals = "  ".join([
        f"{avg_weight_matrix[i, j]:8.3f}" if not np.isnan(avg_weight_matrix[i, j]) else "   ---  "
        for j in range(len(unique_motors))
    ])
    print(f"{src_local:>11} | {row_vals}")

# # Optional: visualize as heatmap
# try:
#     import matplotlib.pyplot as plt
#     plt.imshow(avg_weight_matrix, cmap="viridis", interpolation="nearest")
#     plt.colorbar(label="Average Weight")
#     plt.xlabel("Motor neuron (local index)")
#     plt.ylabel("Input neuron (local index)")
#     plt.title("Average Input→Motor Weights")
#     plt.show()
# except ImportError:
#     pass

plot_policy(
    input_to_motor=connections_data["input_to_motor"],
    input_to_striatum=connections_data["input_to_striatum"],
    input_map=input_map,
    motor_map=motor_map,
    grid_size=grid_size
)

