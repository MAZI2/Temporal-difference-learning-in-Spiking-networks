# -*- coding: utf-8 -*-
#
# generate_gif_singleplayer.py
#
# Modified from original NEST Pong GIF generator for single-player Pong.
#
# Supports only:
#   - left paddle (RL agent)
#   - ball
#   - right-side wall instead of opponent paddle
#   - left player's network heatmap
#   - rewards over time
#
# Loads: experiment_output.pkl (your training output)
#

import gzip
import os
import pickle
import sys
from copy import copy
from glob import glob
import shutil

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from pong import GameOfPong as Pong

# -----------------------------
# Graphics configuration
# -----------------------------

gridsize = (12, 16)  # Plot layout grid

left_color = np.array((204, 0, 153))      # Purple
left_color_hex = "#cc0099"

white = np.array((255, 255, 255))
wall_color = np.array((150, 150, 150))     # Light gray wall

GAME_GRID = np.array([Pong.x_grid, Pong.y_grid])
GRID_SCALE = 24
GAME_GRID_SCALED = GAME_GRID * GRID_SCALE

BALL_RAD = 6
PADDLE_LEN = int(0.1 * GAME_GRID_SCALED[1])
PADDLE_WID = 18

FIELD_PADDING = PADDLE_WID * 2
FIELD_SIZE = copy(GAME_GRID_SCALED)
FIELD_SIZE[0] += 2 * FIELD_PADDING

DEFAULT_SPEED = 4


# -----------------------------
# Helpers
# -----------------------------

def scale_coordinates(coords: np.array):
    """Scale (x,y) coords from simulation units to pixel units."""
    coords[:, 0] = coords[:, 0] * GAME_GRID_SCALED[0] / Pong.x_length + FIELD_PADDING
    coords[:, 1] = coords[:, 1] * GAME_GRID_SCALED[1] / Pong.y_length
    return coords.astype(int)


def grayscale_to_heatmap(in_img, min_val, max_val, base_color):
    """Convert grayscale weight matrix into a colored heatmap."""
    x_len, y_len = in_img.shape
    out_img = np.ones((x_len, y_len, 3), dtype=np.uint8)

    span = max_val - min_val
    if span == 0:
        return out_img * base_color

    for x in range(x_len):
        for y in range(y_len):
            scale = (in_img[x, y] - min_val) / span
            out_img[x, y, :] = base_color + (white - base_color) * scale

    return out_img


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    keep_temps = False
    out_file = "pong_singleplayer.gif"

    if len(sys.argv) != 2:
        print("Usage: python generate_gif_singleplayer.py <folder_with_experiment_output.pkl>")
        sys.exit(1)

    input_folder = sys.argv[1]

    if os.path.exists(out_file):
        print(f"<{out_file}> already exists, aborting!")
        sys.exit(1)

    temp_dir = "temp_singleplayer"
    if os.path.exists(temp_dir):
        print(f"Temporary folder <{temp_dir}> already exists, removing it...")
        shutil.rmtree(temp_dir)

    # Create a fresh temp folder
    os.mkdir(temp_dir)

    print(f"Reading simulation data from {input_folder}/experiment_output.pkl")

    with open(os.path.join(input_folder, "experiment_output.pkl"), "rb") as f:
        data = pickle.load(f)

    # -----------------------------
    # Extract game trajectory
    # -----------------------------
    ball_positions = np.array([list(pos) for pos in data["ball_pos"]], dtype=float)
    l_paddle_positions = np.array([list(pos) for pos in data["left_paddle"]], dtype=float)
    r_paddle_positions = np.array([list(pos) for pos in data["right_paddle"]], dtype=float)

    ball_positions = scale_coordinates(ball_positions)
    l_paddle_positions = scale_coordinates(l_paddle_positions)

    # No right paddle — produce no coordinates
    r_paddle_positions = None

    score = np.array(data["score"]).astype(int)

    # -----------------------------
    # Extract network info
    # -----------------------------

    net = data["network"]
    weights_left = np.array(net["weights"].tolist(), dtype=float)
    rewards_left = np.array(net["rewards"]).astype(float)
    name_left = net["network_type"]

    min_w, max_w = np.min(weights_left), np.max(weights_left)

    print(f"Loaded weights: min={min_w}, max={max_w}")
    print(f"Loaded {len(score)} simulation steps.")

    # -----------------------------
    # Generate images
    # -----------------------------

    print(f"Rendering frames into: {temp_dir}")

    n_iterations = score.shape[0]
    i = 0
    output_speed = DEFAULT_SPEED

    while i < n_iterations:
        px = 1 / plt.rcParams["figure.dpi"]
        fig, ax = plt.subplots(figsize=(400 * px, 300 * px))
        ax.set_axis_off()
        plt.rcParams.update({"font.size": 6})

        # -----------------------------
        # Set up layout grid
        # -----------------------------
        title = plt.subplot2grid(gridsize, (0, 0), 1, 16)
        l_info = plt.subplot2grid(gridsize, (1, 0), 7, 2)
        field = plt.subplot2grid(gridsize, (1, 2), 7, 12)
        l_hm = plt.subplot2grid(gridsize, (8, 0), 4, 4)
        reward_plot = plt.subplot2grid(gridsize, (8, 6), 4, 6)

        for axx in [title, l_info, field, l_hm]:
            axx.axis("off")

        # -----------------------------
        # Draw playing field
        # -----------------------------
        playing_field = np.zeros((FIELD_SIZE[0], FIELD_SIZE[1], 3), dtype=np.uint8)

        # Wall at right side
        wall_x0 = FIELD_PADDING + GAME_GRID_SCALED[0] - PADDLE_WID
        playing_field[wall_x0:wall_x0 + PADDLE_WID, :] = wall_color

        # Ball
        x, y = ball_positions[i]
        playing_field[x - BALL_RAD:x + BALL_RAD, y - BALL_RAD:y + BALL_RAD] = white

        # Left paddle
        px_l, py_l = l_paddle_positions[i]
        py_l = max(PADDLE_LEN, py_l)
        py_l = min(FIELD_SIZE[1] - PADDLE_LEN, py_l)
        playing_field[px_l:px_l + PADDLE_WID, py_l - PADDLE_LEN:py_l + PADDLE_LEN] = left_color

        field.imshow(np.transpose(playing_field, [1, 0, 2]))

        # -----------------------------
        # Heatmap (left network)
        # -----------------------------
        w = np.array(weights_left[i], dtype=float)
        if w.ndim == 1:
            w = w.reshape(1, -1)  # 1 row, num_motor_neurons columns
        heatmap_l = grayscale_to_heatmap(w, min_w, max_w, left_color)
        l_hm.imshow(heatmap_l)
        l_hm.set_xlabel("output")
        l_hm.set_ylabel("input")
        l_hm.set_title("left weights", y=-0.3)

        # -----------------------------
        # Rewards
        # -----------------------------
        reward_plot.plot(rewards_left[: i + 1], color=left_color / 255)
        reward_plot.set_ylabel("reward")
        reward_plot.set_ylim(0, 1.0)

        reward_plot.set_xlim(0, max(n_iterations, 10))

        # -----------------------------
        # Title + info panel
        # -----------------------------
        title.text(0.5, 0.75, name_left, ha="center", fontsize=16, color=left_color_hex)

        l_score = score[i]

        l_info.text(0, 0.9, "step:", fontsize=14)
        l_info.text(0, 0.75, str(i), fontsize=14)
        l_info.text(1, 0.5, f"score:{l_score}", ha="right", va="center", fontsize=18, c=left_color_hex)

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.35, hspace=0.35)
        plt.savefig(os.path.join(temp_dir, f"img_{str(i).zfill(6)}.png"))
        plt.close()

        # Speed control
        if 75 <= i < 100 or n_iterations - 400 <= i < n_iterations - 350:
            output_speed = 10
        elif 100 <= i < n_iterations - 350:
            output_speed = 50
        else:
            output_speed = DEFAULT_SPEED

        i += output_speed

    # -----------------------------
    # Build GIF
    # -----------------------------
    print("Collecting frames into GIF...")

    filenames = sorted(glob(os.path.join(temp_dir, "*.png")))

    with imageio.get_writer(out_file, mode="I", duration=150) as writer:
        for filename in filenames:
            img = imageio.imread(filename)
            writer.append_data(img)

    print(f"GIF created: {out_file}")

    # Cleanup
    if not keep_temps:
        print("Cleaning up temporary files...")
        for f in filenames:
            os.unlink(f)
        os.rmdir(temp_dir)

    print("Done.")

