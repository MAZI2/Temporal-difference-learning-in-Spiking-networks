import pickle
import numpy as np
import matplotlib.pyplot as plt

grid_size = (4, 4)
start = (0, 0)
goal = (3, 3)

# ============================================================
# PLOT POLICY (ARROWS ONLY)
# ============================================================
def plot_policy(input_to_motor, input_map, motor_map, grid_size=(4, 4)):
    """
    Plot arrows showing the policy, with cell color representing
    the maximum difference between outgoing motor weights.
    """

    rows, cols = grid_size
    fig, ax = plt.subplots(figsize=(cols, rows))

    # --------------------------------------------------------
    # Convert input → motor connections
    # --------------------------------------------------------
    sources_motor = []
    targets_motor = []
    weights_motor = []

    for s, t, w in zip(input_to_motor["source"],
                       input_to_motor["target"],
                       input_to_motor["weight"]):
        if s in input_map and t in motor_map:
            sources_motor.append(input_map[s])
            targets_motor.append(motor_map[t])
            weights_motor.append(w)

    sources_motor = np.array(sources_motor)
    targets_motor = np.array(targets_motor)
    weights_motor = np.array(weights_motor)

    # Motor direction vectors (unchanged)
    motor_dxdy = {
        0: (0, 1),   # right
        1: (0, -1),  # left
        2: (-1, 0),  # up
        3: (1, 0)    # down
    }

    # --------------------------------------------------------
    # Precompute max difference per cell for coloring
    # --------------------------------------------------------
    maxdiff_map = {}

    for cell in range(rows * cols):
        # Collect weights to motors 0..3
        wvals = []
        for m in range(4):
            mask = (sources_motor == cell) & (targets_motor == m)
            if np.any(mask):
                wvals.append(np.mean(weights_motor[mask]))

        if len(wvals) >= 2:
            maxdiff = max(wvals) - min(wvals)
        else:
            maxdiff = 0.0

        maxdiff_map[cell] = maxdiff

    # Normalize color range
    diffs = np.array(list(maxdiff_map.values()))
    vmin, vmax = diffs.min(), diffs.max() if diffs.max() > 0 else 1e-6

    # --------------------------------------------------------
    # Draw cells + arrows
    # --------------------------------------------------------
    for i in range(rows):
        for j in range(cols):
            input_idx = i * cols + j
            diff = maxdiff_map[input_idx]

            # Normalize to [0,1]
            norm = (diff - vmin) / (vmax - vmin)

            # draw colored cell **instead of white**

            if (i, j) == goal:
                # draw green goal cell
                ax.add_patch(
                    plt.Rectangle(
                        (j, rows - i - 1), 1, 1,
                        color="green",
                        alpha=0.8,
                        edgecolor="none"
                    )
                )
                draw_arrow = False
            else:
                # normal white background
                ax.add_patch(
                    plt.Rectangle(
                        (j, rows - i - 1), 1, 1,
                        color=plt.cm.plasma(norm),
                        alpha=0.85,
                        edgecolor="none"
                    )
                )
                draw_arrow = True

            # --------------------------------------------------------
            # Compute combined motor direction (UNCHANGED)
            # --------------------------------------------------------
            arrow_dx = arrow_dy = 0.0
            max_arrow_length = 0.35

            for m in range(4):
                mask = (sources_motor == input_idx) & (targets_motor == m)
                if np.any(mask):
                    avg_w = np.mean(weights_motor[mask])
                    dx, dy = motor_dxdy[m]
                    arrow_dx += dx * avg_w
                    arrow_dy += dy * avg_w

            # Normalize arrow
            L = np.hypot(arrow_dx, arrow_dy)
            if draw_arrow and L > 0:
                arrow_dx /= L
                arrow_dy /= L

                ax.arrow(
                    j + 0.5,
                    rows - i - 0.5,
                    arrow_dx * max_arrow_length,
                    arrow_dy * max_arrow_length,
                    head_width=0.08,
                    head_length=0.1,
                    width=0.02,
                    fc="black",
                    ec="black"
                )

    # Draw grid lines
    for x in range(cols + 1):
        ax.axvline(x, color="k", lw=1)
    for y in range(rows + 1):
        ax.axhline(y, color="k", lw=1)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Max difference between motor weights", rotation=90)

    plt.show()


# ============================================================
# LOAD CONNECTION DATA
# ============================================================
with open("connections.pkl", "rb") as f:
    connections_data = pickle.load(f)

input_to_motor = connections_data["input_to_motor"]


# ============================================================
# MAPS FOR 4×4 GRID
# ============================================================
motor_map = {
    92: 0,
    93: 1,
    94: 2,
    95: 3
}

input_map = {
    44: 0,
    45: 1,
    46: 2,
    47: 3,
    48: 4,
    49: 5,
    50: 6,
    51: 7,
    52: 8,
    53: 9,
    54: 10,
    55: 11,
    56: 12,
    57: 13,
    58: 14,
    59: 15
}


def print_weight_table(input_to_motor, input_map, motor_map):
    """
    Prints a formatted table of average weights:
        Input local index  →  Motor local index
    """

    sources = np.array(input_to_motor["source"])
    targets = np.array(input_to_motor["target"])
    weights = np.array(input_to_motor["weight"])

    # Filter for actual connections in the maps
    valid_mask = np.isin(sources, list(input_map.keys())) & \
                 np.isin(targets, list(motor_map.keys()))

    sources = sources[valid_mask]
    targets = targets[valid_mask]
    weights = weights[valid_mask]

    # Unique local indices
    unique_inputs = sorted(input_map.values())
    unique_motors = sorted(motor_map.values())

    # Initialize matrix
    avg_matrix = np.zeros((len(unique_inputs), len(unique_motors)))

    for src_global, src_local in input_map.items():
        for tgt_global, tgt_local in motor_map.items():
            mask = (sources == src_global) & (targets == tgt_global)
            if np.any(mask):
                avg_matrix[src_local, tgt_local] = np.mean(weights[mask])
            else:
                avg_matrix[src_local, tgt_local] = np.nan  # no connection

    # -----------------------------
    # PRINT TABLE
    # -----------------------------
    header = "Input\\Motor | " + "  ".join([f"{m:>10}" for m in unique_motors])
    print("\n" + header)
    print("-" * len(header))

    for i, src_local in enumerate(unique_inputs):
        row_vals = "  ".join(
            f"{avg_matrix[i, j]:10.3f}" if not np.isnan(avg_matrix[i, j]) else "     ---   "
            for j in range(len(unique_motors))
        )
        print(f"{src_local:>11} | {row_vals}")

    print()

print_weight_table(input_to_motor, input_map, motor_map)

# ============================================================
# PLOT GRID POLICY (ARROWS ONLY)
# ============================================================
plot_policy(
    input_to_motor=input_to_motor,
    input_map=input_map,
    motor_map=motor_map,
    grid_size=grid_size
)


