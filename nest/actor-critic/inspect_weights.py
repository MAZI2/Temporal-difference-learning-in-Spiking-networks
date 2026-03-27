import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

grid_size = (4, 4)
start = (0, 0)
goal = (3, 3)
STR_MIN = 150   # minimum weight to show (anything below = min color)
STR_MAX = 450  # maximum weight to show (anything above = max color)


# ============================================================
# PLOT POLICY
# ============================================================
def plot_policy(input_to_motor, input_to_striatum, input_map, input_raw_map, motor_map, grid_size=(4, 4)):
    """
    Plot gridworld policy based on:
      • input → motor weights  (using input_map)
      • input → striatum weights (using input_raw_map)

    input_to_motor : dict with keys ["source","target","weight"]
    input_to_striatum : same
    input_map : global → local index for input→motor
    input_raw_map : global → local index for input→striatum
    motor_map : global → local index
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

    # --------------------------------------------------------
    # Convert input → striatum connections (uses INPUT_RAW_MAP!)
    # --------------------------------------------------------
    sources_str = []
    weights_str = []

    for s, w in zip(input_to_striatum["source"],
                    input_to_striatum["weight"]):
        if s in input_raw_map:
            sources_str.append(input_raw_map[s])
            weights_str.append(w)

    sources_str = np.array(sources_str)
    weights_str = np.array(weights_str)

    # Compute averages per input cell
    if len(sources_str) > 0:
        avg_str_weights = {src: np.mean(weights_str[sources_str == src])
                           for src in np.unique(sources_str)}
        min_w = np.min(list(avg_str_weights.values()))
        max_w = np.max(list(avg_str_weights.values()))
        if max_w == min_w:
            max_w += 1e-6
    else:
        # fallback: no striatum weights
        avg_str_weights = {i: 0.0 for i in range(rows * cols)}
        min_w, max_w = 0, 1

    # Motor direction mapping
    motor_dxdy = {
        0: (0, 1),   # right
        1: (0, -1),  # left
        2: (-1, 0),  # up
        3: (1, 0)    # down
    }

    # --------------------------------------------------------
    # Draw cells
    # --------------------------------------------------------
    for i in range(rows):
        for j in range(cols):
            input_idx = i * cols + j

            # background color based on striatum weight
            #w_norm = (avg_str_weights.get(input_idx, min_w) - min_w) / (max_w - min_w)
            w = avg_str_weights.get(input_idx, 0.0)
            w_clamped = np.clip(w, STR_MIN, STR_MAX)  # clamp
            w_norm = (w_clamped - STR_MIN) / (STR_MAX - STR_MIN)  # normalize within fixed range
            
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
                        color=plt.cm.plasma(w_norm),
                        alpha=0.85,
                        edgecolor="none"
                    )
                )
                draw_arrow = True

            # directional arrows
            arrow_dx = arrow_dy = 0.0
            max_arrow_length = 0.3

            for m in range(4):
                mask = (sources_motor == input_idx) & (targets_motor == m)
                if np.any(mask):
                    avg_w = np.mean(weights_motor[mask])
                    dx, dy = motor_dxdy[m]
                    arrow_dx += dx * avg_w
                    arrow_dy += dy * avg_w

            L = np.hypot(arrow_dx, arrow_dy)
            if draw_arrow and L > 0:
                arrow_dx /= L
                arrow_dy /= L
                ax.arrow(
                    j + 0.5,
                    rows - i - 0.5,
                    arrow_dx * max_arrow_length,
                    arrow_dy * max_arrow_length,
                    head_width=0.05,
                    head_length=0.05,
                    width=0.005,
                    fc='k',
                    ec='k'
                )

    # Draw grid
    for x in range(cols + 1):
        ax.axvline(x, color="k", lw=1)
    for y in range(rows + 1):
        ax.axhline(y, color="k", lw=1)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    # --------------------------------------------------------
    # Colorbar for striatum weights
    # --------------------------------------------------------
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=STR_MIN, vmax=STR_MAX)
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Input → Striatum Weight (pA)")
    plt.show()


# ============================================================
# LOAD CONNECTION DATA
# ============================================================
with open("conns/0_4000_0.3_0_0_0-connections.pkl", "rb") as f:
    connections_data = pickle.load(f)

input_to_motor = connections_data["input_to_motor"]
input_to_striatum = connections_data["input_to_striatum"]


# ============================================================
# MAPS FOR 4×4 GRID
# ============================================================
motor_map = {
    57: 0,
    58: 1,
    59: 2,
    60: 3
}

# used for input → motor
input_map = {
    41: 0,
    42: 1,
    43: 2,
    44: 3,
    45: 4,
    46: 5,
    47: 6,
    48: 7,
    49: 8,
    50: 9,
    51: 10,
    52: 11,
    53: 12,
    54: 13,
    55: 14,
    56: 15
}

# used for input → striatum
input_raw_map = {
    25: 0,
    26: 1,
    27: 2,
    28: 3,
    29: 4,
    30: 5,
    31: 6,
    32: 7,
    33: 8,
    34: 9,
    35: 10,
    36: 11,
    37: 12,
    38: 13,
    39: 14,
    40: 15
}


# ============================================================
# PRINT AVERAGE INPUT→STRIATUM WEIGHTS
# ============================================================
sources_str = np.array(input_to_striatum["source"])
weights_str = np.array(input_to_striatum["weight"])

valid = np.isin(sources_str, list(input_raw_map.keys()))
sources_str = sources_str[valid]
weights_str = weights_str[valid]

print("\nAverage input→striatum weights:\n")
for src in np.unique(sources_str):
    avg_w = np.mean(weights_str[sources_str == src])
    print(f"Input neuron {src} (local {input_raw_map[src]}): avg weight = {avg_w:.3f}")


# Filter only connections that exist in the maps
sources = np.array(connections_data["input_to_motor"]["source"])
targets = np.array(connections_data["input_to_motor"]["target"])
weights = np.array(connections_data["input_to_motor"]["weight"])

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

# Print formatted table
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



# ============================================================
# PLOT GRID POLICY
# ============================================================
plot_policy(
    input_to_motor=input_to_motor,
    input_to_striatum=input_to_striatum,
    input_map=input_map,
    input_raw_map=input_raw_map,
    motor_map=motor_map,
    grid_size=grid_size
)

