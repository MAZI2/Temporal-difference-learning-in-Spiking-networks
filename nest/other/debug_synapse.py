import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------------
# Config
# ----------------------------------
LOG_FILE = "out.txt"   # or "c_delayed_out.txt"
start =1500                     # time filter
end = 2000
plot_ix = 0                      # index within each t_trig block
pattern = re.compile(
    r"post_node_id=(\d+).*?t_trig=(\d+).*?c_delayed=([\-0-9.eE]+)"
)
# (use c_delayed=... instead of c_current=... if plotting c_delayed)

# ----------------------------------
# Parse the log file
# ----------------------------------
with open(LOG_FILE, "r") as f:
    lines = f.readlines()

# Store (t_trig, c_val) for each neuron separately
neuron_data = defaultdict(list)

for line in lines:
    m = pattern.search(line)
    if not m:
        continue

    post_id = int(m.group(1))
    t_trig = int(m.group(2))
    c_val = float(m.group(3))

    if not (start < t_trig < end):
        continue

    neuron_data[post_id].append((t_trig, c_val))

# ----------------------------------
# Group by t_trig for each neuron (preserving order)
# ----------------------------------
neuron_blocks = {}

for post_id, entries in neuron_data.items():
    blocks = []
    current_ttrig = None
    current_block = []

    for (t_trig, c_val) in entries:
        if current_ttrig is None:
            current_ttrig = t_trig
            current_block = [c_val]
        elif t_trig == current_ttrig:
            current_block.append(c_val)
        else:
            blocks.append((current_ttrig, current_block))
            current_ttrig = t_trig
            current_block = [c_val]

    if current_block:
        blocks.append((current_ttrig, current_block))

    neuron_blocks[post_id] = blocks

# ----------------------------------
# Plot stacked vertically (one subplot per neuron)
# ----------------------------------
fig, axes = plt.subplots(len(neuron_blocks), 1, figsize=(8, 4 * len(neuron_blocks)), sharex=True)

if len(neuron_blocks) == 1:
    axes = [axes]

for ax, (post_id, blocks) in zip(axes, sorted(neuron_blocks.items())):
    if not blocks:
        continue

    min_len = min(len(block[1]) for block in blocks)
    if plot_ix >= min_len:
        print(f"Skipping neuron {post_id}: plot_ix={plot_ix} out of range (max {min_len-1})")
        continue

    x_vals = [b[0] for b in blocks]
    y_vals = [b[1][plot_ix] for b in blocks]

    ax.plot(x_vals, y_vals, color='purple', label=f"post_node_id={post_id}")
    ax.set_title(f"Neuron {post_id} — index {plot_ix} (t_trig {start}-{end})")
    ax.set_ylabel("c_current")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("t_trig")

plt.tight_layout()
plt.show()
