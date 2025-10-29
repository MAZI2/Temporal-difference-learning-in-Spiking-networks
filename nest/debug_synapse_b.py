import re
import matplotlib.pyplot as plt

#LOG_FILE = "c_delayed_out.txt"
LOG_FILE = "c_current_out.txt"

# Regex to extract t_trig and c_delayed
#pattern = re.compile(r"t_trig=(\d+).*?c_delayed=([\-0-9.eE]+)")
pattern = re.compile(r"t_trig=(\d+).*?c_current=([\-0-9.eE]+)")

# Parse the log file
with open(LOG_FILE, "r") as f:
    lines = f.readlines()

# Collect per t_trig in order
blocks = []
current_ttrig = None
current_block = []

for ix, line in enumerate(lines):
    m = pattern.search(line)

    if not m:
        continue

    t_trig = int(m.group(1))
    c_delayed = float(m.group(2))

    if current_ttrig is None:
        current_ttrig = t_trig
        current_block = [c_delayed]
    elif t_trig == current_ttrig:
        current_block.append(c_delayed)
    else:
        blocks.append((current_ttrig, current_block))
        current_ttrig = t_trig
        current_block = [c_delayed]

if current_block:
    blocks.append((current_ttrig, current_block))

print(f"Found {len(blocks)} t_trig blocks")
print("block", blocks[0])

# We need to align by index — all blocks should be same length
min_len = min(len(block[1]) for block in blocks)
print(f"Using first {min_len} entries per block for alignment")

# Build series per line index
index_series = []
for i in range(min_len):
    x_vals = [b[0] for b in blocks]  # t_trig values
    y_vals = [b[1][i] for b in blocks]
    index_series.append((x_vals, y_vals))

# Plot each line index as its own subplot
fig, axes = plt.subplots(min_len, 1, figsize=(8, 2 * min_len), sharex=True)
if min_len == 1:
    axes = [axes]

for i, (ax, (x_vals, y_vals)) in enumerate(zip(axes, index_series)):
    ax.plot(x_vals, y_vals, color='purple')
    ax.set_title(f"Line index {i}")
    ax.set_ylabel("c_delayed")
    ax.grid(True)

axes[-1].set_xlabel("t_trig")

plt.tight_layout()
plt.show()
