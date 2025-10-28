import subprocess
import re
import matplotlib.pyplot as plt

# 1️⃣ Run the main simulation script
result = subprocess.run(
    ["python3", "test.py"],
    capture_output=True,
    text=True,
    bufsize=1,  # line-buffered
)

# Combine stdout + stderr
output = result.stdout + result.stderr

# 2️⃣ Regex to capture debug info
debug_pattern = re.compile(
    r"\[DEBUG trigger_update_weight\]\s*\|\s*post_node_id=(\d+)\s*\|\s*t_trig=(\S+)\s*\|.*?c_delayed=(\S+)"
)

# Extract all matches
debug_data = [
    (int(m.group(1)), float(m.group(2)), float(m.group(3)))
    for m in debug_pattern.finditer(output)
]

# 3️⃣ Filter for neurons of interest
target_neurons = [19, 20, 21, 22]
neuron_data = {nid: [] for nid in target_neurons}

for post_id, t, c in debug_data:
    if post_id in neuron_data:
        neuron_data[post_id].append((t, c))

# 4️⃣ Plot results
valid_neurons = [nid for nid, data in neuron_data.items() if data]

if valid_neurons:
    fig, axes = plt.subplots(len(valid_neurons), 1, figsize=(10, 2 * len(valid_neurons)), sharex=True)

    if len(valid_neurons) == 1:
        axes = [axes]  # Ensure iterable if only one neuron

    for ax, nid in zip(axes, valid_neurons):
        data = sorted(neuron_data[nid], key=lambda x: x[0])  # sort by time
        time_points, c_delayed_values = zip(*data)
        ax.plot(time_points, c_delayed_values, label=f"Neuron {nid}", color="purple")
        ax.set_ylabel("c_delayed")
        ax.set_title(f"Eligibility trace for post neuron {nid}")
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()
else:
    print("No c_delayed entries found for neurons 19–22.")
