# simplified_nest_delayed_traces.py
import nest
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from contextlib import contextmanager
from collections import defaultdict

# ---------- USER CONFIG: set presynaptic and postsynaptic spike times here ----------
# Times are in ms.
pre_spike_times = [10.0, 30.0]    # presynaptic spike times
post_spike_times = [12.0, 32.0]         # postsynaptic spike times you force manually

sim_duration = 150.0  # ms

# ---------- NEST / simulation parameters ----------
SEED = 12351
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": SEED})
np.random.seed(SEED)

# If your delayed synapse model is provided by a custom module, keep this:
try:
    nest.Install("mymodule")
except Exception:
    # If not available, continue; required only if your delayed_synapse is in a module
    pass

# Create single presynaptic spike generator and single postsynaptic parrot (to set spikes manually)
pre_spike_generator = nest.Create("spike_generator", params={"spike_times": pre_spike_times})
post_spike_generator = nest.Create("spike_generator", params={"spike_times": post_spike_times})

# We make the postsynaptic node a parrot neuron so that "postsynaptic spikes" are exactly the times we set.
pre = nest.Create("parrot_neuron")
post = nest.Create("iaf_psc_exp", params={"tau_minus": 10.0})

# Recorders
sd_pre = nest.Create("spike_recorder")
sd_post = nest.Create("spike_recorder")

nest.Connect(pre, sd_pre)
nest.Connect(post, sd_post)

# Connect the manual post spike generator to the parrot so it emits the exact post spikes
nest.Connect(post_spike_generator, post, syn_spec={"weight": 3000})
nest.Connect(pre_spike_generator, pre)

# Setup a simple volume transmitter + modulatory spike generator for dopamine-like events
vt = nest.Create("volume_transmitter")
dopa_spike_times = list(np.arange(20.0, sim_duration, 40.0))  # example modulatory spikes (you can change)
mod_spikes = nest.Create("spike_generator", params={"spike_times": dopa_spike_times})
dopa = nest.Create("parrot_neuron")
nest.Connect(mod_spikes, dopa)
nest.Connect(dopa, vt)

# Delayed synapse defaults (tune to your model); these fields may vary depending on your module
nest.SetDefaults(
    "delayed_synapse",
    {
        "volume_transmitter": vt,
        "Wmin": -10.0,
        "Wmax": 1000.0,
        "tau_c": 50.0,
        "tau_c_delay": 50.0,
        "tau_n": 10.0,
        "tau_plus": 10.0,
        "b": 0.0,
        "A_plus": 0.2,
        "A_minus": 0.2,
    },
)

# Connect pre -> post using the delayed synapse model and an initial weight
initial_weight = 0.0
conn = nest.Connect(pre, post, syn_spec={"synapse_model": "delayed_synapse", "weight": initial_weight, "delay": 0.5})

print("Connection created:", conn)

# ---------- helper: capture C++ stdout (NEST debug prints) ----------
@contextmanager
def capture_cpp_stdout():
    # duplicate fd 1,o create pipe, redirect fd 1 to pipe write end
    old_stdout_fd = os.dup(1)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 1)
    os.close(w_fd)
    try:
        yield r_fd
    finally:
        os.dup2(old_stdout_fd, 1)
        os.close(old_stdout_fd)

# ---------- parse various trace keys from NEST debug output ----------
# We will look for lines containing 'post_node_id' and parse numeric trace key=value pairs appearing in the same line.
trace_keys_to_capture = [
    "c_current", "c_delayed", "e_current", "eligibility", "eligibility_trace",
    "delayed_e", "non_delayed_e", "n", "n_dopamine", "n_trace", "n_dopa"
]

def parse_debug_output(output, traces):
    # iterate line by line looking for post_node_id and t_trig, then extract key=value for our keys
    for line in output.splitlines():
        if "post_node_id" not in line:
            continue
        m_post = re.search(r"post_node_id\s*=\s*(\d+)", line)
        m_t = re.search(r"t_trig\s*=\s*([0-9]+(?:\.[0-9]+)?)", line)
        if not m_post or not m_t:
            # try alternative patterns (space/no-space)
            m_post = re.search(r"post_node_id=(\d+)", line)
            m_t = re.search(r"t_trig=(\d+)", line)
        if not m_post or not m_t:
            continue
        post_id = int(m_post.group(1))
        t_trig = float(m_t.group(1))
        for k in trace_keys_to_capture:
            m_k = re.search(rf"{k}\s*=\s*([\-0-9.eE]+)", line)
            if m_k:
                # print("PARSED", k, "from line:", line.strip())
                val = float(m_k.group(1))
                traces[k][post_id].append((t_trig, val))

# ---------- run simulation and capture outputs ----------
traces = defaultdict(lambda: defaultdict(list))   # traces[key][post_id] -> list of (t_trig, value)
full_output = []
weight_history = []

# Run simulation step-by-step so we can capture debug prints per ms if needed
total_steps = int(np.ceil(sim_duration))
for step in range(total_steps):
    with capture_cpp_stdout() as r_fd:
        nest.Simulate(1.0)
        # read from pipe (may be empty)
        out = os.read(r_fd, 50_000).decode(errors="ignore")
        full_output.append(out)
    # parse and store trace values present in this iteration
    parse_debug_output(out, traces)
    # read current synapse weight
    conns = nest.GetConnections(pre, post)
    if len(conns) > 0:
        w = conns[0].weight
        weight_history.append((step+1, w))
    else:
        weight_history.append((step+1, np.nan))

# Combine full output into a single string for debugging / inspection
all_output = "\n".join(full_output)

print("\n===== Captured debug output (snippet) =====\n")
#print(all_output[:40000])  # print the first chunk for inspection

# ---------- fetch spike data ----------
events_pre = nest.GetStatus(sd_pre, "events")[0]
times_pre = np.array(events_pre["times"]) if events_pre["times"].size else np.array([])

events_post = nest.GetStatus(sd_post, "events")[0]
times_post = np.array(events_post["times"]) if events_post["times"].size else np.array([])

# ---------- display simple ISI info (if relevant) ----------
if len(times_post) >= 2:
    intervals = np.diff(times_post)
    print("\nPost neuron ISIs (ms):", np.round(intervals, 2))
    print("Mean ISI:", np.mean(intervals), "ms; Variance:", np.var(intervals), "ms^2")
else:
    print("\nNot enough post spikes for ISI calculation (0 or 1 spike).")

# ---------- plotting ----------
# Count how many trace subplots exist
trace_keys_to_plot = ["c_delayed", "n"]
n_trace_subplots = sum(1 for k in trace_keys_to_plot if traces[k])

# Total rows = raster + trace subplots + weight
n_rows = 1 + n_trace_subplots + 1

# Height ratios: raster small, traces medium, weight medium
height_ratios = [0.5] + [2]*n_trace_subplots + [2]

fig, axes = plt.subplots(
    n_rows, 1, figsize=(10, 3 * n_rows),
    gridspec_kw={"height_ratios": height_ratios},
    sharex=True
)

# Make axes iterable
if n_rows == 1:
    axes = [axes]

# ---- Raster (top) ----
ax_raster = axes[0]
if times_pre.size:
    ax_raster.scatter(times_pre, np.zeros_like(times_pre)+1, s=50, color="blue", marker='.', label="Pre")
if times_post.size:
    ax_raster.scatter(times_post, np.zeros_like(times_post)+2, s=50, color="red", marker='.', label="Post")
ax_raster.set_yticks([1,2])
ax_raster.set_yticklabels(["Pre","Post"])
ax_raster.set_title("Spike raster")
ax_raster.grid(True, axis="x", alpha=0.6)
ax_raster.legend()

# ---- Traces ----
subplot_idx = 1
for k in trace_keys_to_plot:
    if not traces[k]:
        continue
    ax = axes[subplot_idx]
    for post_id, entries in traces[k].items():
        if not entries:
            continue
        # remove duplicates, sort
        d = {float(t):v for t,v in entries}
        xs = sorted(d.keys())
        ys = [d[x] for x in xs]
        ax.plot(xs, ys, label=f"{k} (post {post_id})")

    # highlight ±0.5 ms around presynaptic spikes
    highlight_half_width = 0.5
    if times_pre.size > 0:
        for t in times_pre:
            ax.axvspan(t - highlight_half_width, t + highlight_half_width, color="gray", alpha=0.15, linewidth=0)

    if times_post.size > 0:
        for t in times_post:
            ax.axvspan(t - highlight_half_width, t + highlight_half_width, color="gray", alpha=0.15, linewidth=0)

    ax.set_ylabel(k)
    ax.set_title(f"{k} trace")
    ax.legend()
    ax.grid(True)
    subplot_idx += 1

# ---- Weight subplot (bottom) ----
ax_weight = axes[-1]
times_w = [t for t,w in weight_history]
weights_w = [w for t,w in weight_history]
ax_weight.plot(times_w, weights_w, label="synapse weight")
ax_weight.set_ylabel("Weight")
ax_weight.set_xlabel("Time (ms)")
ax_weight.set_title("Weight evolution (pre->post)")
ax_weight.grid(True)
ax_weight.legend()

plt.tight_layout()
plt.show()

