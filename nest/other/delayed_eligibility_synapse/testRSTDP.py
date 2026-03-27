# rstdp_debug_using_pong_functions.py
import nest
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# -------------------------
# Parameters (matching your provided scripts)
# -------------------------
stdp_amplitude = 36.0
stdp_tau = 64.0  # ms
stdp_saturation = 128
learning_rate = 0.7
sim_duration = 1000  # ms
SEED = 12356

# constant reward for the whole simulation (user requested const reward)
CONSTANT_REWARD = 100

# neuron params copied from your debugging script

# -------------------------
# Deterministic NEST setup
# -------------------------
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": SEED})
np.random.seed(SEED)
random.seed(SEED)

# -------------------------
# Build network nearly identical to your dopamine debug script
# -------------------------
parrot = nest.Create("parrot_neuron")
n_posts = 3
post_neurons = nest.Create("iaf_psc_exp", n_posts)

# recorders: one pre recorder for parrot and one recorder per post neuron
sd_pre = nest.Create("spike_recorder")
sd_post = nest.Create("spike_recorder", n_posts)

nest.Connect(parrot, sd_pre)
for i, n in enumerate(post_neurons):
    nest.Connect(n, sd_post[i])

# Input spike generator (0.5, 1.5, ..., 200.5)
spike_times = np.arange(0.5, 201.0, 10.0).tolist()
pre = nest.Create("spike_generator", params={"spike_times": spike_times})

# Motor noise as in your script
poisson_motor_ex = nest.Create("poisson_generator", n_posts, params={"rate": 15})
poisson_motor_inh = nest.Create("poisson_generator", n_posts, params={"rate": 0})
nest.Connect(poisson_motor_ex, post_neurons, conn_spec={"rule": "one_to_one"}, syn_spec={"weight": 500})
nest.Connect(poisson_motor_inh, post_neurons, conn_spec={"rule": "one_to_one"}, syn_spec={"weight": 500})

poisson_pre_ex = nest.Create("poisson_generator", 1, params={"rate": 15})
poisson_pre_inh = nest.Create("poisson_generator", 1, params={"rate": 0})
nest.Connect(poisson_pre_ex, parrot, conn_spec={"rule": "one_to_one"}, syn_spec={"weight": 500})
nest.Connect(poisson_pre_inh, parrot, conn_spec={"rule": "one_to_one"}, syn_spec={"weight": 500})

# connect pre -> parrot and parrot -> posts with initial weights 150 + 20*i, delay 1.0
nest.Connect(pre, parrot)
for i, post in enumerate(post_neurons):
    nest.Connect(parrot, post, {"rule": "all_to_all"}, {"weight": 1275 + 8 * i, "delay": 1.0})

print("Connections (parrot -> posts):")
print(nest.GetConnections(parrot, post_neurons))

# -------------------------
# Exact copy of calculate_stdp (from PongNetRSTDP) as a free function
# Note: this is the verbatim algorithmic logic (adapted as function).
# -------------------------
def calculate_stdp(pre_spikes, post_spikes, only_causal=True, next_neighbor=True):
    """
    Calculates the STDP trace for given spike trains.
    Copied from PongNetRSTDP.calculate_stdp (behavior identical).
    Returns a scalar accumulated STDP trace:
      - if only_causal=True -> returns min(facilitation, stdp_saturation)
      - else -> returns min(facilitation - depression, stdp_saturation)
    """
    pre_spikes, post_spikes = np.sort(pre_spikes), np.sort(post_spikes)
    facilitation = 0
    depression = 0
    positions = np.searchsorted(pre_spikes, post_spikes)
    last_position = -1
    for spike, position in zip(post_spikes, positions):
        if position == last_position and next_neighbor:
            continue  # Only next-neighbor pairs
        if position > 0:
            before_spike = pre_spikes[position - 1]
            facilitation += stdp_amplitude * np.exp(-(spike - before_spike) / stdp_tau)
        if position < len(pre_spikes):
            after_spike = pre_spikes[position]
            depression += stdp_amplitude * np.exp(-(after_spike - spike) / stdp_tau)
        last_position = position
    if only_causal:
        return min(facilitation, stdp_saturation)
    else:
        return min(facilitation - depression, stdp_saturation)

# -------------------------
# apply_rstdp: a function that implements the same loop/logic as PongNetRSTDP.apply_rstdp
# We'll adapt it to the debugging script: it operates on the single input neuron 'parrot'
# and reads spike recorders sd_post (list) to obtain post spike times.
# -------------------------
def apply_rstdp(input_train, reward, input_node, post_nodes, post_recorders, lr=learning_rate):
    """
    Apply R-STDP updates to all connections from input_node to post_nodes.
    input_train : list of pre spike times (for this iteration / event)
    reward : scalar reward (constant here)
    input_node : parrot node (NodeCollection / id)
    post_nodes : list of postsynaptic node IDs
    post_recorders : list of spike_recorders (one per post neuron)
    lr : learning rate
    """
    # Build post_events dictionary like in the original method:
    post_events = {}
    offset = post_nodes[0].get("global_id")
    # In the original code they used self.spike_recorders.get("events") which returns
    # event dicts per recorder in order. Here we replicate that:
    for index in range(len(post_recorders)):
        events = nest.GetStatus(post_recorders[index], "events")[0]
        post_events[offset + index] = events["times"]

    # Iterate over all connections from the input node (parrot)
    conns = nest.GetConnections(input_node, post_nodes)
    for conn in conns:
        # get target id and existing weight
        # use conn.get(...) dict style like original code pattern
        vals = conn.get(["target", "weight"])
        motor_neuron = vals["target"]
        old_weight = vals["weight"]

        # motor_spikes read from post_events
        motor_spikes = post_events.get(motor_neuron, [])

        # compute correlation using the exact calculate_stdp function
        correlation = calculate_stdp(input_train, motor_spikes)

        # weight update exactly as in PongNetRSTDP:
        new_weight = old_weight + lr * correlation * reward

        # set the new weight
        conn.set({"weight": new_weight})

        # return trace value for logging (here we store correlation as the c_current-like trace)
        # We'll log externally; the function itself does not return anything (matching original),
        # but we want to collect the correlation per target, so we optionally return tuple.
    # Note: original method didn't return traces; we will separately re-compute correlation to log per-target

# -------------------------
# Data structures for logging traces & weights (per-post global id)
# -------------------------
neuron_trace = defaultdict(list)   # neuron_gid -> list of (t_trig, trace_value)
weight_data = defaultdict(list)    # neuron_gid -> list of weights recorded each ms
time_points = set()

# Get connection list (objects) for repeated access
conns = nest.GetConnections(parrot, post_neurons)
conn_list = []
for c in conns:
    # store the connection object and its target gid so we can query quickly
    tgt = c.get("target")
    conn_list.append((c, tgt["target"] if isinstance(tgt, dict) else tgt))

# Helper to map post node global ids to recorder indices
gid_to_index = {}
for idx, n in enumerate(post_neurons):
    gid_to_index[n.get("global_id")] = idx

# -------------------------
# Simulation loop (1 ms steps)
# For each ms:
#  - nest.Simulate(1.0)
#  - detect pre spikes that happened in this ms
#  - for each pre spike time t_trig:
#      - set input_train = [t_trig]  (mirrors per-event update; PongNet used input_train per iteration)
#      - call apply_rstdp(input_train, CONSTANT_REWARD, ...)
#      - compute and log trace (we compute correlation again per-target for logging)
#  - record weights (after potential updates) for plotting
# -------------------------
for step in range(sim_duration):
    nest.Simulate(1.0)

    # Find pre spikes in this 1 ms bin. pre spike times are fractional (.5, 1.5, ...).
    t_trigs = [t for t in spike_times if int(np.floor(t)) == step]

    for t_trig in t_trigs:
        # The PongNet implementation used self.input_train for the current iteration.
        # Here we mirror that and set input_train to a 1-element list containing this pre spike time.
        input_train = [t_trig]

        # Apply R-STDP update for all parrot->post connections
        apply_rstdp(input_train, CONSTANT_REWARD, parrot, post_neurons, sd_post, lr=learning_rate)

        # For logging: compute correlation (calculate_stdp) per target and save as trace
        for c, tgt in conn_list:
            # get motor spike times recorded so far
            rec_idx = gid_to_index[tgt]
            post_events = nest.GetStatus(sd_post[rec_idx], "events")[0]
            motor_spikes = post_events["times"]
            corr = calculate_stdp(input_train, motor_spikes)  # only_causal=True by default
            neuron_trace[tgt].append((int(np.round(t_trig)), corr))
            time_points.add(int(np.round(t_trig)))

    # store weights for all connections (after updates this ms)
    for c, tgt in conn_list:
        w = c.get("weight")
        
        # Some nest bindings return dict; unify:
        if isinstance(w, dict):
            wval = w.get("weight", 0.0)
        else:
            wval = w
        weight_data[tgt].append(wval)

# -------------------------
# After simulation: collect spike events and plot results
# -------------------------
spikes_pre = nest.GetStatus(sd_pre, "events")[0]
times_pre = spikes_pre["times"]

spikes_post = [nest.GetStatus(sd_post[i], "events")[0] for i in range(len(post_neurons))]
times_post = [sp["times"] for sp in spikes_post]

# -------------------------
# Plot stacked figure with same layout as dopamine debug script
# -------------------------
n_subplots = 2 + len(post_neurons) + 2  # pre + post rasters + c_current rows + weights + firing rates
fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 3 * n_subplots), sharex=True)

# --- Pre raster ---
axes[0].eventplot([times_pre], lineoffsets=[1], linelengths=0.8)
axes[0].set_yticks([1])
axes[0].set_yticklabels(["Pre"])
axes[0].set_title("Presynaptic Spikes")
axes[0].grid(True, axis="x", linestyle="--", alpha=0.6)

# --- Post rasters ---
axes[1].eventplot(
    [times_post[i] for i in range(len(post_neurons))],
    lineoffsets=np.arange(1, len(post_neurons) + 1),
    linelengths=0.8,
)
axes[1].set_yticks(np.arange(1, len(post_neurons) + 1))
axes[1].set_yticklabels([f"Post {i}" for i in range(len(post_neurons))])
axes[1].set_title("Postsynaptic Spikes")
axes[1].grid(True, axis="x", linestyle="--", alpha=0.6)

# --- c_current per neuron (we use our neuron_trace)
trace_start_ax = 2
# Sort neuron_trace items descending by gid to mimic the ordering in your original plotting
sorted_neurons = sorted(neuron_trace.items(), reverse=True)
for i, (post_id, entries) in enumerate(sorted_neurons):
    ax = axes[trace_start_ax + i]
    if not entries:
        continue
    x_vals = [e[0] for e in entries]
    y_vals = [e[1] for e in entries]
    ax.plot(x_vals, y_vals, label=f"Neuron {post_id}")
    ax.set_ylabel("c_current")
    ax.set_title(f"Neuron {post_id} c_current")
    ax.legend()
    ax.grid(True)

# --- Weight evolution ---
ax_w = axes[-2]
for tgt, weights in sorted(weight_data.items()):
    weights = np.array(weights)
    norm_w = weights - weights[0]
    times = np.arange(len(weights)) - 200
    times = np.clip(times, 0, None)
    ax_w.plot(times, norm_w, label=f"target={tgt}")
ax_w.set_ylabel("ΔWeight (normalized)")
ax_w.set_title("Weight evolution per post neuron (shifted 200 ms earlier)")
ax_w.legend()
ax_w.grid(True)

# --- Firing rate per 50 ms bin ---
ax_fr = axes[-1]
bin_size = 50.0
n_bins = int(np.ceil(sim_duration / bin_size))
bin_edges = np.arange(0, (n_bins + 1) * bin_size, bin_size)

for i, sp in enumerate(spikes_post):
    times = np.array(sp["times"])
    n_spikes = len(times)
    duration_sec = sim_duration / 1000.0
    total_freq = n_spikes / duration_sec

    # Per-bin frequencies
    hist, _ = np.histogram(times, bins=bin_edges)
    bin_freqs = hist / (bin_size / 1000.0)
    ax_fr.plot(bin_edges[:-1] + bin_size / 2, bin_freqs, label=f"Post {i}")

    print(f"\nPost neuron {i}: {n_spikes} spikes total, avg = {total_freq:.2f} Hz")
    print(f"  Firing rates per 50 ms bin (Hz): {np.round(bin_freqs, 2)}")

ax_fr.set_xlabel("Time (ms)")
ax_fr.set_ylabel("Firing rate (Hz / 50 ms)")
ax_fr.set_title("Postsynaptic firing rates per 50 ms bin")
ax_fr.legend()
ax_fr.grid(True)

plt.tight_layout()
plt.show()

