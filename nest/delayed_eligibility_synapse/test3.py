import nest
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Simulation parameters
# -------------------------
SEED = 12345
N_LAYERS = 3
NEURONS_PER_LAYER = 10
SIM_DURATION = 500.0  # ms
PSC_TYPE = "iaf_psc_exp"

# Neuron parameters
neuron_params = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "V_th": 20.0,
    "V_reset": 0.0,
    "t_ref": 2.0,
    "E_L": 0.0,
    "tau_syn_ex": 5.0,
}

# Poisson input
N_POISSON_PER_NEURON = 50
RATE_INPUT = 100.0
WEIGHT_INPUT = 25.0
DELAY = 1.0

# Feedforward random weights
WEIGHT_MIN = 30.0
WEIGHT_MAX = 100.0

# -------------------------
# Reset kernel
# -------------------------
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": SEED, "print_time": False})
np.random.seed(SEED)

# -------------------------
# Create input layer (Poisson generators)
# -------------------------
poisson_input = nest.Create("poisson_generator", N_POISSON_PER_NEURON, params={"rate": RATE_INPUT})

# -------------------------
# Create feedforward layers and spike recorders
# -------------------------
layers = []
recorders = []

for l in range(N_LAYERS):
    layer = nest.Create(PSC_TYPE, NEURONS_PER_LAYER, params=neuron_params)
    layers.append(layer)
    sr = nest.Create("spike_recorder", NEURONS_PER_LAYER)
    recorders.append(sr)
    for i in range(NEURONS_PER_LAYER):
        nest.Connect(layer[i], sr[i])

# -------------------------
# Connect input → first layer
# -------------------------
nest.Connect(poisson_input, layers[0], "all_to_all", {"weight": WEIGHT_INPUT, "delay": DELAY})

# -------------------------
# Connect feedforward layers with random weights
# -------------------------
for l in range(1, N_LAYERS):
    for pre in layers[l-1]:
        for post in layers[l]:
            w = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX)
            nest.Connect(pre, post, "all_to_all", {"weight": w, "delay": DELAY})

# -------------------------
# Run simulation
# -------------------------
nest.Simulate(SIM_DURATION)

# -------------------------
# Collect spike times
# -------------------------
layer_spike_times = []
for recs in recorders:
    times = []
    for sr in recs:
        events = nest.GetStatus(sr, "events")[0]
        times.append(events["times"])
    layer_spike_times.append(times)

# -------------------------
# Compute ISI statistics
# -------------------------
def compute_isi_stats(spike_times_layer):
    means = []
    vars_ = []
    for neuron_times in spike_times_layer:
        if len(neuron_times) < 2:
            means.append(np.nan)
            vars_.append(np.nan)
        else:
            isi = np.diff(neuron_times)
            means.append(np.mean(isi))
            vars_.append(np.var(isi))
    return means, vars_

# Print statistics
print("\n=== Layer-wise ISI Statistics ===")
for l, times in enumerate(layer_spike_times):
    mean_isi, var_isi = compute_isi_stats(times)
    print(f"\nLayer {l+1}:")
    for n in range(NEURONS_PER_LAYER):
        print(f"  Neuron {n}: mean ISI = {mean_isi[n]:.2f} ms, variance = {var_isi[n]:.2f} ms^2")
    print(f"  Layer mean ISI = {np.nanmean(mean_isi):.2f} ms, Layer ISI variance = {np.nanmean(var_isi):.2f} ms^2")

# -------------------------
# Plot rasters
# -------------------------
fig, axs = plt.subplots(N_LAYERS, 1, figsize=(12, 3*N_LAYERS), sharex=True)

for l in range(N_LAYERS):
    for i, t in enumerate(layer_spike_times[l]):
        axs[l].scatter(t, np.ones_like(t)*(i+1), s=10)
    axs[l].set_ylabel(f"Layer {l+1}")
    axs[l].set_title(f"Raster plot: Layer {l+1}")
    axs[l].grid(True, alpha=0.5)

axs[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.show()

