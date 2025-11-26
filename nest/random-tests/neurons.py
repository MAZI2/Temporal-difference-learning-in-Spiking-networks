import nest
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
SEED = 12345
SIM_DURATION = 50.0  # ms
NEURONS = 1  # postsynaptic neurons
tau_syn_ex = 5.0
neuron_params = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "V_th": 20.0,
    "V_reset": 0.0,
    "t_ref": 2.0,
    "E_L": 0.0,
    "tau_syn_ex": tau_syn_ex,  # ms
}

weight_exp = 50.0  # weight for exponential PSC
weight_alpha = weight_exp  # adjust for total charge for alpha PSC

# Poisson input parameters
spike_times = [0.1]  # ms
delay = 1.0  # ms

# -------------------------
# Reset kernel
# -------------------------
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": SEED, "print_time": False})
np.random.seed(SEED)

# -------------------------
# Create neurons
# -------------------------
post_exp = nest.Create("iaf_psc_exp", NEURONS, params=neuron_params)
post_alpha = nest.Create("iaf_psc_alpha", NEURONS, params=neuron_params)

# -------------------------
# Create spike generators
# -------------------------
sg_exp = nest.Create("spike_generator", params={"spike_times": spike_times})
sg_alpha = nest.Create("spike_generator", params={"spike_times": spike_times})

# -------------------------
# Recorders for PSC
# -------------------------
dc_exp = nest.Create("multimeter", params={"record_from": ["I_syn_ex"], "interval": 0.1})
dc_alpha = nest.Create("multimeter", params={"record_from": ["I_syn_ex"], "interval": 0.1})

nest.Connect(dc_exp, post_exp)
nest.Connect(dc_alpha, post_alpha)

# -------------------------
# Connect spike generators to neurons
# -------------------------
nest.Connect(sg_exp, post_exp, syn_spec={"weight": weight_exp, "delay": delay})
nest.Connect(sg_alpha, post_alpha, syn_spec={"weight": weight_alpha, "delay": delay})

# -------------------------
# Run simulation
# -------------------------
nest.Simulate(SIM_DURATION)

# -------------------------
# Extract recorded PSC
# -------------------------
data_exp = nest.GetStatus(dc_exp)[0]["events"]
times_exp = data_exp["times"]-1
I_exp = data_exp["I_syn_ex"]

data_alpha = nest.GetStatus(dc_alpha)[0]["events"]
times_alpha = data_alpha["times"]-1
I_alpha = data_alpha["I_syn_ex"]

# -------------------------
# Plot PSCs
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(times_exp, I_exp, label="Eksponentno jedro")
plt.plot(times_alpha, I_alpha, label="Alfa jedro")

plt.xlabel("Čas (ms)")
plt.ylabel("Postsinaptični tok (pA)")
plt.title("Postsinaptični tok: eksponentno vs alfa jedro")
plt.legend()
plt.grid(True)

# mark x-axis every 5 ms
# plt.xticks(np.arange(0, SIM_DURATION + 1, 5))
# plt.minorticks_on()
# plt.grid(which='minor', linestyle=':', linewidth=0.5)

plt.axvline(tau_syn_ex+spike_times[0], linestyle='--', linewidth=1, color="gray")
plt.text(tau_syn_ex+spike_times[0]+0.4, 0, r"$\tau_{\text{syn, ex}}$",
         rotation=0, va='bottom', ha='left')

# --- 3) Horizontal Y-marker at 36.7% of max exponential PSC ---
y_mark = max(I_exp) * 0.368
plt.axhline(y_mark, linestyle='--', linewidth=1, color="gray")
plt.text(20, y_mark, f" 36.8%", va='bottom')

plt.show()


