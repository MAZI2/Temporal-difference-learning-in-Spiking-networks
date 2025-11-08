import nest
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
SEED = 12345
SIM_DURATION = 100.0  # ms
NEURONS = 1  # postsynaptic neurons

neuron_params = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "V_th": 20.0,
    "V_reset": 0.0,
    "t_ref": 2.0,
    "E_L": 0.0,
    "tau_syn_ex": 5.0,  # ms
}

weight_exp = 50.0  # weight for exponential PSC
weight_alpha = weight_exp  # adjust for total charge for alpha PSC

# Poisson input parameters
spike_times = [20.0]  # ms
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
times_exp = data_exp["times"]
I_exp = data_exp["I_syn_ex"]

data_alpha = nest.GetStatus(dc_alpha)[0]["events"]
times_alpha = data_alpha["times"]
I_alpha = data_alpha["I_syn_ex"]

# -------------------------
# Plot PSCs
# -------------------------
plt.figure(figsize=(10,5))
plt.plot(times_exp, I_exp, label="Exponential PSC", color="red")
plt.plot(times_alpha, I_alpha, label="Alpha PSC", color="blue")
plt.xlabel("Time (ms)")
plt.ylabel("Postsynaptic current (pA)")
plt.title("Postsynaptic Current: Exponential vs Alpha PSC")
plt.legend()
plt.grid(True)
plt.show()

