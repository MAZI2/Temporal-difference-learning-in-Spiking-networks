import nest
import matplotlib.pyplot as plt
import io
import os
import re
from contextlib import contextmanager
import sys

# -------------------------
# NEST setup
# -------------------------
neuron_params = {
    "C_m": 250.0,      # membrane capacitance in pF
    "tau_m": 10.0,     # membrane time constant in ms
    "V_reset": 0.0,  # reset potential mV
    "V_th": 20.0,     # spike threshold mV
    "t_ref": 0.5,      # absolute refractory period ms
    "V_m": 0.0,      # initial membrane potential mV
    "E_L": 0.0,      # resting potential mV
}

nest.Install("mymodule")
neuron = nest.Create("parrot_neuron")
neuron2 = nest.Create("iaf_psc_alpha", params=neuron_params)
sd1 = nest.Create("spike_recorder")
sd2 = nest.Create("spike_recorder")

nest.Connect(neuron, sd1)
nest.Connect(neuron2, sd2)
spike_times = [float(t) for t in range(1, 201, 1)]
#pre = nest.Create("spike_generator", {"spike_times": [10.0, 50.0, 90.0]})
pre = nest.Create("spike_generator", {"spike_times": spike_times})
post = nest.Create("spike_generator", {"spike_times": [12.0, 52.0, 92.0]})

pg = nest.Create("poisson_generator", params={"rate": 200.0})  # pre-syn
vt = nest.Create("volume_transmitter")

dopa_spike_times = [float(t) for t in range(1, 401, 20)]
mod_spikes = nest.Create("spike_generator", {"spike_times": dopa_spike_times})
dopa = nest.Create("parrot_neuron")
nest.Connect(mod_spikes, dopa)
nest.Connect(dopa, vt)

# nest.SetDefaults(
#             "stdp_dopamine_synapse",
#             {
#                 "volume_transmitter": vt,
#                 "tau_c": 50,
#                 "tau_n": 50,
#                 "tau_plus": 100,
#                 "Wmin": 1220,
#                 "Wmax": 1550,
#                 "b": 0.0,
#                 "A_plus": 0.6,
#             },
#         )
nest.SetDefaults(
            "delayed_synapse",
            {
                "volume_transmitter": vt,
                "tau_c": 50,
                "tau_c_delay": 200,
                "tau_n": 10,
                "tau_plus": 50,
                "Wmin": 150,
                "Wmax": 500,
                "b": 0.0,
                "A_plus": 0.75,
            },
        )

# syn_conn = nest.Connect(neuron, neuron2, {"rule": "all_to_all"},
#                         {"synapse_model": "stdp_dopamine_synapse", "weight": 150.0})
syn_conn = nest.Connect(neuron, neuron2, {"rule": "all_to_all"},
                        {"synapse_model": "delayed_synapse"})
print(nest.GetConnections(neuron, neuron2))

nest.Connect(pre, neuron)



# -------------------------
# Helpers for capturing C++ stdout
# -------------------------
@contextmanager
def capture_cpp_stdout():
    old_stdout_fd = os.dup(1)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 1)  # redirect C++ stdout to pipe
    os.close(w_fd)
    try:
        yield r_fd
    finally:
        os.dup2(old_stdout_fd, 1)
        os.close(old_stdout_fd)

# -------------------------
# Data storage
# -------------------------
debug_pattern = re.compile(
    r"\[DEBUG trigger_update_weight\]\s+"
    r".*?t_trig=(\S+).*?"
    r"c_delayed=(\S+).*?"
    r"n=(\S+)"
)


c_delayed_data = []
n_data = []
weight_data = []
time_points = []

for step in range(600):  # simulate 200 ms in 1 ms chunks
    with capture_cpp_stdout() as r_fd:
        nest.Simulate(1.0)
        os.close(1)
        output = os.read(r_fd, 10_000).decode(errors="ignore")

    for m in debug_pattern.finditer(output):
        t_trig = float(m.group(1))
        n_val = float(m.group(2))
        c_delayed = float(m.group(3))
        c_delayed_data.append(c_delayed)
        n_data.append(n_val)
        time_points.append(t_trig)

    # capture weight evolution (get last weight each ms)
    w = nest.GetConnections(neuron, neuron2)[0].weight
    weight_data.append(w)

# ---------------------------------------
# Spike data
# ---------------------------------------
spikes1 = nest.GetStatus(sd1, "events")[0]
spikes2 = nest.GetStatus(sd2, "events")[0]
times1 = spikes1["times"]
times2 = spikes2["times"]

# ---------------------------------------
# Plot results
# ---------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# Raster for pre and post spikes
axs[0].eventplot([times1, times2], colors=['blue', 'red'], lineoffsets=[1, 2], linelengths=0.8)
axs[0].set_yticks([1, 2])
axs[0].set_yticklabels(["Pre", "Post"])
axs[0].set_title("Pre/Post Spikes")

# c_delayed trace
axs[1].plot(time_points, c_delayed_data, color='green')
axs[1].set_ylabel("n")
axs[1].set_title("n")

# n trace
axs[2].plot(time_points, n_data, color='purple')
axs[2].set_ylabel("c")
axs[2].set_title("c")

# Weight evolution
axs[3].plot(range(len(weight_data)), weight_data, color='black')
axs[3].set_xlabel("Time (ms)")
axs[3].set_ylabel("Weight")
axs[3].set_title("Weight evolution")

plt.tight_layout()
plt.show()
