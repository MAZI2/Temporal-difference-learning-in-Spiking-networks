import nest
import matplotlib.pyplot as plt
import io
import os
import re
from contextlib import contextmanager

# -------------------------
# NEST setup
# -------------------------
nest.Install("mymodule")
neuron = nest.Create("iaf_psc_alpha")
neuron2 = nest.Create("iaf_psc_alpha")
sd1 = nest.Create("spike_recorder")
sd2 = nest.Create("spike_recorder")

nest.Connect(neuron, sd1)
nest.Connect(neuron2, sd2)

pre = nest.Create("spike_generator", {"spike_times": [10.0, 50.0, 90.0]})
post = nest.Create("spike_generator", {"spike_times": [12.0, 52.0, 92.0]})

pg = nest.Create("poisson_generator", params={"rate": 200.0})  # pre-syn
vt = nest.Create("volume_transmitter")
mod_spikes = nest.Create("spike_generator", {"spike_times": [25.0]})
dopa = nest.Create("parrot_neuron")
nest.Connect(mod_spikes, dopa)
nest.Connect(dopa, vt)

nest.SetDefaults("delayed_synapse",
                 {"volume_transmitter": vt,
                  "tau_c": 250.0,
                  "tau_c_delay": 50,
                  "tau_n": 200.0,
                  "tau_plus": 45.0,
                  "Wmin": 0.0,
                  "Wmax": 1550.0,
                  "b": 0,
                  "A_plus": 0.81,
                  "A_minus": 0.5})

syn_conn = nest.Connect(neuron, neuron2, {"rule": "all_to_all"},
                        {"synapse_model": "delayed_synapse", "weight": 800.0, "delay": 1.0})
nest.Connect(pg, neuron, {"rule": "all_to_all"},
             {"synapse_model": "static_synapse", "weight": 800.0, "delay": 1.0})

nest.Simulate(200.0)
# ---------------------------------------
# Spike data
# ---------------------------------------
spikes1 = nest.GetStatus(sd1, "events")[0]
spikes2 = nest.GetStatus(sd2, "events")[0]
times1 = spikes1["times"]
times2 = spikes2["times"]
