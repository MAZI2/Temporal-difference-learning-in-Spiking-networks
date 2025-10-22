import nest
import matplotlib.pyplot as plt

nest.Install("mymodule")
neuron = nest.Create("iaf_psc_alpha")
neuron2 = nest.Create("iaf_psc_alpha")
sd1 = nest.Create("spike_recorder")
sd2 = nest.Create("spike_recorder")

# Connect neurons to spike detectors
nest.Connect(neuron, sd1)
nest.Connect(neuron2, sd2)

pre = nest.Create("spike_generator", {"spike_times": [10.0, 50.0, 90.0]})
post = nest.Create("spike_generator", {"spike_times": [12.0, 52.0, 92.0]})

pg = nest.Create("poisson_generator", params={"rate": 200.0})    # pre-syn
vt = nest.Create("volume_transmitter")

nest.SetDefaults("delayed_synapse",
                 {"volume_transmitter": vt,
                  "tau_c": 250.0,
                  "tau_c_delay": 50,
                  "tau_n": 200.0,
                  "tau_plus": 45.0,
                  "Wmin": 0.0,         # lower Wmin to see changes
                  "Wmax": 1550.0,
                  "b": 0,
                  "A_plus": 0.81,
                  "A_minus": 0.5})
nest.SetDefaults("test_synapse",
                 {"volume_transmitter": vt,
                  "tau_c": 250.0,
                  "tau_n": 200.0,
                  "tau_plus": 45.0,
                  "Wmin": 0.0,         # lower Wmin to see changes
                  "Wmax": 1550.0,
                  "b": 0,
                  "A_plus": 0.81,
                  "A_minus": 0.5})


nest.Connect(neuron, neuron2, {"rule": "all_to_all"}, {"synapse_model": "delayed_synapse", "weight": 800.0, "delay": 1.0})
nest.Connect(pg, neuron, {"rule": "all_to_all"},
             {"synapse_model": "static_synapse", "weight": 800.0, "delay": 1.0})

nest.Simulate(100.0)

spikes1 = nest.GetStatus(sd1, "events")[0]
spikes2 = nest.GetStatus(sd2, "events")[0]

times1 = spikes1["times"]
senders1 = spikes1["senders"]
times2 = spikes2["times"]
senders2 = spikes2["senders"]

print("pre", times1)
print("post", times2)

# Plot spikes as raster
"""
plt.figure(figsize=(10, 4))
plt.eventplot([times1, times2], colors=['blue', 'red'], lineoffsets=[1, 2], linelengths=0.8)
plt.yticks([1, 2], ["Neuron 1", "Neuron 2"])
plt.xlabel("Time (ms)")
plt.ylabel("Neuron")
plt.title("Spike raster plot")
plt.show()
"""
