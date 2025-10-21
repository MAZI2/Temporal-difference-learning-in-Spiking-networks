import nest

nest.ResetKernel()

neuron = nest.Create("iaf_psc_alpha")
spike_rec = nest.Create("spike_recorder")
nest.Connect(neuron, spike_rec)

nest.Simulate(100.0)

print(nest.GetStatus(spike_rec)[0]["n_events"])
