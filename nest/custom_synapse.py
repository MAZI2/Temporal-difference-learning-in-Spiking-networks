import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
dt = 1.0            # ms, simulation timestep
sim_time = 300      # ms

pre_spike_times = [100.0]       # ms
post_spike_times = [120.0]      # ms
dopamine_spike_times = [150.0]  # ms

# Synapse parameters
A_plus = 0.05          # base potentiation scale
tau_c = 50.0           # ms, eligibility decay
tau_n = 50.0           # ms, dopamine decay
delta_t_opt = 30.0     # ms, optimal dopamine delay after post spike
sigma = 10.0           # ms, width of Gaussian timing window

# ===============================
# CUSTOM GAUSSIAN DA SYNAPSE
# ===============================
class GaussianDASynapse:
    def __init__(self, A_plus, tau_c, tau_n, delta_t_opt, sigma):
        self.A_plus = A_plus
        self.tau_c = tau_c
        self.tau_n = tau_n
        self.delta_t_opt = delta_t_opt
        self.sigma = sigma

        self.c = 0.0          # eligibility
        self.n = 0.0          # dopamine trace
        self.weight = 0.0     # synaptic weight
        self.last_post_spike = None

        # history for plotting
        self.history = {
            "time": [], "weight": [], "c": [], "n": [], "timing_factor": []
        }

    def update(self, t, pre_spikes, post_spikes, dopamine_spikes):
        # Decay traces
        self.c *= np.exp(-dt/self.tau_c)
        self.n *= np.exp(-dt/self.tau_n)

        # Handle pre spike: increment eligibility
        if t in pre_spikes:
            self.c += 1.0

        # Handle post spike: record last post spike time
        if t in post_spikes:
            self.last_post_spike = t

        # Handle dopamine spike: update weight using Gaussian timing
        timing_factor = 0.0
        if t in dopamine_spikes and self.last_post_spike is not None:
            delta_t = t - self.last_post_spike
            timing_factor = np.exp(-((delta_t - self.delta_t_opt)**2) / (2*self.sigma**2))
            self.weight += self.A_plus * self.c * self.n * timing_factor

        # If dopamine spike occurs, increase dopamine trace
        if t in dopamine_spikes:
            self.n += 1.0

        # record history
        self.history["time"].append(t)
        self.history["weight"].append(self.weight)
        self.history["c"].append(self.c)
        self.history["n"].append(self.n)
        self.history["timing_factor"].append(timing_factor)

# ===============================
# SIMULATION LOOP
# ===============================
syn = GaussianDASynapse(A_plus, tau_c, tau_n, delta_t_opt, sigma)
time_points = np.arange(0, sim_time + dt, dt)

for t in time_points:
    syn.update(t, pre_spike_times, post_spike_times, dopamine_spike_times)

# ===============================
# PLOTTING
# ===============================
history = syn.history

plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(history["time"], history["c"], label="Eligibility c")
plt.ylabel("c")
plt.legend()

plt.subplot(4,1,2)
plt.plot(history["time"], history["n"], label="Dopamine n", color='orange')
plt.ylabel("n")
plt.legend()

plt.subplot(4,1,3)
plt.plot(history["time"], history["timing_factor"], label="Timing factor", color='green')
plt.ylabel("Timing factor")
plt.legend()

plt.subplot(4,1,4)
plt.plot(history["time"], history["weight"], label="Synaptic weight w", color='red')
plt.ylabel("Weight")
plt.xlabel("Time (ms)")
plt.legend()

plt.tight_layout()
plt.show()

print("Final synaptic weight:", history["weight"][-1])
