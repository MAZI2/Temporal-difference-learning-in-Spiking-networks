import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
dt = 1.0            # ms
sim_time = 1000      # ms

pre_spike_times = [100.0]
post_spike_times = [120.0]
dopamine_spike_times = [200.0]

# Synapse parameters
A_plus = 0.05
tau_c = 50.0          # eligibility decay (ms)
tau_n = 300.0          # dopamine decay (ms)
tau_k = 30.0
k = 0.1          # kernel time constant (ms)
inflect = 150

# ===============================
# CUSTOM EXPONENTIAL KERNEL SYNAPSE
# ===============================
class ExpKernelDASynapse:
    def __init__(self, A_plus, tau_c, tau_n, tau_k, k, inflect):
        self.A_plus = A_plus
        self.tau_c = tau_c
        self.tau_n = tau_n
        self.tau_k = tau_k
        self.k = k
        self.inflect = inflect

        self.c = 0.0          # eligibility
        self.n = 0.0          # dopamine trace
        self.weight = 0.0     # synaptic weight
        self.last_post_spike = None
        self.last_dopa_spike = None

        # history for plotting
        self.history = {
            "time": [], "weight": [], "c": [], "n": [], "timing_factor": []
        }

    def exponential_kernel(self, delta_t):
        """Exponential rise kernel: 0 at 0, →1 as delta_t→∞"""
        if delta_t <= 0:
            return 0.0
        return 1 - np.exp(-delta_t / self.tau_k)

    def logistic_kernel(self, delta_t):
        f_raw = 1 / (1 + np.exp(-self.k * (delta_t - self.inflect)))

        # shift so f(0)=0 and normalize to 1
        f = (f_raw - (1 / (1 + np.exp(self.k * self.inflect)))) / (1 - 1 / (1 + np.exp(self.k * self.inflect)))

        return f

    def update(self, t, pre_spikes, post_spikes, dopamine_spikes):
        # Decay traces
        self.c *= np.exp(-dt / self.tau_c)
        self.n *= np.exp(-dt / self.tau_n)

        # Pre spike increases eligibility
        if any(abs(t - s) < 0.5 for s in pre_spikes):
            self.c += 1.0

        # Record post spike time
        if any(abs(t - s) < 0.5 for s in post_spikes):
            self.last_post_spike = t

        # Dopamine spike → increase dopamine trace
        if any(abs(t - s) < 0.5 for s in dopamine_spikes):
            self.n += 1.0
            self.last_dopa_spike = t

        # Weight update: if dopamine has occurred, compute timing_factor relative to post spike
        timing_factor = 0.0
        if self.last_post_spike is not None: #and self.last_dopa_spike is not None:
            delta_t = t - self.last_post_spike#self.last_dopa_spike - self.last_post_spike
            timing_factor = self.logistic_kernel(delta_t)
            print(t, delta_t)
            self.weight += self.A_plus * self.c * self.n * timing_factor

        # record history
        self.history["time"].append(t)
        self.history["weight"].append(self.weight)
        self.history["c"].append(self.c)
        self.history["n"].append(self.n)
        self.history["timing_factor"].append(timing_factor)

# ===============================
# SIMULATION LOOP
# ===============================
syn = ExpKernelDASynapse(A_plus, tau_c, tau_n, tau_k, k, inflect)
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
plt.plot(history["time"], history["timing_factor"], label="Exponential kernel(t)", color='green')
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
