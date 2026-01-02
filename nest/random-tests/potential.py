import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1  # ms
T = 80    # total simulation time in ms
time = np.arange(0, T, dt)

# LIF neuron parameters
E_L = -65.0      # mV, resting potential
V_th = -50.0     # mV, threshold
V_reset = -65.0  # mV, reset potential
V_min = -80.0    # mV, absolute lower bound
tau_m = 10.0     # ms, membrane time constant
R_m = 10.0       # MΩ
t_ref = 5.0      # ms, refractory period
I_e = 0.0        # external current (constant)

# Synapse parameters
w_ex = 6.0       # excitatory weight (nA)
w_in = -6.0      # inhibitory weight (nA)
tau_syn = 5.0    # ms, PSC decay

# Presynaptic spike times
pre_spike_ex = [5]   # excitatory spike
pre_spike_in = [35]  # inhibitory spike

# Function to generate PSC
def generate_psc(time, spike_times, w, tau_syn, dt):
    psc = np.zeros(len(time))
    for t_spike in spike_times:
        idx = int(t_spike / dt)
        for i in range(idx, len(time)):
            psc[i] += w * np.exp(-(time[i] - t_spike)/tau_syn)
    return psc

# Total PSC including inhibitory spike
psc_total = generate_psc(time, pre_spike_ex, w_ex, tau_syn, dt) + \
            generate_psc(time, pre_spike_in, w_in, tau_syn, dt)

# LIF simulation function
def lif_simulation(time, tau_m, V_reset, V_th, E_L, R_m, I_syn, t_ref):
    V_m = np.zeros(len(time))
    V_m[0] = E_L
    last_spike_time = -np.inf
    spike_times = []
    for i in range(1, len(time)):
        # Refractory
        if time[i] - last_spike_time < t_ref:
            V_m[i] = V_reset
            continue
        # Membrane update
        dV = dt * ((E_L - V_m[i-1]) + R_m * I_syn[i] + I_e*R_m) / tau_m
        V_m[i] = V_m[i-1] + dV
        # Threshold
        if V_m[i] >= V_th:
            V_m[i] = V_th
            last_spike_time = time[i]
            spike_times.append(i)
        # Lower bound
        if V_m[i] < V_min:
            V_m[i] = V_min
    return V_m, spike_times

# Simulate neuron for two tau_m values
V_m_tau, spike_times = lif_simulation(time, tau_m, V_reset, V_th, E_L, R_m, psc_total, t_ref)
V_m_half_tau, _ = lif_simulation(time, tau_m/2, V_reset, V_th, E_L, R_m, psc_total, t_ref)

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Membrane potential subplot
axs[0].plot(time, V_m_half_tau, label=f'V_m (τ_m={tau_m/2} ms)', linewidth=1, color='#c7c7c7')
axs[0].plot(time, V_m_tau, label=f'V_m (τ_m={tau_m} ms)', color="black")
axs[0].axhline(V_th, color='gray', linestyle='--', linewidth=1)
axs[0].text(2, V_th, r"$V_{\text{th}}$", va='bottom')
axs[0].axhline(V_reset, color='gray', linestyle='--', linewidth=1)
axs[0].text(-3, V_reset, r"$E_L$", va='bottom')
axs[0].axhline(V_min, linestyle='--', linewidth=1, color="black")
axs[0].text(-3, V_min, r"$V_{\min}$", va='bottom')

# Refractory period shaded region
for spike_idx in spike_times:
    ref_start = time[spike_idx]
    axs[0].axvspan(ref_start, ref_start + t_ref, color='gray', alpha=0.1)
    axs[0].text(ref_start+(t_ref/2)-1, -85, r"$t_{\text{ref}}$", va='bottom')

axs[0].set_ylabel('Membranski potencial (mV)')
axs[0].legend()
axs[0].set_title('Membranski potencial LIF nevrona v odvisnosti od presinaptičnega toka.')
axs[0].set_ylim(-82, -45)

# PSC subplot (only total PSC)
axs[1].plot(time, psc_total, color='black')
axs[1].set_xlabel('Čas (ms)')
axs[1].set_ylabel(r"$I_{\text{syn}}$ (nA)")
axs[1].set_ylim(-7, 7)

plt.tight_layout()
plt.show()

