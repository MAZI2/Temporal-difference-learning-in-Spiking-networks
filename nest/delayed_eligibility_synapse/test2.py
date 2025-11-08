import nest
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
SEED = 12347
N_POST = 5
SIM_DURATION = 5000.0  # ms

neuron_params = {
    "C_m": 250.0,
    "tau_m": 20.0,
    "V_th": 20.0,
    "V_reset": 0.0,
    "t_ref": 2.0,
    "E_L": 0.0,
}

tau_syn_ex = 5.0  # ms
weight_exp = 25.0  # weight for exponential PSC

# -------------------------
# Reset NEST kernel
# -------------------------
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": SEED, "print_time": False})
np.random.seed(SEED)

# -------------------------
# Create neurons
# -------------------------
post_exp = nest.Create("iaf_psc_exp", N_POST, params={**neuron_params, "tau_syn_ex": tau_syn_ex})
post_alpha = nest.Create("iaf_psc_alpha", N_POST, params={**neuron_params, "tau_syn_ex": tau_syn_ex})

# -------------------------
# Weight adjustment for alpha PSC
# -------------------------
# Total charge: Q_exp = weight_exp * tau_syn
# Alpha injects more charge: Q_alpha = weight_alpha * tau_syn * e
# => weight_alpha = weight_exp / e
weight_alpha = weight_exp / np.e

# -------------------------
# Spike recorders
# -------------------------
sr_exp = nest.Create("spike_recorder", N_POST)
sr_alpha = nest.Create("spike_recorder", N_POST)

for i in range(N_POST):
    nest.Connect(post_exp[i], sr_exp[i])
    nest.Connect(post_alpha[i], sr_alpha[i])

# -------------------------
# Poisson background
# -------------------------
rate_bg = 8000.0  # spikes/sec
poisson_bg_exp = nest.Create("poisson_generator", N_POST, params={"rate": rate_bg})
poisson_bg_alpha = nest.Create("poisson_generator", N_POST, params={"rate": rate_bg})

nest.Connect(poisson_bg_exp, post_exp, "one_to_one", {"weight": weight_exp, "delay": 1.0})
nest.Connect(poisson_bg_alpha, post_alpha, "one_to_one", {"weight": weight_alpha, "delay": 1.0})

# -------------------------
# Simulation
# -------------------------
nest.Simulate(SIM_DURATION)

# -------------------------
# Collect spike times and compute ISIs
# -------------------------
def get_isi_stats(spike_recorders):
    isi_means = []
    isi_vars = []
    all_times = []
    for sr in spike_recorders:
        events = nest.GetStatus(sr, "events")[0]
        times = events["times"]
        all_times.append(times)
        if len(times) < 2:
            isi_means.append(np.nan)
            isi_vars.append(np.nan)
        else:
            isi = np.diff(times)
            isi_means.append(np.mean(isi))
            isi_vars.append(np.var(isi))
    return all_times, isi_means, isi_vars

times_exp, mean_exp, var_exp = get_isi_stats(sr_exp)
times_alpha, mean_alpha, var_alpha = get_isi_stats(sr_alpha)

print("=== Exponential PSC ===")
for i, (m, v) in enumerate(zip(mean_exp, var_exp)):
    print(f"Neuron {i}: mean ISI = {m:.2f} ms, variance = {v:.2f} ms^2")

print("\n=== Alpha PSC ===")
for i, (m, v) in enumerate(zip(mean_alpha, var_alpha)):
    print(f"Neuron {i}: mean ISI = {m:.2f} ms, variance = {v:.2f} ms^2")

# -------------------------
# Plot rasters
# -------------------------

# Exponential PSC raster
"""
for i, t in enumerate(times_exp):
    axs[0].scatter(t, np.ones_like(t)*(i+1), color='red', s=10)
axs[0].set_ylabel("Neuron")
axs[0].set_title("Exponential PSC")

# Alpha PSC raster
for i, t in enumerate(times_alpha):
    axs[1].scatter(t, np.ones_like(t)*(i+1), color='blue', s=10)
axs[1].set_ylabel("Neuron")
axs[1].set_title("Alpha PSC")
axs[1].set_xlabel("Time (ms)")

plt.tight_layout()
plt.show()

"""
plt.figure(figsize=(8,4))
plt.hist(np.concatenate([np.diff(t) for t in times_exp if len(t)>1]),
         bins=60, alpha=0.5, label="Exp PSC")
plt.hist(np.concatenate([np.diff(t) for t in times_alpha if len(t)>1]),
         bins=60, alpha=0.5, label="Alpha PSC")
plt.xlabel("ISI (ms)")
plt.ylabel("Count")
plt.title("ISI Distribution")
plt.legend()
plt.show()

CV_exp = []
CV_alpha = []

for t in times_exp:
    if len(t)>2:
        isi = np.diff(t)
        CV_exp.append(np.std(isi) / np.mean(isi))
    else:
        CV_exp.append(np.nan)

for t in times_alpha:
    if len(t)>2:
        isi = np.diff(t)
        CV_alpha.append(np.std(isi) / np.mean(isi))
    else:
        CV_alpha.append(np.nan)

from scipy.stats import norm, gaussian_kde

# --------------------------------------------------
# Collect ISIs
# --------------------------------------------------
isis_exp = np.concatenate([np.diff(t) for t in times_exp if len(t) > 1])
isis_alpha = np.concatenate([np.diff(t) for t in times_alpha if len(t) > 1])

# --------------------------------------------------
# Fit Gaussian to each distribution
# --------------------------------------------------
mu_exp, sigma_exp = norm.fit(isis_exp)
mu_alpha, sigma_alpha = norm.fit(isis_alpha)

# Gaussian PDFs for plotting
xmin = min(isis_exp.min(), isis_alpha.min())
xmax = max(isis_exp.max(), isis_alpha.max())
x = np.linspace(xmin, xmax, 500)

pdf_exp = norm.pdf(x, mu_exp, sigma_exp)
pdf_alpha = norm.pdf(x, mu_alpha, sigma_alpha)

# --------------------------------------------------
# KDE (non-parametric smoothing)
# --------------------------------------------------
kde_exp = gaussian_kde(isis_exp)
kde_alpha = gaussian_kde(isis_alpha)

kde_exp_vals = kde_exp(x)
kde_alpha_vals = kde_alpha(x)

# --------------------------------------------------
# Plot histogram + fits
# --------------------------------------------------
plt.figure(figsize=(10, 5))

# Histograms
plt.hist(isis_exp, bins=60, alpha=0.4, label="Exp PSC (ISI histogram)")
plt.hist(isis_alpha, bins=60, alpha=0.4, label="Alpha PSC (ISI histogram)")

# Gaussian fits
plt.plot(x, pdf_exp * len(isis_exp) * (xmax-xmin)/60,
         color="red", linewidth=2, label="Exp PSC Gaussian fit")
plt.plot(x, pdf_alpha * len(isis_alpha) * (xmax-xmin)/60,
         color="blue", linewidth=2, label="Alpha PSC Gaussian fit")

# KDE curves
plt.plot(x, kde_exp_vals * len(isis_exp) * (xmax-xmin)/60,
         color="red", linestyle="--", linewidth=2, label="Exp PSC KDE")
plt.plot(x, kde_alpha_vals * len(isis_alpha) * (xmax-xmin)/60,
         color="blue", linestyle="--", linewidth=2, label="Alpha PSC KDE")

plt.xlabel("ISI (ms)")
plt.ylabel("Count")
plt.title("ISI Distribution with Gaussian Fit and KDE")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Print Gaussian fit parameters
# --------------------------------------------------
print("\n=== Gaussian Fit Parameters ===")
print(f"Exp PSC:   mean={mu_exp:.4f} ms, sigma={sigma_exp:.4f} ms")
print(f"Alpha PSC: mean={mu_alpha:.4f} ms, sigma={sigma_alpha:.4f} ms")

