import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar

# ==========================================
# 1. PARAMETERS & CROSSOVER CALCULATION
# ==========================================
np.random.seed(42)

k = 0.02          # Bleaching rate constant per unit illumination per second
I1 = 1.0          # Constant illumination power in Experiment 1 (left y-axis units)
eta = 200.0       # Emission scaling factor [emission units / (illumination unit * active fraction)]
E_target = 160.0  # Target emission intensity (right y-axis units)

# Time when 50% of fluorophores are active (active fraction = 0.5):
# Exp 1 (constant I1): N1(t)/N0 = exp(-k * I1 * t) = 0.5 => t_50_1 = ln(2) / (k * I1)
t_50_1 = np.log(2.0) / (k * I1)

# Exp 2 (constant emission E_target): N2(t)/N0 = 1 - (k / eta) * E_target * t = 0.5
t_50_2 = 0.5 * eta / (k * E_target)

# Cumulative dose crossover time when D1(t) = D2(t):
# D1(t) = I1 * t
# D2(t) = -(eta / k) * ln(1 - (k / eta) * E_target * t)
# Let u = (k / eta) * E_target * t; crossover occurs when 1 - u = exp(-c_eff * u), c_eff = (eta * I1) / E_target
c_eff = (eta * I1) / E_target
t_cross = None
if c_eff > 1.0:
    sol = root_scalar(lambda u: 1.0 - u - np.exp(-c_eff * u), bracket=[1e-7, 0.99999])
    if sol.converged:
        t_cross = sol.root * eta / (k * E_target)

# ==========================================
# 2. PLOT 1: SIMULATED DATA UNTIL 50% ACTIVE
# ==========================================
def active_fraction_model(t, n0_frac, k_eff):
    return n0_frac * np.exp(-k_eff * t)

t_sim = np.linspace(0, t_50_1, 60)
true_frac = active_fraction_model(t_sim, 1.0, k * I1)
noisy_frac = np.clip(true_frac + np.random.normal(0, 0.015, size=t_sim.shape), 0.0, 1.05)

popt, _ = curve_fit(active_fraction_model, t_sim, noisy_frac, p0=[1.0, k * I1])
fit_frac = active_fraction_model(t_sim, *popt)

# ==========================================
# 3. PLOT 2 & 3: EXPERIMENT TRAJECTORIES UNTIL 50% ACTIVE
# ==========================================
# Experiment 1: Without correction (constant illumination I1)
t1 = np.linspace(0, t_50_1, 200)
I_ill_1 = np.full_like(t1, I1)
N1_frac = np.exp(-k * I1 * t1)
E_em_1 = eta * I_ill_1 * N1_frac

# Experiment 2: With correction (dynamic illumination to hold emission at E_target)
t2 = np.linspace(0, t_50_2, 200)
N2_frac = 1.0 - (k / eta) * E_target * t2
I_ill_2 = (E_target / eta) / N2_frac
E_em_2 = np.full_like(t2, E_target)

# ==========================================
# 4. THREE-PANEL SIDE-BY-SIDE PLOTS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Plot 1: Simulated Data Until 50% Fluorophores Active ---
ax1 = axes[0]
ax1.scatter(t_sim, noisy_frac * 100, color='royalblue', s=22, alpha=0.75, label='Simulated Data')
ax1.plot(t_sim, fit_frac * 100, color='navy', linewidth=2.2, label='Fitted Decay')
ax1.axhline(50, color='gray', linestyle=':', linewidth=1.2, label='50% Active Threshold')
ax1.set_xlim(0, t_50_1)
ax1.set_ylim(0, 105)
ax1.set_title('1. Simulated Active Fluorophores\n(Until 50% Active)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Active Fluorophores (%)', fontsize=10)
ax1.legend(loc='lower left', fontsize=8.5, frameon=True)
ax1.grid(True, linestyle='--', alpha=0.4)

def plot_experiment_panel(ax, t_vec, I_vec, E_vec, title_str, t_cross_val):
    # Left y-axis: Illumination power (blue line)
    line_ill, = ax.plot(t_vec, I_vec, color='blue', linewidth=2.2, label='Illumination Intensity')
    ax.set_xlim(0, t_vec[-1])
    ax.set_ylim(0, max(I_ill_2.max(), I1) * 1.15)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Illumination Power (a.u.)', color='blue', fontsize=10)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Shade area under illumination curve ONLY until crossover time point
    fill_handle = None
    if t_cross_val is not None and t_cross_val > 0:
        t_fill_end = min(t_cross_val, t_vec[-1])
        mask_fill = t_vec <= t_fill_end
        t_fill = t_vec[mask_fill]
        I_fill = I_vec[mask_fill]
        if t_fill[-1] < t_fill_end:
            t_fill = np.append(t_fill, t_fill_end)
            I_fill = np.append(I_fill, np.interp(t_fill_end, t_vec, I_vec))
        fill_handle = ax.fill_between(
            t_fill, 0, I_fill, color='blue', alpha=0.18,
            label=f'Cumulative Dose (until t={t_fill_end:.1f}s)'
        )
        ax.axvline(t_fill_end, color='darkslateblue', linestyle='-.', linewidth=1.2)

    # Right y-axis: Emission intensity (green line + dotted target line)
    ax_right = ax.twinx()
    line_em, = ax_right.plot(t_vec, E_vec, color='green', linewidth=2.2, label='Emission Intensity')
    line_target = ax_right.axhline(
        E_target, color='darkgreen', linestyle=':', linewidth=1.8, label='Target Intensity'
    )
    ax_right.set_ylim(0, max(eta * I1, E_target) * 1.15)
    ax_right.set_ylabel('Emission Intensity (a.u.)', color='green', fontsize=10)
    ax_right.tick_params(axis='y', labelcolor='green')

    ax.set_title(title_str, fontsize=11, fontweight='bold')

    handles = [line_ill, line_em, line_target]
    if fill_handle is not None:
        handles.append(fill_handle)
    ax.legend(handles, [h.get_label() for h in handles], loc='upper left', fontsize=8, frameon=True)

# --- Plot 2: Experiment 1 (Without Correction) ---
plot_experiment_panel(
    axes[1], t1, I_ill_1, E_em_1,
    '2. Without Correction (Exp. 1)\n(Until 50% Active)', t_cross
)

# --- Plot 3: Experiment 2 (With Correction) ---
plot_experiment_panel(
    axes[2], t2, I_ill_2, E_em_2,
    '3. With Correction (Exp. 2)\n(Until 50% Active)', t_cross
)

plt.tight_layout()
plt.show()