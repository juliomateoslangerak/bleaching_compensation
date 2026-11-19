import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. SIMULATE FLUORESCENCE BLEACHING DATA
# ==========================================
np.random.seed(42) # For reproducibility

# Time vector (e.g., 50 frames over 100 seconds)
time = np.linspace(0, 100, 100)

# True parameters for simulation
I0_true = 1000.0   # Initial intensity
k_true = 0.025     # Bleaching rate constant (s^-1)
bg_true = 50.0     # Background intensity

# Single-exponential decay model function
def exponential_decay(t, I0, k, bg):
    return I0 * np.exp(-k * t) + bg

# Generate clean signal and add realistic Gaussian noise
clean_signal = exponential_decay(time, I0_true, k_true, bg_true)
noise = np.random.normal(0, 30, size=time.shape)
measured_signal = clean_signal + noise

# ==========================================
# 2. CALCULATE / FIT BLEACHING PARAMETERS
# ==========================================
# Initial guesses for curve fitting [I0, k, bg]
p0 = [900.0, 0.02, 40.0]

# Perform curve fitting using SciPy
popt, pcov = curve_fit(exponential_decay, time, measured_signal, p0=p0)
I0_fit, k_fit, bg_fit = popt

print("--- Fit Results ---")
print(f"Fitted Initial Intensity (I0): {I0_fit:.2f}")
print(f"Fitted Bleaching Rate (k):     {k_fit:.4f} s^-1")
print(f"Fitted Background (bg):        {bg_fit:.2f}")

# Generate fitted curve values for plotting
fitted_curve = exponential_decay(time, I0_fit, k_fit, bg_fit)

# ==========================================
# 3. SIMULATE ILLUMINATION COMPENSATION
# ==========================================
# Target constant emission intensity we want to maintain
E_target = 300.0
# Corresponding constant illumination for uncorrected scenario
I_const = 0.5
# Dynamic illumination required to keep emission constant: I_ill(t) = I0 / (1 - k * I0 * t)
# We safely slice time up to near the singularity limit
safe_idx = np.where(k_fit * I_const * time < 0.85)[0]
t_safe = time[safe_idx]
I_dynamic = I_const / (1.0 - k_fit * I_const * t_safe)

# ==========================================
# 4. PLOT RESULTS SIDE-BY-SIDE
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Bleaching Data & Fit ---
ax1.scatter(time, measured_signal, color='dodgerblue', alpha=0.7, label='Measured Data (Noisy)')
ax1.plot(time, fitted_curve, color='crimson', linewidth=2.5,
         label=f'Exponential Fit\n$k = {k_fit:.4f}\\,\\text{{s}}^{{-1}}$')
ax1.set_title('Photobleaching Characterization & Fit', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=10)
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Plot 2: Illumination Strategies Comparison ---
ax2.plot(time, np.full_like(time, I_const), color='dodgerblue', linestyle='--', linewidth=2,
         label='Uncorrected (Constant Illumination)')
ax2.plot(t_safe, I_dynamic, color='darkorange', linewidth=2.5,
         label='Compensated (Dynamic Illumination)')
ax2.set_title('Illumination Intensity Over Time', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Excitation Intensity ($I_{\\text{ill}}$)', fontsize=10)
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()