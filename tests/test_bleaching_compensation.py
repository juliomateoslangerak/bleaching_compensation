import unittest
import numpy as np
from scipy.optimize import curve_fit


def simulate_exp1_emission(t, I1, E1_0, k_eff1, bg=0.0, noise_std=0.0, rng=None):
    """Simulate Experiment 1 emission under constant illumination I1."""
    t_arr = np.asarray(t, dtype=float)
    signal = E1_0 * np.exp(-k_eff1 * t_arr) + bg
    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        signal = signal + rng.normal(0.0, noise_std, size=t_arr.shape)
    return signal


def fit_exp1_kinetics(t, emission, bg=0.0, p0=None):
    """Fit background-subtracted emission E(t) - bg = E1_0 * exp(-k_eff1 * t)."""
    t_arr = np.asarray(t, dtype=float)
    y = np.asarray(emission, dtype=float) - bg
    if p0 is None:
        p0 = [max(float(y[0]), 1.0), 0.02]
    popt, _ = curve_fit(
        lambda x, e0, k_eff: e0 * np.exp(-k_eff * x),
        t_arr,
        y,
        p0=p0,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
    )
    return float(popt[0]), float(popt[1])


def build_compensation_model(
    I1: float,
    E1_0: float,
    k_eff1: float,
    mode: str = "equal_dose",
    T_exp: float = 20.0,
    f_min: float = 0.5,
    I_max: float | None = None,
):
    """Derives E_target, I2(0), and dynamic correction curve I2(t) from Exp 1 fit."""
    if I1 <= 0 or E1_0 <= 0 or k_eff1 <= 0 or T_exp <= 0:
        raise ValueError("I1, E1_0, k_eff1, and T_exp must be positive.")

    k = k_eff1 / I1

    if mode == "equal_dose":
        E_target = E1_0 * (1.0 - np.exp(-k_eff1 * T_exp)) / (k_eff1 * T_exp)
    elif mode == "final_emission":
        E_target = E1_0 * np.exp(-k_eff1 * T_exp)
    elif mode == "min_active_fraction":
        if not (0.0 < f_min < 1.0):
            raise ValueError("f_min must be strictly between 0 and 1.")
        E_target = E1_0 * (1.0 - f_min) / (k_eff1 * T_exp)
    elif mode == "max_power_cap":
        if I_max is None or I_max <= 0:
            raise ValueError("A positive I_max must be provided for 'max_power_cap'.")
        r_max = I_max / (I1 + I_max * k_eff1 * T_exp)
        E_target = r_max * E1_0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    I2_0 = E_target * (I1 / E1_0)
    t_singularity = 1.0 / (k * I2_0)

    def I2_func(t):
        t_arr = np.asarray(t, dtype=float)
        if np.any(t_arr >= t_singularity):
            raise ValueError(
                f"Requested time reaches/exceeds singularity at t = {t_singularity:.2f} s."
            )
        return I2_0 / (1.0 - k * I2_0 * t_arr)

    return {
        "E_target": float(E_target),
        "I2_0": float(I2_0),
        "k": float(k),
        "t_singularity": float(t_singularity),
        "I2_func": I2_func,
    }


def simulate_exp2_emission(t, I2_func, k, I2_0, E1_0, I1):
    """Simulate Experiment 2 emission under dynamic illumination I2_func(t)."""
    t_arr = np.asarray(t, dtype=float)
    I2_t = I2_func(t_arr)
    N_frac = 1.0 - k * I2_0 * t_arr
    alpha = E1_0 / I1
    return alpha * I2_t * N_frac


class TestBleachingCompensation(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.I1 = 2.0
        self.E1_0_true = 400.0
        self.k_eff1_true = 0.03
        self.bg = 25.0
        self.t = np.linspace(0.0, 40.0, 81)

    def test_fit_recovers_simulated_parameters(self):
        noisy_emission = simulate_exp1_emission(
            self.t,
            I1=self.I1,
            E1_0=self.E1_0_true,
            k_eff1=self.k_eff1_true,
            bg=self.bg,
            noise_std=3.0,
            rng=self.rng,
        )
        e0_fit, k_eff_fit = fit_exp1_kinetics(self.t, noisy_emission, bg=self.bg)
        self.assertAlmostEqual(e0_fit, self.E1_0_true, delta=5.0)
        self.assertAlmostEqual(k_eff_fit, self.k_eff1_true, delta=0.002)

    def test_equal_dose_matches_cumulative_dose_at_T_exp(self):
        T_exp = 20.0
        model = build_compensation_model(
            I1=self.I1,
            E1_0=self.E1_0_true,
            k_eff1=self.k_eff1_true,
            mode="equal_dose",
            T_exp=T_exp,
        )
        t_grid = np.linspace(0.0, T_exp, 5000)
        dose_exp1 = self.I1 * T_exp
        dose_exp2 = np.trapezoid(model["I2_func"](t_grid), t_grid)
        self.assertAlmostEqual(dose_exp2, dose_exp1, places=2)

    def test_compensated_simulation_maintains_constant_emission(self):
        T_exp = 15.0
        model = build_compensation_model(
            I1=self.I1,
            E1_0=self.E1_0_true,
            k_eff1=self.k_eff1_true,
            mode="final_emission",
            T_exp=T_exp,
        )
        t_eval = np.linspace(0.0, T_exp, 50)
        E2_sim = simulate_exp2_emission(
            t_eval,
            I2_func=model["I2_func"],
            k=model["k"],
            I2_0=model["I2_0"],
            E1_0=self.E1_0_true,
            I1=self.I1,
        )
        np.testing.assert_allclose(
            E2_sim, np.full_like(t_eval, model["E_target"]), rtol=1e-10
        )

    def test_singularity_raises_value_error(self):
        model = build_compensation_model(
            I1=self.I1,
            E1_0=self.E1_0_true,
            k_eff1=self.k_eff1_true,
            mode="equal_dose",
            T_exp=10.0,
        )
        with self.assertRaises(ValueError):
            model["I2_func"](model["t_singularity"] + 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)