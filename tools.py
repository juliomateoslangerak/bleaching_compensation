import numpy as np


def build_compensation_model(
    I1: float,
    E1_0: float,
    k_eff1: float,
    mode: str = "equal_dose",
    T_exp: float = 20.0,
    f_min: float = 0.5,
    I_max: float | None = None,
):
    """
    Derives E_target, I2(0), and the dynamic correction curve I2(t)
    from Experiment 1 constant-illumination fit parameters.

    Parameters
    ----------
    I1 : float
        Constant illumination power used in Experiment 1.
    E1_0 : float
        Fitted initial emission intensity (background-subtracted) at t = 0 in Exp 1.
    k_eff1 : float
        Fitted exponential decay rate (in s^-1) in Experiment 1.
    mode : str
        Strategy for choosing E_target:
        - "equal_dose": matches cumulative dose of Exp 1 over [0, T_exp]
        - "final_emission": matches Exp 1 emission at T_exp
        - "min_active_fraction": guarantees active fraction >= f_min at T_exp
        - "max_power_cap": sets E_target so I2(T_exp) reaches I_max
    T_exp : float
        Planned experiment duration (in seconds).
    f_min : float
        Minimum remaining active fluorophore fraction at T_exp (0 < f_min < 1).
    I_max : float or None
        Maximum allowed illumination power at T_exp (required for "max_power_cap").
    """
    if I1 <= 0 or E1_0 <= 0 or k_eff1 <= 0 or T_exp <= 0:
        raise ValueError("I1, E1_0, k_eff1, and T_exp must be positive.")

    # Per-unit-power bleaching rate: k_eff1 = k * I1
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
        raise ValueError(
            "mode must be one of 'equal_dose', 'final_emission', "
            "'min_active_fraction', or 'max_power_cap'."
        )

    # Initial illumination power for Experiment 2: E_target = (E1_0 / I1) * I2(0)
    I2_0 = E_target * (I1 / E1_0)

    # Theoretical time where denominator (1 - k * I2_0 * t) hits zero
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


# --- Example usage ---
model = build_compensation_model(
    I1=2.0,
    E1_0=400.0,
    k_eff1=0.04,
    mode="equal_dose",
    T_exp=20.0,
)

times = np.linspace(0, 20.0, 5)
print(f"Target Emission (E_target):    {model['E_target']:.2f}")
print(f"Initial Power (I2(0)):         {model['I2_0']:.3f}")
print(f"Per-unit-power bleaching rate: {model['k']:.3f}")
print(f"Singularity Time:              {model['t_singularity']:.2f} s")
print("I2(t) at sample times:        ", np.round(model["I2_func"](times), 3))