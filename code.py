"""
Kinetic modeling toolkit for the n-hexane hydrocracking project.

The script supports the four project questions:
1. Exploratory analysis of operating conditions (activity/selectivity/stability).
2. Derivation-oriented kinetic rate implementation and isothermal regression.
3. Non-isothermal regression with Arrhenius & Van't Hoff laws in original and
   reparameterised forms.
4. High-level interpretation scaffolding via consolidated fit statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R_GAS = 8.31446261815324  # J mol-1 K-1
P_REF_BAR = 1.0
MU_TO_SI = 1e-6
G_TO_KG = 1e-3


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw Excel headers to Python-friendly identifiers."""

    def _clean(col: str) -> str:
        base = col.split("[", maxsplit=1)[0].strip()
        mapping = {
            "F0C6": "F0_C6",
            "F0H2": "F0_H2",
            "Pressure": "pressure_bar",
            "Temperature": "temperature_c",
            "Wcat": "Wcat_g",
            "F2MeC5": "F_2MeC5",
            "F3MeC5": "F_3MeC5",
            "FC3": "F_C3",
        }
        return mapping.get(base, base)

    return df.rename(columns={col: _clean(col) for col in df.columns})


def prepare_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the experimental dataset and add derived quantities needed downstream.
    Returned DataFrame uses SI units for rates and explicit mole fractions.
    """

    df_raw = pd.read_excel(data_path)
    df = _rename_columns(df_raw).copy()

    df["temperature_K"] = df["temperature_c"] + 273.15
    df["Wcat_kg"] = df["Wcat_g"] * G_TO_KG

    # Outlet flow of n-hexane from the carbon balance (µmol s-1).
    df["F_C6_out"] = (
        df["F0_C6"]
        - df["F_2MeC5"]
        - df["F_3MeC5"]
        - 0.5 * df["F_C3"]
    )

    # Flow totals (µmol s-1) and conversions to mol s-1.
    df["F_total"] = (
        df["F_C6_out"] + df["F_2MeC5"] + df["F_3MeC5"] + df["F_C3"] + df["F0_H2"]
    )

    flow_cols = [
        "F0_C6",
        "F0_H2",
        "F_C6_out",
        "F_2MeC5",
        "F_3MeC5",
        "F_C3",
        "F_total",
    ]
    for col in flow_cols:
        df[f"{col}_mol_s"] = df[col] * MU_TO_SI

    # Mole fractions and partial pressures (bar) at reactor outlet (CSTR assumption).
    for species in ("C6_out", "2MeC5", "3MeC5", "C3"):
        df[f"y_{species}"] = df[f"F_{species}"] / df["F_total"]
        df[f"P_{species}"] = df[f"y_{species}"] * df["pressure_bar"]

    df["y_H2"] = df["F0_H2"] / df["F_total"]
    df["P_H2"] = df["y_H2"] * df["pressure_bar"]

    # Observed rates (mol kg-1 s-1).
    df["r_2MeC5_obs"] = df["F_2MeC5_mol_s"] / df["Wcat_kg"]
    df["r_3MeC5_obs"] = df["F_3MeC5_mol_s"] / df["Wcat_kg"]
    df["r_C3_obs"] = (df["F_C3_mol_s"] / 2.0) / df["Wcat_kg"]

    df["conversion"] = (
        df["F_2MeC5"] + df["F_3MeC5"] + 0.5 * df["F_C3"]
    ) / df["F0_C6"]
    produced = df["F_2MeC5"] + df["F_3MeC5"] + 0.5 * df["F_C3"]
    df["selectivity_iso"] = ((df["F_2MeC5"] + df["F_3MeC5"]) / produced).fillna(0.0)
    df["selectivity_crack"] = ((0.5 * df["F_C3"]) / produced).fillna(0.0)

    return df


def summarize_operating_conditions(df: pd.DataFrame) -> None:
    """Console summary for Question 1."""
    print("\n=== Question 1: Operating condition screening ===")
    profile = (
        df.groupby("temperature_c")[["conversion", "selectivity_iso", "selectivity_crack"]]
        .agg(["mean", "std"])
        .sort_index()
    )
    print("\nTemperature effect (mean ± std):")
    print(profile.round(4))

    for var in ("pressure_bar", "F0_H2", "temperature_c"):
        corr = df[var].corr(df["conversion"])
        print(f"Correlation(conv, {var}) = {corr:.3f}")

    print("\nStability proxy (conversion vs experiment index):")
    stability_coef = np.polyfit(np.arange(len(df)), df["conversion"], deg=1)[0]
    print(f"  Trend slope ≈ {stability_coef:.3e} conversion units / experiment.")


def _theta_c6(
    P_c6: np.ndarray,
    P_iso_sum: np.ndarray,
    K_ads: float,
) -> np.ndarray:
    """Langmuir-based surface coverage of physisorbed n-hexane."""
    denom = 1.0 + K_ads * (P_c6 + P_iso_sum)
    return (K_ads * P_c6) / np.clip(denom, 1e-12, None)


def _predict_rates(
    params: np.ndarray,
    P_c6: np.ndarray,
    P_2Me: np.ndarray,
    P_3Me: np.ndarray,
    P_h2: np.ndarray,
) -> np.ndarray:
    """Return predicted rates (mol kg-1 s-1) for the three RDS."""
    k1, k2, k3, K_ads = params
    theta = _theta_c6(P_c6, P_2Me + P_3Me, K_ads)
    base = np.clip(theta, 1e-18, None)
    return np.column_stack(
        (
            k1 * base,
            k2 * base,
            k3 * base,
        )
    )


def _residual_log_ratio(
    params_log: np.ndarray,
    obs: np.ndarray,
    pressures: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Residuals in log space for robust isothermal regression."""
    params = np.exp(params_log)
    pred = np.clip(
        _predict_rates(params, *pressures),
        1e-18,
        None,
    )
    residual = np.log(pred) - np.log(np.clip(obs, 1e-18, None))
    return residual.ravel(order="C")


@dataclass
class IsothermalResult:
    temperature_c: float
    parameters: Dict[str, float]
    covariance: pd.DataFrame | None
    rmse_log: float
    residuals: pd.DataFrame


def fit_isothermal(df: pd.DataFrame) -> Dict[float, IsothermalResult]:
    """Estimate kinetic parameters independently per temperature."""
    results: Dict[float, IsothermalResult] = {}
    for temperature, block in df.groupby("temperature_c"):
        obs = block[["r_2MeC5_obs", "r_3MeC5_obs", "r_C3_obs"]].to_numpy()
        pressures = (
            block["P_C6_out"].to_numpy(),
            block["P_2MeC5"].to_numpy(),
            block["P_3MeC5"].to_numpy(),
            block["P_H2"].to_numpy(),
        )

        initial = np.log(np.array([1e-3, 6e-4, 1e-4, 0.1]))
        lower = np.log(np.array([1e-8, 1e-8, 1e-8, 1e-4]))
        upper = np.log(np.array([1e-1, 1e-1, 1e-1, 1e3]))
        fit = least_squares(
            _residual_log_ratio,
            initial,
            args=(obs, pressures),
            method="trf",
            bounds=(lower, upper),
        )

        params = dict(
            zip(
                ("k_iso_2Me", "k_iso_3Me", "k_crack", "K_ads"),
                np.exp(fit.x),
            )
        )

        dof = obs.size - fit.x.size
        covariance = None
        if fit.jac is not None and dof > 0:
            try:
                jac = fit.jac
                sigma2 = (fit.fun @ fit.fun) / dof
                cov = np.linalg.inv(jac.T @ jac) * sigma2
                covariance = pd.DataFrame(
                    cov,
                    index=list(params.keys()),
                    columns=list(params.keys()),
                )
            except np.linalg.LinAlgError:
                covariance = None

        pred = _predict_rates(np.exp(fit.x), *pressures)
        residual_table = pd.DataFrame(
            {
                "r_2MeC5_obs": obs[:, 0],
                "r_2MeC5_pred": pred[:, 0],
                "r_3MeC5_obs": obs[:, 1],
                "r_3MeC5_pred": pred[:, 1],
                "r_C3_obs": obs[:, 2],
                "r_C3_pred": pred[:, 2],
            }
        )

        rmse_log = np.sqrt(np.mean(fit.fun ** 2))
        results[temperature] = IsothermalResult(
            temperature_c=temperature,
            parameters=params,
            covariance=covariance,
            rmse_log=rmse_log,
            residuals=residual_table,
        )

    return results


def _prepare_initial_global(
    iso_results: Dict[float, IsothermalResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(iso_results.items())
    temps_K = np.array(
        [temp + 273.15 for temp, _ in ordered],
        dtype=float,
    )
    k_values = np.array(
        [
            [
                res.parameters["k_iso_2Me"],
                res.parameters["k_iso_3Me"],
                res.parameters["k_crack"],
            ]
            for _, res in ordered
        ],
        dtype=float,
    )
    Kads = np.array(
        [res.parameters["K_ads"] for _, res in ordered],
        dtype=float,
    )
    return temps_K, k_values, Kads


def _linear_arrhenius_fit(
    temperatures_K: np.ndarray,
    k_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ln(k) = intercept + slope / T -> convert to A, Ea."""
    inv_T = 1.0 / temperatures_K
    coeffs = np.polyfit(inv_T, np.log(k_values), deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    Ea = -slope * R_GAS
    A = np.exp(intercept)
    return A, Ea


def _compute_rate_constants(
    params: np.ndarray,
    temperatures: np.ndarray,
    mode: Literal["original", "reparam"],
    T_ref: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "original":
        log_A = params[:3]
        log_K0 = params[3]
        log_Ea = params[4:7]
        delta_H = params[7]
        A = np.exp(np.clip(log_A, -100.0, 100.0))
        Ea = np.exp(np.clip(log_Ea, -50.0, 50.0))
        inv_RT = 1.0 / (R_GAS * temperatures)
        expo = np.exp(np.clip(-np.outer(inv_RT, Ea), -700.0, 700.0))
        k = A[None, :] * expo
        K0 = np.exp(np.clip(log_K0, -100.0, 100.0))
        exponent = np.clip(-delta_H * inv_RT, -700.0, 700.0)
        Kads = K0 * np.exp(exponent)
        return k, Kads

    if mode == "reparam":
        log_k_ref = params[:3]
        log_K_ref = params[3]
        log_Ea = params[4:7]
        delta_H = params[7]

        k_ref = np.exp(np.clip(log_k_ref, -100.0, 100.0))
        K_ref = np.exp(np.clip(log_K_ref, -100.0, 100.0))
        Ea = np.exp(np.clip(log_Ea, -50.0, 50.0))
        inv_T = 1.0 / temperatures
        inv_T_ref = 1.0 / T_ref
        expo = np.exp(
            np.clip(
                -((inv_T - inv_T_ref)[:, None]) * (Ea[None, :] / R_GAS),
                -700.0,
                700.0,
            )
        )
        k = k_ref[None, :] * expo
        exponent = np.clip(-delta_H / R_GAS * (inv_T - inv_T_ref), -700.0, 700.0)
        Kads = K_ref * np.exp(exponent)
        return k, Kads

    raise ValueError(f"Unknown mode {mode}")


def _global_residuals(
    params: np.ndarray,
    df: pd.DataFrame,
    mode: Literal["original", "reparam"],
    T_ref: float,
) -> np.ndarray:
    temps = df["temperature_K"].to_numpy()
    k_matrix, Kads_vec = _compute_rate_constants(params, temps, mode, T_ref)

    theta = _theta_c6(
        df["P_C6_out"].to_numpy(),
        df["P_2MeC5"].to_numpy() + df["P_3MeC5"].to_numpy(),
        Kads_vec,
    )
    theta = np.clip(theta, 1e-18, None)

    pred = np.column_stack(
        (
            k_matrix[:, 0] * theta,
            k_matrix[:, 1] * theta,
            k_matrix[:, 2] * theta,
        )
    )
    obs = df[["r_2MeC5_obs", "r_3MeC5_obs", "r_C3_obs"]].to_numpy()
    residual = np.log(np.clip(pred, 1e-18, None)) - np.log(np.clip(obs, 1e-18, None))
    return residual.ravel(order="C")


@dataclass
class GlobalFitResult:
    parameters: Dict[str, float]
    covariance: pd.DataFrame | None
    rmse_log: float
    mode: Literal["original", "reparam"]


def fit_global(
    df: pd.DataFrame,
    iso_results: Dict[float, IsothermalResult],
    mode: Literal["original", "reparam"],
) -> GlobalFitResult:
    """Non-isothermal regression."""
    temps, k_estimates, Kads_values = _prepare_initial_global(iso_results)
    T_ref = np.median(df["temperature_K"])

    log_A_init = []
    log_Ea_init = []
    for i in range(3):
        A0, Ea0 = _linear_arrhenius_fit(temps, k_estimates[:, i])
        log_A_init.append(np.log(max(A0, 1e-12)))
        log_Ea_init.append(np.log(max(Ea0, 1e3)))

    A_K, Ea_K = _linear_arrhenius_fit(temps, Kads_values)

    if mode == "original":
        initial = np.concatenate(
            (
                log_A_init,
                [np.log(max(A_K, 1e-12))],
                log_Ea_init,
                [Ea_K],
            )
        )
        lower = np.array(
            [
                -30.0,
                -30.0,
                -30.0,
                -10.0,
                np.log(1e4),
                np.log(1e4),
                np.log(1e4),
                -3e5,
            ]
        )
        upper = np.array(
            [
                15.0,
                15.0,
                15.0,
                10.0,
                np.log(5e5),
                np.log(5e5),
                np.log(5e5),
                3e5,
            ]
        )
    else:
        idx_ref = np.argmin(np.abs(temps - T_ref))
        k_ref_guess = np.log(np.clip(k_estimates[idx_ref, :], 1e-12, None))
        initial = np.concatenate(
            (
                k_ref_guess,
                [np.log(max(A_K * np.exp(-Ea_K / (R_GAS * T_ref)), 1e-12))],
                log_Ea_init,
                [Ea_K],
            )
        )
        lower = np.array(
            [
                -20.0,
                -20.0,
                -20.0,
                -10.0,
                np.log(1e4),
                np.log(1e4),
                np.log(1e4),
                -3e5,
            ]
        )
        upper = np.array(
            [
                5.0,
                5.0,
                5.0,
                10.0,
                np.log(5e5),
                np.log(5e5),
                np.log(5e5),
                3e5,
            ]
        )

    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)

    fit = least_squares(
        _global_residuals,
        initial,
        args=(df, mode, T_ref),
        method="trf",
        bounds=(lower, upper),
    )

    names = (
        ["log_A_k1", "log_A_k2", "log_A_k3", "log_K0", "log_Ea_k1", "log_Ea_k2", "log_Ea_k3", "delta_H_ads"]
        if mode == "original"
        else ["log_kref_k1", "log_kref_k2", "log_kref_k3", "log_Kref", "log_Ea_k1", "log_Ea_k2", "log_Ea_k3", "delta_H_ads"]
    )

    params = dict(zip(names, fit.x))

    dof = df.shape[0] * 3 - fit.x.size
    covariance = None
    if fit.jac is not None and dof > 0:
        try:
            sigma2 = (fit.fun @ fit.fun) / dof
            cov = np.linalg.inv(fit.jac.T @ fit.jac) * sigma2
            covariance = pd.DataFrame(cov, index=names, columns=names)
        except np.linalg.LinAlgError:
            covariance = None

    rmse = np.sqrt(np.mean(fit.fun ** 2))
    return GlobalFitResult(parameters=params, covariance=covariance, rmse_log=rmse, mode=mode)


def display_isothermal(results: Dict[float, IsothermalResult]) -> None:
    print("\n=== Question 2: Isothermal parameter estimates ===")
    rows = []
    for temp, res in sorted(results.items()):
        params = res.parameters
        rows.append(
            {
                "T (°C)": temp,
                "k_2Me (mol kg-1 s-1)": params["k_iso_2Me"],
                "k_3Me (mol kg-1 s-1)": params["k_iso_3Me"],
                "k_crack (mol kg-1 s-1)": params["k_crack"],
                "K_ads (bar-1)": params["K_ads"],
                "RMSE_log": res.rmse_log,
            }
        )
    frame = pd.DataFrame(rows).sort_values("T (°C)")
    pd.options.display.float_format = "{:0.3e}".format
    print(frame.to_string(index=False))
    pd.reset_option("display.float_format")


def display_global(global_fit: GlobalFitResult) -> None:
    print(f"\n=== Question 3: Non-isothermal fit ({global_fit.mode}) ===")
    for key, value in global_fit.parameters.items():
        print(f"{key:>18s} = {value: 0.6e}")
    print(f"RMSE (log-space): {global_fit.rmse_log:0.5f}")


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "KMS_n_Hexane_Hydrocracking_experimental data.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = prepare_dataset(data_path)
    summarize_operating_conditions(df)

    iso_results = fit_isothermal(df)
    display_isothermal(iso_results)

    global_original = fit_global(df, iso_results, mode="original")
    display_global(global_original)

    global_reparam = fit_global(df, iso_results, mode="reparam")
    display_global(global_reparam)


if __name__ == "__main__":
    main()
