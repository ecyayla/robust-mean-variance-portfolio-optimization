import numpy as np
import pandas as pd
import time
import glob
import json
import os
from itertools import product
from datetime import datetime
from main import mainRMVP1BnB, RMVP1_mipGUROBI, mainRMVP2BnB, RMVP2_mipGUROBI, zeropadding

def _base_problem_data(file_path, sheet_name, r_c, gamma, beta,
                       target_return_factor, scale_returns):
    """Common problem data shared by RMVP1 and RMVP2, BEFORE any assumption
    filtering: loads returns, builds D, tau, tau_bar and resolves gamma.
    If gamma == 0, a dynamic gamma just below the feasibility boundary is used:
    gamma = min(|tau|/sqrt(D_ii)) - eps."""
    # 1. Load data (no header assumed) and scale.
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    returns = df.values * scale_returns
    n_samples, n_assets = returns.shape

    # 2. Statistics.
    r_hat = returns.mean(axis=0)
    returns_centered = returns - r_hat
    D = (returns_centered.T @ returns_centered) / (n_samples - 1)

    # 3. Target / excess parameters.
    r_c_scaled = r_c * scale_returns
    bar_r = r_c_scaled * target_return_factor
    if bar_r <= r_c_scaled:                 # generic guard: target must exceed r_c
        bar_r = r_c_scaled + 1e-4
    tau = r_hat - r_c_scaled
    tau_bar = bar_r - r_c_scaled

    # Dynamic gamma (only when gamma == 0).
    if gamma == 0:
        ratios = np.abs(tau) / np.maximum(np.sqrt(np.diag(D)), 1e-8)
        gamma = max(1e-6, np.min(ratios) - 1e-5)

    return {
        "n": n_assets, "D": D, "r_hat": r_hat, "r_c": r_c_scaled, "gamma": gamma,
        "beta": beta, "bar_r": bar_r, "tau": tau, "tau_bar": tau_bar,
        "num_samples": n_samples, "scale_returns": scale_returns,
    }


def generate_rmvp1_data(
    file_path: str,
    sheet_name: str = 'AssetReturns',
    r_c: float = 0.00001,
    gamma: float = 0.01,
    beta: float = 1.0,
    target_return_factor: float = 1.1,
    scale_returns: float = 100.0
):
    """
    Generates RMVP1 problem data with Assumption 1 applied:
        keep asset i iff |tau_i| > gamma * sqrt(D_ii).
    Returns the FILTERED D/tau plus bookkeeping (n_full, n_filtered, dropped_gamma,
    dropped_t=0, idx_keep). "n" is the filtered problem size (so zero-padding a BnB
    solution to "n" stays consistent with the returned D). If all assets are dropped,
    D and tau are None.
    """
    data = _base_problem_data(file_path, sheet_name, r_c, gamma, beta,
                              target_return_factor, scale_returns)
    D_full, tau_full, gamma = data["D"], data["tau"], data["gamma"]

    keep = np.abs(tau_full) > gamma * np.sqrt(np.diag(D_full))   # Assumption 1
    idx_keep = np.where(keep)[0]
    data["n_full"] = data["n"]
    data["dropped_gamma"] = int(np.sum(~keep))
    data["dropped_t"] = 0
    data["idx_keep"] = idx_keep.tolist()
    data["n_filtered"] = int(len(idx_keep))
    if len(idx_keep) == 0:
        data["D"] = None
        data["tau"] = None
        data["n"] = 0
        return data
    data["D"] = D_full[np.ix_(idx_keep, idx_keep)]
    data["tau"] = tau_full[idx_keep]
    data["n"] = int(len(idx_keep))          # filtered problem size
    return data


def generate_rmvp2_data(
    file_path: str, 
    sheet_name: str = 'AssetReturns', 
    r_c: float = 0.00001, 
    gamma: float = 0.01, 
    beta: float = 1.0, 
    target_return_factor: float = 1.1,
    t: float = None,
    scale_returns: float = 100.0
):
    """
    Generates RMVP2 problem data from an Excel file.
    Integrates preprocessing assumptions:
      1) |tau_i| / sqrt(D_ii) > gamma (same as RMVP1)
      2) t > min_i beta / (|tau_i|/sqrt(D_ii) - gamma)
         If t == 0, set t to min bound + eps.
    """
    # Independent base build: call the shared helper, NOT generate_rmvp1_data, so
    # RMVP1's own Assumption-1 filtering can never double-apply here.
    base_data = _base_problem_data(file_path, sheet_name, r_c, gamma, beta,
                                   target_return_factor, scale_returns)

    D_full = base_data["D"]
    tau_full = base_data["tau"]
    gamma = base_data["gamma"]
    beta = base_data["beta"]


    # Apply preprocessing assumptions and filtering (inline)
    diag_sqrt = np.sqrt(np.diag(D_full))
    ratio = np.abs(tau_full) / np.maximum(diag_sqrt, 1e-8)

    # Assumption 1: |tau_i| / sqrt(D_ii) > gamma
    keep_mask = ratio > (gamma)
    dropped_gamma = int(np.sum(~keep_mask))
    if not np.any(keep_mask):
        base_data["D"] = None
        base_data["tau"] = None
        base_data["t"] = t
        base_data["idx_keep"] = []
        base_data["n_full"] = base_data["n"]
        base_data["dropped_gamma"] = dropped_gamma
        base_data["dropped_t"] = 0
        base_data["n_filtered"] = 0
        base_data["n"] = 0
        return base_data

    idx_keep = np.where(keep_mask)[0]
    D_f = D_full[np.ix_(idx_keep, idx_keep)]
    tau_f = tau_full[idx_keep]

    # Recompute ratios on filtered set
    diag_sqrt_f = np.sqrt(np.diag(D_f))
    ratio_f = np.abs(tau_f) / np.maximum(diag_sqrt_f, 1e-8)

    # Assumption 2: t > min_i beta / (|tau_i|/sqrt(D_ii) - gamma)
    if t == 0:
        denom = ratio_f - gamma
        denom = np.maximum(denom, 1e-8)
        min_ratio = np.min(ratio_f)
        t_adj = beta / max(min_ratio - gamma, 1e-8) + 1e-8
        threshold = gamma + beta / t_adj
        keep_mask_t = ratio_f > threshold
    elif t is None:
        denom = ratio_f - gamma
        denom = np.maximum(denom, 1e-8)
        min_t_bound = np.min(beta / denom)
        t_adj = min_t_bound + 1e-8
        threshold = gamma + beta / t_adj
        keep_mask_t = ratio_f > (threshold)
    else:
        t_adj = t
        # Filter by t lower bound: |tau_i|/sqrt(D_ii) > gamma + beta / t
        threshold = gamma + beta / t_adj
        keep_mask_t = ratio_f > (threshold)
    dropped_t = int(np.sum(~keep_mask_t))
    if not np.any(keep_mask_t):
        base_data["D"] = None
        base_data["tau"] = None
        base_data["t"] = t_adj
        base_data["idx_keep"] = []
        base_data["n_full"] = base_data["n"]
        base_data["dropped_gamma"] = dropped_gamma
        base_data["dropped_t"] = dropped_t
        base_data["n_filtered"] = 0
        base_data["n"] = 0
        return base_data

    idx_keep = idx_keep[keep_mask_t]
    D_f = D_full[np.ix_(idx_keep, idx_keep)]
    tau_f = tau_full[idx_keep]

    base_data["D"] = D_f
    base_data["tau"] = tau_f
    base_data["t"] = t_adj
    base_data["idx_keep"] = idx_keep.tolist()
    base_data["n_full"] = base_data["n"]
    base_data["dropped_gamma"] = dropped_gamma
    base_data["dropped_t"] = dropped_t
    base_data["n_filtered"] = len(tau_f)
    base_data["n"] = int(len(tau_f))          # filtered problem size (consistent w/ RMVP1)
    return base_data





def load_experiment_config(config_path: str = "experiment_config.json"):
    """
    Load experiment configuration from a JSON file, resolved relative
    to this script's directory. Returns an empty dict if the file is
    missing.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, config_path)
    if not os.path.exists(full_path):
        return {}
    with open(full_path, "r") as f:
        return json.load(f)


def get_dataset_files():
    """
    Return all .xlsx dataset files under the rmvp/datasets folder,
    regardless of whether the current working directory is the repo root
    or the rmvp subfolder.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # .../rmvp
    data_dir = os.path.join(base_dir, "datasets")
    pattern = os.path.join(data_dir, "*.xlsx")
    return sorted(glob.glob(pattern))


