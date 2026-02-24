import numpy as np
import pandas as pd
import time
import glob
import json
import os
from itertools import product
from datetime import datetime
from main import mainRMVP1BnB, RMVP1_mipGUROBI, mainRMVP2BnB, RMVP2_mipGUROBI, zeropadding

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
    Generates RMVP1 problem data from an Excel file.
    If gamma is passed as 0, it calculates a dynamic gamma based on the feasibility boundary:
    gamma = min(|tau| / sqrt(D_ii)) - eps
    """
    # 1. Load Data (No header assumed)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    returns = df.values * scale_returns  # Matrix of daily returns (scaled)
    n_samples, n_assets = returns.shape
    
    # 2. Compute Statistics
    r_hat = returns.mean(axis=0)
    returns_centered = returns - r_hat

    D = (returns_centered.T @ returns_centered) / (n_samples - 1)
    
    # 3. Setup Problem Parameters
    # Scale risk-free rate to match scaled returns
    r_c_scaled = r_c * scale_returns

    if r_c == 0:
        #avg_pos_return = np.mean(r_hat[r_hat > 0]) if np.any(r_hat > 0) else 0.001
        #bar_r = avg_pos_return * target_return_factor
        bar_r = 0.001 * scale_returns
    else:
        bar_r = r_c_scaled * target_return_factor
    
    if bar_r <= r_c_scaled:
            # Adjust bar_r if assumption violated, or let it slide? 
            # For testing, we might want to enforce valid inputs, but let's just proceed or adjust.
            bar_r = r_c_scaled + 1e-4
    
    # Calculate Excess parameters
    tau = r_hat - r_c_scaled             
    tau_bar = bar_r - r_c_scaled
    
    # Calculate ratios for gamma logic
    D_diag_sqrt = np.sqrt(np.diag(D))
    # Avoid division by zero
    ratios = np.abs(tau) / np.maximum(D_diag_sqrt, 1e-8)
    min_ratio = np.min(ratios)

    # Dynamic Gamma Logic
    if gamma == 0:
        eps = 1e-5
        # Ensure gamma is positive and strictly less than min_ratio
        gamma = max(1e-6, min_ratio - eps)

    # Scale beta consistently with covariance scaling (D scales with scale_returns^2)
    #beta_scaled = beta * (scale_returns ** 2)

    # 4. Pack into Dictionary
    problem_data = {
        "n": n_assets,
        "D": D,
        "r_hat": r_hat,
        "r_c": r_c_scaled,
        "gamma": gamma,
        "beta": beta,
        "bar_r": bar_r,
        "tau": tau,
        "tau_bar": tau_bar,
        "num_samples": n_samples,
        "scale_returns": scale_returns
    }
    
    return problem_data


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
    # Reuse RMVP1 generation for common parameters
    base_data = generate_rmvp1_data(
        file_path, sheet_name, r_c, gamma, beta, target_return_factor, scale_returns
    )

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


def run_experiments_rmvp1():
    results = []

    # Load configuration
    config = load_experiment_config()
    cfg_rmvp1 = config.get("rmvp1", {})
    dataset_params = cfg_rmvp1.get("datasets", {})

    # Discover datasets under rmvp/datasets/ (path is resolved relative to this file)
    data_files = get_dataset_files()
    
    if not data_files:
        print("No datasets found.")
        return

    # Timestamp for the output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp1_experiment_results_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    # Loop over datasets so we can pick a parameter grid for each file
    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        # If there is no config for this dataset, or it is disabled, skip it
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        rc_values = ds_cfg["rc_values"]
        beta_values = ds_cfg["beta_values"]
        target_return_factors = ds_cfg["target_return_factors"]
        gamma_inputs = ds_cfg["gamma_inputs"]

        # Generate combinations of parameters excluding gamma
        param_combinations = product(rc_values, beta_values, target_return_factors)

        for r_c, beta, tr_factor in param_combinations:
            print(f"Processing {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor}")

            for gamma_in in gamma_inputs:
                # Generate data with specific or dynamic gamma
                problem_data = generate_rmvp1_data(
                    file_path=file_path,
                    sheet_name=0,
                    r_c=r_c,
                    gamma=gamma_in, 
                    beta=beta,
                    target_return_factor=tr_factor
                )

                if not problem_data:
                    continue

                D = problem_data['D']
                tau = problem_data['tau']
                tau_bar = problem_data['tau_bar']
                gamma = problem_data['gamma'] # The actual used gamma
                beta_scaled = problem_data['beta']
                n = problem_data['n']
                
                # Count assets violating assumption
                D_diag_sqrt = np.sqrt(np.diag(D))
                feasibility_margin = np.abs(tau) - (gamma * D_diag_sqrt)
                num_removed = np.sum(feasibility_margin <= 0)
                
                print(f"    Running gamma={gamma:.6f} (Dropped: {num_removed})")

                tau_minus_gamma_bnb = "ALL_DROPPED"
                tau_minus_gamma_mip_gurobi = "ALL_DROPPED"
                nnz_bnb = "ALL_DROPPED"
                nnz_mip_gurobi = "ALL_DROPPED"
                bnb_time = bnb_time_cpu = 0
                mip_time_gurobi = mip_time_gurobi_cpu = 0

                # Check if all assets are dropped
                if num_removed == n or num_removed == n-1:
                    print(f"    Skipping solver: All {n} assets dropped.")
                    obj_bnb = "ALL_DROPPED"
                    obj_mip_gurobi = "ALL_DROPPED"
                else:
                    # --- BnB Solver ---
                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_bnb, _, supp_bnb, _ = mainRMVP1BnB(D, tau, tau_bar, gamma, beta_scaled)
                    bnb_time = time.time() - start_time
                    bnb_time_cpu = time.process_time() - start_cpu
                    x_bnb = zeropadding(x_bnb, supp_bnb, n)
                    obj_bnb = (x_bnb.T @ D @ x_bnb + beta_scaled * np.sum(np.abs(x_bnb) > 1e-8))[0][0]
                    tauTx_bnb = (tau.T @ x_bnb)[0]
                    normD_bnb = np.sqrt(x_bnb.T @ D @ x_bnb)[0]
                    tau_minus_gamma_bnb = tauTx_bnb - gamma * normD_bnb
                    nnz_bnb = int(np.sum(np.abs(x_bnb) > 1e-8))
                    print("bnb completed")

                    # --- MIP GUROBI Solver ---
                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_mip_gurobi, _, gap_mip_gurobi = RMVP1_mipGUROBI(D, tau, tau_bar, gamma, beta_scaled)
                    mip_time_gurobi = time.time() - start_time
                    mip_time_gurobi_cpu = time.process_time() - start_cpu
                    obj_mip_gurobi = (x_mip_gurobi.T @ D @ x_mip_gurobi + beta_scaled * np.sum(np.abs(x_mip_gurobi) > 1e-8))[0][0]
                    tauTx_mip_gurobi = (tau.T @ x_mip_gurobi)[0]
                    normD_mip_gurobi = np.sqrt(x_mip_gurobi.T @ D @ x_mip_gurobi)[0]
                    tau_minus_gamma_mip_gurobi = tauTx_mip_gurobi - gamma * normD_mip_gurobi
                    nnz_mip_gurobi = int(np.sum(np.abs(x_mip_gurobi) > 1e-8))
                    print("mip gurobi completed")


                # Store result
                result_row = {
                    "Dataset": file_path,
                    "n": n,
                    "gamma": gamma,
                    "r_c": r_c,
                    "beta": beta,
                    "beta_scaled": beta_scaled,
                    "tr_factor": tr_factor,
                    "dropped_assets": num_removed,
                    "tau_bar": tau_bar,
                    "obj_BnB": obj_bnb,
                    "time_BnB": bnb_time,
                    "time_cpu_BnB": bnb_time_cpu,
                    "nnz_BnB": nnz_bnb,
                    "obj_MIP_GUROBI": obj_mip_gurobi,
                    "time_MIP_GUROBI": mip_time_gurobi,
                    "time_cpu_MIP_GUROBI": mip_time_gurobi_cpu,
                    "nnz_MIP_GUROBI": nnz_mip_gurobi,
                    "gap_MIP_GUROBI": gap_mip_gurobi,
                    "tauTx_minus_gammaNormD_BnB": tau_minus_gamma_bnb if num_removed != n else "ALL_DROPPED",
                    "tauTx_minus_gammaNormD_MIP_GUROBI": tau_minus_gamma_mip_gurobi if num_removed != n else "ALL_DROPPED",
                    "gap_MIP_GUROBI": gap_mip_gurobi if num_removed != n else "ALL_DROPPED"
                }
                results.append(result_row)
                
                # Append to Excel immediately
                df_row = pd.DataFrame([result_row])
                
                if glob.glob(output_file):
                    with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                        try:
                            existing_df = pd.read_excel(output_file)
                            start_row = len(existing_df) + 1
                            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
                        except ValueError:
                                df_row.to_excel(writer, index=False)
                else:
                    df_row.to_excel(output_file, index=False)

    print(f"Experiments completed. Results saved to {output_file}")


def run_experiments_rmvp2(run_gurobi: bool = False):
    results = []

    # Load configuration
    config = load_experiment_config()
    cfg_rmvp2 = config.get("rmvp2", {})
    dataset_params = cfg_rmvp2.get("datasets", {})

    # Discover datasets under rmvp/datasets/ (path is resolved relative to this file)
    data_files = get_dataset_files()
    
    if not data_files:
        print("No datasets found.")
        return

    # Timestamp for the output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp2_experiment_results_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    # Loop over datasets so we can pick a parameter grid for each file
    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        # If there is no config for this dataset, or it is disabled, skip it
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        rc_values = ds_cfg["rc_values"]
        beta_values = ds_cfg["beta_values"]
        target_return_factors = ds_cfg["target_return_factors"]
        gamma_inputs = ds_cfg["gamma_inputs"]
        # Optional list of t values to sweep over. If omitted, t is chosen automatically.
        t_values = ds_cfg.get("t_values", [None])

        # Generate combinations of parameters excluding gamma
        param_combinations = product(rc_values, beta_values, target_return_factors, t_values)

        for r_c, beta, tr_factor, t_val in param_combinations:
            t_label = "auto" if t_val is None else f"{t_val}"
            print(f"Processing RMVP2 {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor} | t={t_label}")

            for gamma_in in gamma_inputs:
                # Generate data with specific or dynamic gamma and optional t
                problem_data = generate_rmvp2_data(
                    file_path=file_path,
                    sheet_name=0,
                    r_c=r_c,
                    gamma=gamma_in, 
                    beta=beta,
                    target_return_factor=tr_factor,
                    t=t_val
                )
                
                if not problem_data:
                    continue

                D = problem_data["D"]
                tau = problem_data["tau"]
                tau_bar = problem_data["tau_bar"]
                gamma = problem_data["gamma"]  # The actual used gamma
                beta_scaled = problem_data["beta"]
                t = problem_data["t"]
                n_full = problem_data.get("n_full", len(tau) if tau is not None else 0)
                n = problem_data.get("n_filtered", len(tau) if tau is not None else 0)
                dropped_gamma = problem_data.get("dropped_gamma", 0)
                dropped_t = problem_data.get("dropped_t", 0)
                
                if n == 0 or n == 1:
                    print("    Skipping solver: insufficient assets after assumption filtering.")
                    obj_bnb = obj_mip_cvx = obj_mip_gurobi = "ALL_DROPPED"
                    bnb_time = bnb_time_cpu = mip_time_cvx = mip_time_gurobi = mip_time_gurobi_cpu = 0
                    nnz_bnb = nnz_mip_gurobi = "ALL_DROPPED"
                    tau_minus_gamma_bnb = tau_minus_gamma_mip_gurobi = "ALL_DROPPED"
                    gap_mip_gurobi = "ALL_DROPPED"
                    if not run_gurobi:
                        obj_mip_gurobi = "no gurobi"
                        mip_time_gurobi = "no gurobi"
                        mip_time_gurobi_cpu = "no gurobi"
                        nnz_mip_gurobi = "no gurobi"
                        gap_mip_gurobi = "no gurobi"
                        tau_minus_gamma_mip_gurobi = "no gurobi"
                    num_removed = dropped_gamma + dropped_t
                else:
                    num_removed = dropped_gamma + dropped_t
                    print(f"    Running gamma={gamma:.6f} | t={t:.4f} (Dropped: {num_removed})")
                    # --- BnB Solver ---
                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_bnb, _, supp_bnb, _ = mainRMVP2BnB(D, tau, tau_bar, gamma, beta_scaled, t)
                    bnb_time = time.time() - start_time
                    bnb_time_cpu = time.process_time() - start_cpu
                    x_bnb = zeropadding(x_bnb, supp_bnb, n)
                    # Recalculate objective: -tau^T x + tau_bar + gamma * ||x||_D + beta * ||x||_0
                    quad_bnb = (x_bnb.T @ D @ x_bnb)[0][0]
                    norm_D_bnb = np.sqrt(quad_bnb)
                    tauTx_bnb = (tau.T @ x_bnb)[0]
                    obj_bnb = -tauTx_bnb + tau_bar + gamma * norm_D_bnb + beta_scaled * np.sum(np.abs(x_bnb) > 1e-8)
                    tau_minus_gamma_bnb = tauTx_bnb - gamma * norm_D_bnb
                    nnz_bnb = int(np.sum(np.abs(x_bnb) > 1e-8))
                    print("bnb completed")

                    if run_gurobi:
                        # --- MIP GUROBI Solver ---
                        start_time = time.time()
                        start_cpu = time.process_time()
                        x_mip_gurobi, _, gap_mip_gurobi = RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta_scaled, t)
                        mip_time_gurobi = time.time() - start_time
                        mip_time_gurobi_cpu = time.process_time() - start_cpu
                        quad_gurobi = (x_mip_gurobi.T @ D @ x_mip_gurobi)[0][0]
                        norm_D_gurobi = np.sqrt(quad_gurobi)
                        tauTx_mip_gurobi = (tau.T @ x_mip_gurobi)[0]
                        obj_mip_gurobi = -tauTx_mip_gurobi + tau_bar + gamma * norm_D_gurobi + beta_scaled * np.sum(np.abs(x_mip_gurobi) > 1e-8)
                        tau_minus_gamma_mip_gurobi = tauTx_mip_gurobi - gamma * norm_D_gurobi
                        nnz_mip_gurobi = int(np.sum(np.abs(x_mip_gurobi) > 1e-8))
                        print("mip gurobi completed")
                    else:
                        obj_mip_gurobi = "no gurobi"
                        mip_time_gurobi = "no gurobi"
                        mip_time_gurobi_cpu = "no gurobi"
                        nnz_mip_gurobi = "no gurobi"
                        gap_mip_gurobi = "no gurobi"
                        tau_minus_gamma_mip_gurobi = "no gurobi"

                # Store result
                result_row = {
                    "Dataset": file_path,
                    "n": n_full,
                    "n_filtered": n,
                    "gamma": gamma,
                    "r_c": r_c,
                    "beta": beta,
                    "beta_scaled": beta_scaled,
                    "tr_factor": tr_factor,
                    "t": t,
                    "dropped_assets": num_removed,
                    "dropped_gamma": dropped_gamma,
                    "dropped_t": dropped_t,
                    "tau_bar": tau_bar,
                    "obj_BnB": obj_bnb,
                    "time_BnB": bnb_time,
                    "time_cpu_BnB": bnb_time_cpu,
                    "nnz_BnB": nnz_bnb,
                    "obj_MIP_GUROBI": obj_mip_gurobi,
                    "time_MIP_GUROBI": mip_time_gurobi,
                    "time_cpu_MIP_GUROBI": mip_time_gurobi_cpu,
                    "nnz_MIP_GUROBI": nnz_mip_gurobi,
                    "gap_MIP_GUROBI": gap_mip_gurobi,
                    "tauTx_minus_gammaNormD_BnB": tau_minus_gamma_bnb if num_removed != n_full else "ALL_DROPPED",
                    "tauTx_minus_gammaNormD_MIP_GUROBI": tau_minus_gamma_mip_gurobi if num_removed != n_full else "ALL_DROPPED"
                }
                results.append(result_row)
                
                # Append to Excel immediately
                df_row = pd.DataFrame([result_row])
                
                if glob.glob(output_file):
                    with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                        try:
                            existing_df = pd.read_excel(output_file)
                            start_row = len(existing_df) + 1
                            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
                        except ValueError:
                                df_row.to_excel(writer, index=False)
                else:
                    df_row.to_excel(output_file, index=False)

    print(f"RMVP2 Experiments completed. Results saved to {output_file}")



if __name__ == "__main__":

    #run_experiments_rmvp1()
    run_experiments_rmvp2()



