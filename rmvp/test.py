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
        avg_pos_return = np.mean(r_hat[r_hat > 0]) if np.any(r_hat > 0) else 0.001
        bar_r = avg_pos_return * target_return_factor
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
    Estimates t if not provided based on H = sqrt(tau^T D^-1 tau).
    """
    # Reuse RMVP1 generation for common parameters
    base_data = generate_rmvp1_data(file_path, sheet_name, r_c, gamma, beta, target_return_factor, scale_returns)
    
    D = base_data['D']
    tau = base_data['tau']
    gamma = base_data['gamma']
    beta = base_data['beta']
    
    # Estimate/enforce t (D-norm constraint bound)
    # Assumption: t > min_i { beta / (|tau[i]|/sqrt(D_ii) - gamma) }
    D_inv = np.linalg.inv(D)
    H = float(np.sqrt(tau.T @ D_inv @ tau))

    D_diag_sqrt = np.sqrt(np.diag(D))
    denominators = (np.abs(tau) / np.maximum(D_diag_sqrt, 1e-8)) - gamma
    # Guard against nonpositive denominators (assumption violation)
    denominators = np.maximum(denominators, 1e-8)
    min_t_bound = np.min(beta / denominators)

    # If t is provided but too small, lift it; otherwise set using H and the bound
    if t is None:
        t = max(1.1 * H, 1.1 * min_t_bound)
    elif t <= min_t_bound:
        print(f"Provided t={t} violates lower bound {min_t_bound:.6e}; adjusting.")
        t = 1.1 * min_t_bound
        
    base_data['t'] = t
    return base_data


def preprocess_rmvp2(D, tau, gamma, beta, t, tol=1e-8):
    """
    Enforce assumptions for RMVP2 by filtering assets:
      1) |tau_i| / sqrt(D_ii) > gamma  (robust feasibility)
      2) Given t > 0: |tau_i| / sqrt(D_ii) > gamma + beta / t
         If t is None or 0: set t to (1+tol) * min_i beta / (|tau_i|/sqrt(D_ii) - gamma)
    Returns filtered (D, tau), adjusted t, and kept indices.
    """
    diag_sqrt = np.sqrt(np.diag(D))
    ratio = np.abs(tau) / np.maximum(diag_sqrt, 1e-8)

    # Filter by gamma assumption
    keep_mask = ratio > (gamma + tol)
    if not np.any(keep_mask):
        return None, None, None, []
    idx_keep = np.where(keep_mask)[0]
    D_f = D[np.ix_(idx_keep, idx_keep)]
    tau_f = tau[idx_keep]
    diag_sqrt = np.sqrt(np.diag(D_f))
    ratio = np.abs(tau_f) / np.maximum(diag_sqrt, 1e-8)

    # Determine or adjust t
    if t is None or t == 0:
        denom = ratio - gamma
        denom = np.maximum(denom, 1e-8)
        min_t_bound = np.min(beta / denom)
        t_adj = (1.0 + tol) * min_t_bound
    else:
        t_adj = t

    # If t_adj is provided, filter assets that violate t lower bound
    threshold = gamma + beta / t_adj
    keep_mask_t = ratio > (threshold + tol)
    if not np.any(keep_mask_t):
        return None, None, None, []
    idx_keep = idx_keep[keep_mask_t]
    D_f = D[np.ix_(idx_keep, idx_keep)]
    tau_f = tau[idx_keep]

    return D_f, tau_f, t_adj, idx_keep


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


def run_experiments_rmvp2():
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

        # Generate combinations of parameters excluding gamma
        param_combinations = product(rc_values, beta_values, target_return_factors)

        for r_c, beta, tr_factor in param_combinations:
            print(f"Processing RMVP2 {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor}")

            for gamma_in in gamma_inputs:
                # Generate data with specific or dynamic gamma
                problem_data = generate_rmvp2_data(
                    file_path=file_path,
                    sheet_name=0,
                    r_c=r_c,
                    gamma=gamma_in, 
                    beta=beta,
                    target_return_factor=tr_factor
                )
                
                if not problem_data:
                    continue

                D_full = problem_data['D']
                tau_full = problem_data['tau']
                tau_bar = problem_data['tau_bar']
                gamma = problem_data['gamma'] # The actual used gamma
                beta_scaled = problem_data['beta']
                t_in = problem_data['t']

                # Enforce assumptions and filter assets
                D, tau, t, idx_keep = preprocess_rmvp2(D_full, tau_full, gamma, beta_scaled, t_in, tol=1e-8)
                n_full = len(tau_full)
                n = len(tau) if tau is not None else 0
                
                if n == 0 or n == 1:
                    print("    Skipping solver: all assets dropped after assumption filtering.")
                    obj_bnb = obj_mip_cvx = obj_mip_gurobi = "ALL_DROPPED"
                    bnb_time = bnb_time_cpu = mip_time_cvx = mip_time_gurobi = mip_time_gurobi_cpu = 0
                    nnz_bnb = nnz_mip_gurobi = "ALL_DROPPED"
                    tau_minus_gamma_bnb = tau_minus_gamma_mip_gurobi = "ALL_DROPPED"
                    gap_mip_gurobi = "ALL_DROPPED"
                    num_removed = n_full
                else:
                    num_removed = n_full - n
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

    run_experiments_rmvp1()
    # run_experiments_rmvp2()