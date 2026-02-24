import numpy as np
import pandas as pd
import time
from datetime import datetime
from itertools import product

from main import mainRMVP1BnB, RMVP1_mipGUROBI, mainRMVP2BnB, RMVP2_mipGUROBI, zeropadding
from test import generate_rmvp1_data, generate_rmvp2_data, load_experiment_config, get_dataset_files
from warm_start import warm_start_rmvp1, warm_start_rmvp2


def solve_bnb_rmvp2(D, tau, tau_bar, gamma, beta, t):
    n = D.shape[0]
    start_time = time.time()
    start_cpu = time.process_time()
    x_bnb, _, supp_bnb, _ = mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t)
    time_wall = time.time() - start_time
    time_cpu = time.process_time() - start_cpu

    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    quad = (x_bnb.T @ D @ x_bnb)[0][0]
    norm_D = np.sqrt(quad)
    tauTx = (tau.T @ x_bnb)[0]
    obj = -tauTx + tau_bar + gamma * norm_D + beta * np.sum(np.abs(x_bnb) > 1e-8)
    nnz = int(np.sum(np.abs(x_bnb) > 1e-8))
    return obj, nnz, time_wall, time_cpu


def solve_bnb_rmvp1(D, tau, tau_bar, gamma, beta):
    n = D.shape[0]
    start_time = time.time()
    start_cpu = time.process_time()
    x_bnb, _, supp_bnb, _ = mainRMVP1BnB(D, tau, tau_bar, gamma, beta)
    time_wall = time.time() - start_time
    time_cpu = time.process_time() - start_cpu

    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    obj = (x_bnb.T @ D @ x_bnb + beta * np.sum(np.abs(x_bnb) > 1e-8))[0][0]
    nnz = int(np.sum(np.abs(x_bnb) > 1e-8))
    return obj, nnz, time_wall, time_cpu


def solve_gurobi_rmvp2(D, tau, tau_bar, gamma, beta, t):
    start_time = time.time()
    start_cpu = time.process_time()
    x_mip_gurobi, _, gap_mip_gurobi = RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta, t)
    time_wall = time.time() - start_time
    time_cpu = time.process_time() - start_cpu

    quad = (x_mip_gurobi.T @ D @ x_mip_gurobi)[0][0]
    norm_D = np.sqrt(quad)
    tauTx = (tau.T @ x_mip_gurobi)[0]
    obj = -tauTx + tau_bar + gamma * norm_D + beta * np.sum(np.abs(x_mip_gurobi) > 1e-8)
    nnz = int(np.sum(np.abs(x_mip_gurobi) > 1e-8))
    return obj, nnz, gap_mip_gurobi, time_wall, time_cpu


def solve_gurobi_rmvp1(D, tau, tau_bar, gamma, beta):
    start_time = time.time()
    start_cpu = time.process_time()
    x_mip_gurobi, _, gap_mip_gurobi = RMVP1_mipGUROBI(D, tau, tau_bar, gamma, beta)
    time_wall = time.time() - start_time
    time_cpu = time.process_time() - start_cpu

    obj = (x_mip_gurobi.T @ D @ x_mip_gurobi + beta * np.sum(np.abs(x_mip_gurobi) > 1e-8))[0][0]
    nnz = int(np.sum(np.abs(x_mip_gurobi) > 1e-8))
    return obj, nnz, gap_mip_gurobi, time_wall, time_cpu


def run_warm_start_rmvp1_compare(drop_fractions=None, run_gurobi: bool = False):
    results = []
    gurobi_cache = {}

    config = load_experiment_config()
    cfg_rmvp1 = config.get("rmvp1", {})
    dataset_params = cfg_rmvp1.get("datasets", {})

    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    if drop_fractions is None:
        drop_fractions = [0.1]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp1_warm_start_compare_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    for file_path in data_files:
        ds_key = file_path.split("\\")[-1]
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        rc_values = ds_cfg["rc_values"]
        beta_values = ds_cfg["beta_values"]
        target_return_factors = ds_cfg["target_return_factors"]
        gamma_inputs = ds_cfg["gamma_inputs"]

        param_combinations = product(rc_values, beta_values, target_return_factors)

        for r_c, beta, tr_factor in param_combinations:
            print(f"Processing RMVP1 {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor}")

            for gamma_in in gamma_inputs:
                problem_data = generate_rmvp1_data(
                    file_path=file_path,
                    sheet_name=0,
                    r_c=r_c,
                    gamma=gamma_in,
                    beta=beta,
                    target_return_factor=tr_factor,
                )

                if not problem_data:
                    continue

                D = problem_data["D"]
                tau = problem_data["tau"]
                tau_bar = problem_data["tau_bar"]
                gamma = problem_data["gamma"]
                beta_scaled = problem_data["beta"]
                n = problem_data["n"]

                diag_sqrt = np.sqrt(np.diag(D))
                num_removed = int(np.sum(np.abs(tau) <= gamma * diag_sqrt))
                if num_removed >= n - 1:
                    print("    Skipping solver: too many assets violate assumptions.")
                    continue

                if run_gurobi:
                    cache_key = (file_path, r_c, beta, tr_factor, gamma)
                    if cache_key in gurobi_cache:
                        obj_gurobi, nnz_gurobi, gap_gurobi, time_gurobi, time_gurobi_cpu = gurobi_cache[cache_key]
                    else:
                        obj_gurobi, nnz_gurobi, gap_gurobi, time_gurobi, time_gurobi_cpu = solve_gurobi_rmvp1(
                            D, tau, tau_bar, gamma, beta_scaled
                        )
                        gurobi_cache[cache_key] = (obj_gurobi, nnz_gurobi, gap_gurobi, time_gurobi, time_gurobi_cpu)
                else:
                    obj_gurobi = "no gurobi"
                    nnz_gurobi = "no gurobi"
                    gap_gurobi = "no gurobi"
                    time_gurobi = "no gurobi"
                    time_gurobi_cpu = "no gurobi"

                for drop_fraction in drop_fractions:
                    warm = warm_start_rmvp1(D, tau, tau_bar, gamma, beta_scaled, drop_fraction=drop_fraction)
                    D_red = warm["D_reduced"]
                    tau_red = warm["tau_reduced"]
                    n_warm_start = int(len(tau_red))
                    warm_dropped = int(n - n_warm_start)

                    obj_bnb, nnz_bnb, time_bnb, time_bnb_cpu = solve_bnb_rmvp1(
                        D_red, tau_red, tau_bar, gamma, beta_scaled
                    )

                    result_row = {
                        "Dataset": file_path,
                        "n": n,
                        "gamma": gamma,
                        "r_c": r_c,
                        "beta": beta,
                        "beta_scaled": beta_scaled,
                        "tr_factor": tr_factor,
                        "assumption_dropped_assets": num_removed,
                        "drop_fraction": drop_fraction,
                        "n_warm_start": n_warm_start,
                        "warm_start_dropped_assets": warm_dropped,
                        "obj_BnB_warm": obj_bnb,
                        "nnz_BnB_warm": nnz_bnb,
                        "time_BnB_warm": time_bnb,
                        "time_cpu_BnB_warm": time_bnb_cpu,
                        "obj_MIP_GUROBI": obj_gurobi,
                        "nnz_MIP_GUROBI": nnz_gurobi,
                        "time_MIP_GUROBI": time_gurobi,
                        "time_cpu_MIP_GUROBI": time_gurobi_cpu,
                        "gap_MIP_GUROBI": gap_gurobi,
                        "diff_BnB_vs_GUROBI": (obj_bnb - obj_gurobi) if run_gurobi else "no gurobi",
                    }
                    results.append(result_row)

                    df_row = pd.DataFrame([result_row])
                    try:
                        existing_df = pd.read_excel(output_file)
                        start_row = len(existing_df) + 1
                        with pd.ExcelWriter(output_file, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
                    except FileNotFoundError:
                        df_row.to_excel(output_file, index=False)

    print(f"Warm-start RMVP1 comparison completed. Results saved to {output_file}")


def run_warm_start_rmvp2_compare(drop_fractions=None, run_gurobi: bool = False):
    results = []
    gurobi_cache = {}

    config = load_experiment_config()
    cfg_rmvp2 = config.get("rmvp2", {})
    dataset_params = cfg_rmvp2.get("datasets", {})

    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    if drop_fractions is None:
        drop_fractions = [0.1]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp2_warm_start_compare_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    for file_path in data_files:
        ds_key = file_path.split("\\")[-1]
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        rc_values = ds_cfg["rc_values"]
        beta_values = ds_cfg["beta_values"]
        target_return_factors = ds_cfg["target_return_factors"]
        gamma_inputs = ds_cfg["gamma_inputs"]
        t_values = ds_cfg.get("t_values", [None])

        param_combinations = product(rc_values, beta_values, target_return_factors, t_values)

        for r_c, beta, tr_factor, t_val in param_combinations:
            t_label = "auto" if t_val is None else f"{t_val}"
            print(f"Processing RMVP2 {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor} | t={t_label}")

            for gamma_in in gamma_inputs:
                problem_data = generate_rmvp2_data(
                    file_path=file_path,
                    sheet_name=0,
                    r_c=r_c,
                    gamma=gamma_in,
                    beta=beta,
                    target_return_factor=tr_factor,
                    t=t_val,
                )

                if not problem_data:
                    continue

                D = problem_data["D"]
                tau = problem_data["tau"]
                tau_bar = problem_data["tau_bar"]
                gamma = problem_data["gamma"]
                beta_scaled = problem_data["beta"]
                t = problem_data["t"]
                n_full = problem_data.get("n_full", len(tau) if tau is not None else 0)
                n = problem_data.get("n_filtered", len(tau) if tau is not None else 0)
                dropped_gamma = problem_data.get("dropped_gamma", 0)
                dropped_t = problem_data.get("dropped_t", 0)
                num_removed = dropped_gamma + dropped_t

                if n == 0 or n == 1:
                    print("    Skipping solver: insufficient assets after assumption filtering.")
                    continue

                if run_gurobi:
                    cache_key = (file_path, r_c, beta, tr_factor, gamma, t)
                    if cache_key in gurobi_cache:
                        obj_gurobi, nnz_gurobi, gap_gurobi, time_gurobi, time_gurobi_cpu = gurobi_cache[cache_key]
                    else:
                        obj_gurobi, nnz_gurobi, gap_gurobi, time_gurobi, time_gurobi_cpu = solve_gurobi_rmvp2(
                            D, tau, tau_bar, gamma, beta_scaled, t
                        )
                        gurobi_cache[cache_key] = (obj_gurobi, nnz_gurobi, gap_gurobi, time_gurobi, time_gurobi_cpu)
                else:
                    obj_gurobi = "no gurobi"
                    nnz_gurobi = "no gurobi"
                    gap_gurobi = "no gurobi"
                    time_gurobi = "no gurobi"
                    time_gurobi_cpu = "no gurobi"

                for drop_fraction in drop_fractions:
                    warm = warm_start_rmvp2(D, tau, tau_bar, gamma, beta_scaled, t, drop_fraction=drop_fraction)
                    D_red = warm["D_reduced"]
                    tau_red = warm["tau_reduced"]
                    n_warm_start = int(len(tau_red))
                    warm_dropped = int(n - n_warm_start)

                    obj_bnb, nnz_bnb, time_bnb, time_bnb_cpu = solve_bnb_rmvp2(
                        D_red, tau_red, tau_bar, gamma, beta_scaled, t
                    )

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
                        "drop_fraction": drop_fraction,
                        "n_warm_start": n_warm_start,
                        "warm_start_dropped_assets": warm_dropped,
                        "obj_BnB_warm": obj_bnb,
                        "nnz_BnB_warm": nnz_bnb,
                        "time_BnB_warm": time_bnb,
                        "time_cpu_BnB_warm": time_bnb_cpu,
                        "obj_MIP_GUROBI": obj_gurobi,
                        "nnz_MIP_GUROBI": nnz_gurobi,
                        "time_MIP_GUROBI": time_gurobi,
                        "time_cpu_MIP_GUROBI": time_gurobi_cpu,
                        "gap_MIP_GUROBI": gap_gurobi,
                        "diff_BnB_vs_GUROBI": (obj_bnb - obj_gurobi) if run_gurobi else "no gurobi",
                    }
                    results.append(result_row)

                    df_row = pd.DataFrame([result_row])
                    try:
                        existing_df = pd.read_excel(output_file)
                        start_row = len(existing_df) + 1
                        with pd.ExcelWriter(output_file, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
                    except FileNotFoundError:
                        df_row.to_excel(output_file, index=False)

    print(f"Warm-start RMVP2 comparison completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_warm_start_rmvp2_compare(drop_fractions=[0.8, 0.9])
