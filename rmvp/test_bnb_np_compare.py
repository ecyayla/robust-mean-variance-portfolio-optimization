import numpy as np
import pandas as pd
import time
import glob
import os
from datetime import datetime
from queue import PriorityQueue, LifoQueue

from main import solveRMVP1, mainRMVP1BnB, zeropadding, branchVariable_rmvp1
from test import generate_rmvp1_data, load_experiment_config, get_dataset_files


def mainRMVP1BnB_np(
    D,
    tau,
    tau_bar,
    gamma,
    beta,
    method="mosek",
    branch_rule="max_lagrangian_grad",
    traverse_rule="bfs",
    time_limit=None,
):
    """
    Numpy-based implementation of mainRMVP1BnB with array supports.
    The algorithm structure matches mainRMVP1BnB.
    """
    relErr = 1e-8

    if traverse_rule == "bfs":
        q = PriorityQueue()
    elif traverse_rule == "dfs":
        q = LifoQueue()
    else:
        raise ValueError(f"Invalid traverse rule: {traverse_rule}")

    ub = 10e10
    global_ub = ub + 1e2

    x_init, lb, lambda_init = solveRMVP1(D, tau_bar, tau, gamma)

    n = D.shape[1]
    global_supp = np.array([], dtype=np.int32)
    Ssupp = np.array([], dtype=np.int32)
    Psupp = np.arange(n, dtype=np.int32)
    DD = np.diag(D)

    q.put([lb, ub, 0, Ssupp, Psupp, x_init, lambda_init])

    count = 0
    while q.qsize() >= 1:
        [lb, ub, _, Ssupp, Psupp, x1, lambda_val] = q.get()
        count += 1

        if ub - global_ub < relErr:
            global_ub = ub
            global_supp = Ssupp
            global_x = x1

        if global_ub - lb <= relErr:
            break
        if abs(ub - lb) < relErr:
            continue
        if Psupp.size == 0:
            continue

        if Ssupp.size >= 1:
            ind, bb_ind = branchVariable_rmvp1(
                x1, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule
            )
        else:
            tau_Psupp = tau[Psupp]
            var = DD[Psupp]
            bb_dec = var / (tau_Psupp + 1e-8)
            bb_ind = int(np.argmin(bb_dec))
            ind = int(Psupp[bb_ind])

        left_supp = np.sort(np.append(Ssupp, ind)).astype(np.int32)
        Psupp = np.delete(Psupp, bb_ind)

        if (Ssupp.size + Psupp.size) >= 1:
            w = np.concatenate((Ssupp, Psupp)).astype(np.int32)
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, right_lb, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
            right_lb = right_lb + beta * Ssupp.size

            q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, x1, lambda_val])

        if left_supp.size >= 1:
            w = left_supp
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, left_ub, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
            left_ub = left_ub + beta * left_supp.size

            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, x_w_opt, lambda_val])
        else:
            q.put([lb + beta, ub, np.random.rand(), left_supp, Psupp, x1, lambda_val])

    global_supp = np.array(sorted(global_supp.tolist()), dtype=np.int32)
    return global_x, global_ub, global_supp, count


def _calc_obj(D, x, beta):
    return (x.T @ D @ x + beta * np.sum(np.abs(x) > 1e-8))[0][0]


def run_experiments_rmvp1_np_compare():
    results = []

    config = load_experiment_config()
    cfg_rmvp1 = config.get("rmvp1", {})
    dataset_params = cfg_rmvp1.get("datasets", {})

    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp1_bnb_np_compare_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        rc_values = ds_cfg["rc_values"]
        beta_values = ds_cfg["beta_values"]
        target_return_factors = ds_cfg["target_return_factors"]
        gamma_inputs = ds_cfg["gamma_inputs"]

        from itertools import product

        param_combinations = product(rc_values, beta_values, target_return_factors)

        for r_c, beta, tr_factor in param_combinations:
            print(f"Processing {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor}")

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

                D_diag_sqrt = np.sqrt(np.diag(D))
                feasibility_margin = np.abs(tau) - (gamma * D_diag_sqrt)
                num_removed = int(np.sum(feasibility_margin <= 0))

                print(f"    Running gamma={gamma:.6f} (Dropped: {num_removed})")

                if num_removed >= n - 1:
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
                        "obj_BnB": "ALL_DROPPED",
                        "time_BnB": 0,
                        "time_cpu_BnB": 0,
                        "nodes_BnB": "ALL_DROPPED",
                        "obj_BnB_np": "ALL_DROPPED",
                        "time_BnB_np": 0,
                        "time_cpu_BnB_np": 0,
                        "nodes_BnB_np": "ALL_DROPPED",
                        "obj_diff_np_minus_main": "ALL_DROPPED",
                        "time_cpu_diff_np_minus_main": "ALL_DROPPED",
                    }
                else:
                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_bnb, _, supp_bnb, count_bnb = mainRMVP1BnB(D, tau, tau_bar, gamma, beta_scaled)
                    time_bnb = time.time() - start_time
                    time_bnb_cpu = time.process_time() - start_cpu
                    x_bnb = zeropadding(x_bnb, supp_bnb, n)
                    obj_bnb = _calc_obj(D, x_bnb, beta_scaled)

                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_np, _, supp_np, count_np = mainRMVP1BnB_np(D, tau, tau_bar, gamma, beta_scaled)
                    time_np = time.time() - start_time
                    time_np_cpu = time.process_time() - start_cpu
                    x_np = zeropadding(x_np, supp_np.tolist(), n)
                    obj_np = _calc_obj(D, x_np, beta_scaled)

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
                        "time_BnB": time_bnb,
                        "time_cpu_BnB": time_bnb_cpu,
                        "nodes_BnB": count_bnb,
                        "obj_BnB_np": obj_np,
                        "time_BnB_np": time_np,
                        "time_cpu_BnB_np": time_np_cpu,
                        "nodes_BnB_np": count_np,
                        "obj_diff_np_minus_main": obj_np - obj_bnb,
                        "time_cpu_diff_np_minus_main": time_np_cpu - time_bnb_cpu,
                    }

                results.append(result_row)
                df_row = pd.DataFrame([result_row])

                if glob.glob(output_file):
                    with pd.ExcelWriter(output_file, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                        try:
                            existing_df = pd.read_excel(output_file)
                            start_row = len(existing_df) + 1
                            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
                        except ValueError:
                            df_row.to_excel(writer, index=False)
                else:
                    df_row.to_excel(output_file, index=False)

    print(f"Experiments completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_experiments_rmvp1_np_compare()
