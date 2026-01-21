import numpy as np
import pandas as pd
import time
import glob
from itertools import product
from datetime import datetime
from main import generate_rmvp1_data, mainRMVP1BnB, RMVP1_mipGUROBI, zeropadding


def rmvp1_gloabal_lower_bound_check(D, tau, tau_bar, gamma, beta, x, tol=1e-8):
    """
    Check the RMVP1 global lower-bound theorem for a given solution x (padded).

    Args:
        D: (n,n) covariance/PSD matrix.
        tau: (n,) ccr vector (excess returns).
        tau_bar: scalar bar{ccr}.
        gamma: uncertainty radius.
        beta: sparsity penalty.
        x: (n,1) or (n,) full solution vector (padded).
        tol: support threshold for |x_i|.

    Returns:
        {
          "eta": eta_scalar,
          "support_indices": list of indices in support,
          "details": list of dicts per i in support with keys:
             "i", "rho_i", "bound1", "bound2", "lower_bound", "abs_xi", "satisfied"
          "all_satisfied": bool
        }
    """
    x = np.asarray(x).reshape(-1)
    tau = np.asarray(tau).reshape(-1)
    D = np.asarray(D)

    diag_D = np.diag(D)
    diag_sqrt = np.sqrt(diag_D)

    support = np.where(np.abs(x) > tol)[0]

    # eta = min_i { (tau_bar^2 * d_i[i]) / (tau[i] - gamma * sqrt(d_i[i]))^2 }
    denom_eta = (tau - gamma * diag_sqrt) ** 2
    denom_eta = np.maximum(denom_eta, 1e-8)
    eta_vals = (tau_bar ** 2) * diag_D / denom_eta
    eta = float(np.min(eta_vals))

    details = []
    all_satisfied = True

    for i in support:
        # Compute rho_i = max_{j!=i} || e_i ± ((|tau_i|+gamma*sqrt(D_ii)) / (|tau_j| - gamma*sqrt(D_jj))) e_j ||_D
        rho_i = 0.0
        numer_i = np.abs(tau[i]) + gamma * diag_sqrt[i]
        for j in range(len(tau)):
            if j == i:
                continue
            denom_j = np.abs(tau[j]) - gamma * diag_sqrt[j]
            if denom_j <= 0:
                continue  # violates assumption; skip this j
            ratio = numer_i / denom_j
            e_i = np.zeros_like(tau)
            e_j = np.zeros_like(tau)
            e_i[i] = 1.0
            e_j[j] = 1.0
            for sgn in (-1.0, 1.0):
                v = e_i + sgn * ratio * e_j
                norm_v = float(np.sqrt(v.T @ D @ v))
                rho_i = max(rho_i, norm_v)
        if rho_i == 0.0:
            print(f"rho_i is 0 for i={i}")
            rho_i = np.inf

        # Bounds
        denom_i = np.abs(tau[i]) - gamma * diag_sqrt[i]
        bound2 = tau_bar / denom_i if denom_i > 0 else np.inf
        bound1 = (np.sqrt(eta + beta) - np.sqrt(eta)) / rho_i if np.isfinite(rho_i) else np.inf
        lower_bound = min(bound1, bound2)

        abs_xi = np.abs(x[i])
        satisfied = abs_xi + tol >= lower_bound
        all_satisfied = all_satisfied and satisfied

        details.append({
            "i": int(i),
            "rho_i": rho_i,
            "bound1": bound1,
            "bound2": bound2,
            "lower_bound": lower_bound,
            "abs_xi": abs_xi,
            "satisfied": bool(satisfied)
        })

    return {
        "eta": eta,
        "support_indices": support.tolist(),
        "details": details,
        "all_satisfied": all_satisfied
    }


def run_rmvp1_boundcheck():
    """
    Run RMVP1 experiments and check the global lower-bound theorem for BnB and GUROBI.
    Results are written to an Excel file.
    """
    data_files = glob.glob("rmvp/datasets/*.xlsx") + glob.glob("datasets/*.xlsx")
    data_files = sorted(set(data_files))
    if not data_files:
        print("No datasets found.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp1_boundcheck_results_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    rc_values = [0.0002, 0.0004]
    beta_values = [1e-4, 1e-5, 5e-6, 1e-6]
    target_return_factors = [1.05, 1.1, 1.15]
    gamma_inputs = [0.001, 0.01, 0.1, 0.0]

    param_combinations = product(data_files, rc_values, beta_values, target_return_factors)

    def _min_margin(check_res):
        if not check_res or not check_res.get("details"):
            return "NO_SUPPORT"
        margins = [d["abs_xi"] - d["lower_bound"] for d in check_res["details"]]
        return float(np.min(margins))

    for file_path, r_c, beta, tr_factor in param_combinations:
        print(f"Processing {file_path} | r_c={r_c} | beta={beta} | tr={tr_factor}")
        for gamma_in in gamma_inputs:
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
            gamma = problem_data['gamma']
            beta_scaled = problem_data['beta']
            n = problem_data['n']

            D_diag_sqrt = np.sqrt(np.diag(D))
            feasibility_margin = np.abs(tau) - (gamma * D_diag_sqrt)
            num_removed = np.sum(feasibility_margin <= 0)

            print(f"    Running gamma={gamma:.6f} (Dropped: {num_removed})")

            # Defaults
            tau_minus_gamma_bnb = "ALL_DROPPED"
            tau_minus_gamma_mip_gurobi = "ALL_DROPPED"
            nnz_bnb = nnz_mip_gurobi = "ALL_DROPPED"
            bnb_time = bnb_time_cpu = 0
            mip_time_gurobi = mip_time_gurobi_cpu = 0
            gap_mip_gurobi = "ALL_DROPPED"
            bound_ok_bnb = bound_ok_gurobi = "ALL_DROPPED"
            bound_margin_bnb = bound_margin_gurobi = "ALL_DROPPED"

            if num_removed == n or num_removed == n - 1:
                print(f"    Skipping solver: All {n} assets dropped.")
                obj_bnb = "ALL_DROPPED"
                obj_mip_gurobi = "ALL_DROPPED"
            else:
                # BnB
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
                chk_bnb = rmvp1_gloabal_lower_bound_check(D, tau, tau_bar, gamma, beta_scaled, x_bnb)
                bound_ok_bnb = chk_bnb["all_satisfied"]
                bound_margin_bnb = _min_margin(chk_bnb)
                print("bnb completed")

                # GUROBI
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
                chk_gurobi = rmvp1_gloabal_lower_bound_check(D, tau, tau_bar, gamma, beta_scaled, x_mip_gurobi)
                bound_ok_gurobi = chk_gurobi["all_satisfied"]
                bound_margin_gurobi = _min_margin(chk_gurobi)
                print("mip gurobi completed")

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
                "bound_ok_BnB": bound_ok_bnb,
                "bound_margin_BnB": bound_margin_bnb,
                "bound_ok_MIP_GUROBI": bound_ok_gurobi,
                "bound_margin_MIP_GUROBI": bound_margin_gurobi
            }
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

    print(f"RMVP1 bound checks completed. Results saved to {output_file}")


if __name__ == "__main__":
    run_rmvp1_boundcheck()