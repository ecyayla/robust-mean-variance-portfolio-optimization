"""
RMVP2 -- Gurobi (MIP) runs, driven by experiment_config.json ("rmvp2").

Product ranges over t_values; generate_rmvp2_data already applies the assumption + t
filters, so Gurobi solves the same filtered set as the BnB script. Solve once per
(dataset, r_c, beta, tr_factor, t, gamma) with Gurobi (single-thread); append one Excel
row per completed run. No warm-start, no drop-fraction loop.

Run standalone:  python3 run_rmvp2_gurobi.py
"""
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _common import (append_row, timestamped_output, solve_gurobi_rmvp2,
                     RMVP_DIR)                                        # noqa: E402
if RMVP_DIR not in sys.path:
    sys.path.insert(0, RMVP_DIR)
from test import generate_rmvp2_data, load_experiment_config, get_dataset_files  # noqa: E402


def main():
    config = load_experiment_config()
    dataset_params = config.get("rmvp2", {}).get("datasets", {})
    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    output_file = timestamped_output("rmvp2_gurobi")
    print(f"[RMVP2 Gurobi] results -> {output_file}")

    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        t_values = ds_cfg.get("t_values", [None])
        combos = product(ds_cfg["rc_values"], ds_cfg["beta_values"],
                         ds_cfg["target_return_factors"], t_values)
        for r_c, beta, tr_factor, t_val in combos:
            print(f"[RMVP2 Gurobi] {ds_key} | r_c={r_c} beta={beta} tr={tr_factor} t={t_val}")
            for gamma_in in ds_cfg["gamma_inputs"]:
                pdat = generate_rmvp2_data(file_path=file_path, sheet_name=0, r_c=r_c,
                                           gamma=gamma_in, beta=beta,
                                           target_return_factor=tr_factor, t=t_val)
                if not pdat or pdat.get("D") is None:
                    continue
                D = pdat["D"]; tau = pdat["tau"]; tau_bar = pdat["tau_bar"]
                gamma = pdat["gamma"]; beta_scaled = pdat["beta"]; t = pdat["t"]
                n_full = pdat.get("n_full", pdat.get("n"))
                n_filtered = pdat.get("n_filtered", int(len(tau)))
                dropped_gamma = pdat.get("dropped_gamma", 0)
                dropped_t = pdat.get("dropped_t", 0)
                if n_filtered < 2:
                    print(f"    skip gamma={gamma}: {n_filtered} assets after filters")
                    continue

                obj, nnz, gap, wall, cpu = solve_gurobi_rmvp2(
                    D, tau, tau_bar, gamma, beta_scaled, t)

                append_row(output_file, {
                    "Dataset": file_path, "n": n_full, "n_filtered": n_filtered,
                    "gamma": gamma, "r_c": r_c, "beta": beta, "beta_scaled": beta_scaled,
                    "tr_factor": tr_factor, "t": t,
                    "dropped_assets": dropped_gamma + dropped_t,
                    "dropped_gamma": dropped_gamma, "dropped_t": dropped_t,
                    "obj_MIP_GUROBI": obj, "nnz_MIP_GUROBI": nnz,
                    "time_MIP_GUROBI": wall, "time_cpu_MIP_GUROBI": cpu,
                    "gap_MIP_GUROBI": gap,
                })
                print(f"    gamma={gamma} obj={obj:.6g} nnz={nnz} gap={gap} wall={wall:.2f}s")

    print(f"[RMVP2 Gurobi] done -> {output_file}")


if __name__ == "__main__":
    main()
