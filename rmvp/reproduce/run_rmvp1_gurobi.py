"""
RMVP1 -- Gurobi (MIP) runs, driven by experiment_config.json ("rmvp1").

For each enabled dataset and each (r_c, beta, tr_factor, gamma) combination:
  * build the problem (generate_rmvp1_data),
  * apply the assumption filter |tau_i| > gamma*sqrt(D_ii) so Gurobi solves the SAME
    assets as the BnB script,
  * solve once with Gurobi (single-thread), append one row to Excel immediately.

No warm-start and no drop-fraction loop here (Gurobi solves the full filtered problem).
Run standalone:  python3 run_rmvp1_gurobi.py
"""
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _common import (append_row, timestamped_output,
                     solve_gurobi_rmvp1, RMVP_DIR)                    # noqa: E402
if RMVP_DIR not in sys.path:
    sys.path.insert(0, RMVP_DIR)
from test import generate_rmvp1_data, load_experiment_config, get_dataset_files  # noqa: E402


def main():
    config = load_experiment_config()
    dataset_params = config.get("rmvp1", {}).get("datasets", {})
    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    output_file = timestamped_output("rmvp1_gurobi")
    print(f"[RMVP1 Gurobi] results -> {output_file}")

    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        combos = product(ds_cfg["rc_values"], ds_cfg["beta_values"],
                         ds_cfg["target_return_factors"])
        for r_c, beta, tr_factor in combos:
            print(f"[RMVP1 Gurobi] {ds_key} | r_c={r_c} beta={beta} tr={tr_factor}")
            for gamma_in in ds_cfg["gamma_inputs"]:
                pdat = generate_rmvp1_data(file_path=file_path, sheet_name=0, r_c=r_c,
                                           gamma=gamma_in, beta=beta,
                                           target_return_factor=tr_factor)
                # generate_rmvp1_data applies Assumption 1; D/tau are already filtered.
                if not pdat or pdat.get("D") is None:
                    continue
                D = pdat["D"]; tau = pdat["tau"]; tau_bar = pdat["tau_bar"]
                gamma = pdat["gamma"]; beta_scaled = pdat["beta"]
                n_full = pdat["n_full"]; n_filtered = pdat["n_filtered"]
                dropped_gamma = pdat["dropped_gamma"]
                if n_filtered < 2:
                    print(f"    skip gamma={gamma}: {n_filtered} assets survive filter")
                    continue

                obj, nnz, gap, wall, cpu = solve_gurobi_rmvp1(
                    D, tau, tau_bar, gamma, beta_scaled)

                append_row(output_file, {
                    "Dataset": file_path, "n": n_full, "n_filtered": n_filtered,
                    "gamma": gamma, "r_c": r_c, "beta": beta, "beta_scaled": beta_scaled,
                    "tr_factor": tr_factor, "t": None,
                    "dropped_assets": dropped_gamma, "dropped_gamma": dropped_gamma,
                    "dropped_t": 0,
                    "obj_MIP_GUROBI": obj, "nnz_MIP_GUROBI": nnz,
                    "time_MIP_GUROBI": wall, "time_cpu_MIP_GUROBI": cpu,
                    "gap_MIP_GUROBI": gap,
                })
                print(f"    gamma={gamma} obj={obj:.6g} nnz={nnz} gap={gap} wall={wall:.2f}s")

    print(f"[RMVP1 Gurobi] done -> {output_file}")


if __name__ == "__main__":
    main()
