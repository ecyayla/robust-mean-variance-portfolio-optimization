"""
RMVP1 -- Branch-and-Bound runs, driven by experiment_config.json ("rmvp1").

For each enabled dataset and each (r_c, beta, tr_factor, gamma) combination:
  * build the problem (generate_rmvp1_data),
  * apply the assumption filter |tau_i| > gamma*sqrt(D_ii) (align with RMVP2),
  * for each configured drop fraction: warm-start eliminate, then solve with BnB,
  * append one row to the Excel immediately (crash-safe).

BnB keeps numpy/BLAS at default threads (we report wall time). Run standalone:
    python3 run_rmvp1_bnb.py
"""
import os
import sys
import time
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _common import (append_row, timestamped_output, resolve_drop_fractions,
                     solve_bnb_rmvp1, RMVP_DIR)  # noqa: E402
if RMVP_DIR not in sys.path:
    sys.path.insert(0, RMVP_DIR)
from test import generate_rmvp1_data, load_experiment_config, get_dataset_files  # noqa: E402
from warm_start import warm_start_rmvp1                                          # noqa: E402


def main(drop_farthest=True):
    config = load_experiment_config()
    dataset_params = config.get("rmvp1", {}).get("datasets", {})
    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    output_file = timestamped_output("rmvp1_bnb")
    print(f"[RMVP1 BnB] results -> {output_file}")

    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        drop_map = ds_cfg.get("drop_fractions_by_beta", {})
        combos = product(ds_cfg["rc_values"], ds_cfg["beta_values"],
                         ds_cfg["target_return_factors"])

        for r_c, beta, tr_factor in combos:
            print(f"[RMVP1 BnB] {ds_key} | r_c={r_c} beta={beta} tr={tr_factor}")
            for gamma_in in ds_cfg["gamma_inputs"]:
                pdat = generate_rmvp1_data(file_path=file_path, sheet_name=0, r_c=r_c,
                                           gamma=gamma_in, beta=beta,
                                           target_return_factor=tr_factor)
                # generate_rmvp1_data now applies Assumption 1 (|tau_i|>gamma*sqrt(D_ii));
                # D/tau come back already filtered, with bookkeeping fields.
                if not pdat or pdat.get("D") is None:
                    continue
                D = pdat["D"]; tau = pdat["tau"]; tau_bar = pdat["tau_bar"]
                gamma = pdat["gamma"]; beta_scaled = pdat["beta"]
                n_full = pdat["n_full"]; n_filtered = pdat["n_filtered"]
                dropped_gamma = pdat["dropped_gamma"]
                if n_filtered < 2:
                    print(f"    skip gamma={gamma}: {n_filtered} assets survive filter")
                    continue

                drops = resolve_drop_fractions(drop_map, beta)
                if not drops:
                    print(f"    no drop fractions configured for beta={beta}")
                    continue

                for drop_fraction in drops:
                    c0 = time.process_time()
                    warm = warm_start_rmvp1(D, tau, tau_bar, gamma, beta_scaled,
                                            drop_fraction=drop_fraction,
                                            drop_farthest=drop_farthest)
                    ws_cpu = time.process_time() - c0
                    D_red = warm["D_reduced"]; tau_red = warm["tau_reduced"]
                    n_ws = int(len(tau_red))

                    obj, nnz, nodes, wall, cpu, collapse_S, collapse_P = solve_bnb_rmvp1(
                        D_red, tau_red, tau_bar, gamma, beta_scaled)

                    append_row(output_file, {
                        "Dataset": file_path, "n": n_full, "n_filtered": n_filtered,
                        "gamma": gamma, "r_c": r_c, "beta": beta, "beta_scaled": beta_scaled,
                        "tr_factor": tr_factor, "t": None,
                        "dropped_assets": dropped_gamma, "dropped_gamma": dropped_gamma,
                        "dropped_t": 0, "drop_fraction": drop_fraction,
                        "drop_farthest": drop_farthest, "n_warm_start": n_ws,
                        "warm_start_dropped_assets": n_filtered - n_ws,
                        "time_cpu_warm_start": ws_cpu,
                        "obj_BnB_warm": obj, "nnz_BnB_warm": nnz, "bnb_nodes": nodes,
                        "time_BnB_warm": wall, "time_cpu_BnB_warm": cpu,
                        "time_cpu_total": ws_cpu + cpu,
                        "collapse_S_sizes": collapse_S, "collapse_P_sizes": collapse_P,
                    })
                    print(f"    gamma={gamma} drop={drop_fraction} n_ws={n_ws} "
                          f"obj={obj:.6g} nnz={nnz} nodes={nodes} wall={wall:.2f}s")

    print(f"[RMVP1 BnB] done -> {output_file}")


if __name__ == "__main__":
    main()
