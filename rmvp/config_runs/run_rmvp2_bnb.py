"""
RMVP2 -- Branch-and-Bound runs, driven by experiment_config.json ("rmvp2").

Same structure as run_rmvp1_bnb.py but for model 2: the product also ranges over
t_values, and generate_rmvp2_data ALREADY applies the assumption + t filters (so no
extra filter call here). For each configured drop fraction: warm-start then BnB, with
one Excel row appended per completed run.

Run standalone:  python3 run_rmvp2_bnb.py
"""
import os
import sys
import time
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _common import (append_row, timestamped_output, resolve_drop_fractions,
                     solve_bnb_rmvp2, RMVP_DIR)                      
if RMVP_DIR not in sys.path:
    sys.path.insert(0, RMVP_DIR)
from test import generate_rmvp2_data, load_experiment_config, get_dataset_files  
from warm_start import warm_start_rmvp2                                          


def main(drop_farthest=True):
    config = load_experiment_config()
    dataset_params = config.get("rmvp2", {}).get("datasets", {})
    data_files = get_dataset_files()
    if not data_files:
        print("No datasets found.")
        return

    output_file = timestamped_output("rmvp2_bnb")
    print(f"[RMVP2 BnB] results -> {output_file}")

    for file_path in data_files:
        ds_key = os.path.basename(file_path)
        ds_cfg = dataset_params.get(ds_key)
        if ds_cfg is None or not ds_cfg.get("enabled", True):
            continue

        drop_map = ds_cfg.get("drop_fractions_by_beta", {})
        t_values = ds_cfg.get("t_values", [None])
        combos = product(ds_cfg["rc_values"], ds_cfg["beta_values"],
                         ds_cfg["target_return_factors"], t_values)

        for r_c, beta, tr_factor, t_val in combos:
            print(f"[RMVP2 BnB] {ds_key} | r_c={r_c} beta={beta} tr={tr_factor} t={t_val}")
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

                drops = resolve_drop_fractions(drop_map, beta)
                if not drops:
                    print(f"    no drop fractions configured for beta={beta}")
                    continue

                for drop_fraction in drops:
                    c0 = time.process_time()
                    warm = warm_start_rmvp2(D, tau, tau_bar, gamma, beta_scaled, t,
                                            drop_fraction=drop_fraction,
                                            drop_farthest=drop_farthest)
                    ws_cpu = time.process_time() - c0
                    D_red = warm["D_reduced"]; tau_red = warm["tau_reduced"]
                    n_ws = int(len(tau_red))

                    obj, nnz, nodes, wall, cpu, collapse_S, collapse_P = solve_bnb_rmvp2(
                        D_red, tau_red, tau_bar, gamma, beta_scaled, t)

                    append_row(output_file, {
                        "Dataset": file_path, "n": n_full, "n_filtered": n_filtered,
                        "gamma": gamma, "r_c": r_c, "beta": beta, "beta_scaled": beta_scaled,
                        "tr_factor": tr_factor, "t": t,
                        "dropped_assets": dropped_gamma + dropped_t,
                        "dropped_gamma": dropped_gamma, "dropped_t": dropped_t,
                        "drop_fraction": drop_fraction, "drop_farthest": drop_farthest,
                        "n_warm_start": n_ws, "warm_start_dropped_assets": n_filtered - n_ws,
                        "time_cpu_warm_start": ws_cpu,
                        "obj_BnB_warm": obj, "nnz_BnB_warm": nnz, "bnb_nodes": nodes,
                        "time_BnB_warm": wall, "time_cpu_BnB_warm": cpu,
                        "time_cpu_total": ws_cpu + cpu,
                        "collapse_S_sizes": collapse_S, "collapse_P_sizes": collapse_P,
                    })
                    print(f"    gamma={gamma} drop={drop_fraction} n_ws={n_ws} "
                          f"obj={obj:.6g} nnz={nnz} nodes={nodes} wall={wall:.2f}s")

    print(f"[RMVP2 BnB] done -> {output_file}")


if __name__ == "__main__":
    main()
