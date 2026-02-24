"""
Test script to compare the improved BnB implementation with the current one.
Follows the exact same structure as test.py, testing all parameter combinations.
"""

import numpy as np
import pandas as pd
import time
import glob
import os
from itertools import product
from datetime import datetime
from main import mainRMVP1BnB, zeropadding
from bnb_improved import mainRMVP1BnB_improved
from test import generate_rmvp1_data, get_dataset_files, load_experiment_config


def run_bnb_comparison_experiments():
    """
    Run comparison experiments between current and improved BnB implementations.
    Follows the exact same structure as run_experiments_rmvp1() in test.py.
    """
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
    output_file = f"bnb_comparison_results_{timestamp}.xlsx"
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

                # Initialize result variables
                obj_current_bnb = "ALL_DROPPED"
                obj_improved_bnb = "ALL_DROPPED"
                time_current_bnb = time_cpu_current_bnb = 0
                time_improved_bnb = time_cpu_improved_bnb = 0
                nodes_current_bnb = "ALL_DROPPED"
                nodes_improved_bnb = "ALL_DROPPED"
                nnz_current_bnb = "ALL_DROPPED"
                nnz_improved_bnb = "ALL_DROPPED"
                obj_diff = "ALL_DROPPED"
                obj_rel_diff_pct = "ALL_DROPPED"
                node_reduction_pct = "ALL_DROPPED"
                speedup = "ALL_DROPPED"

                # Check if all assets are dropped
                if num_removed == n or num_removed == n-1:
                    print(f"    Skipping solver: All {n} assets dropped.")
                else:
                    # --- Current BnB Solver ---
                    print("    Running current BnB...")
                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_current, obj_current, supp_current, count_current = mainRMVP1BnB(
                        D, tau, tau_bar, gamma, beta_scaled, traverse_rule='bfs'
                    )
                    time_current_bnb = time.time() - start_time
                    time_cpu_current_bnb = time.process_time() - start_cpu
                    x_current = zeropadding(x_current, supp_current, n)
                    obj_current_bnb = float(x_current.T @ D @ x_current + beta_scaled * len(supp_current))
                    nodes_current_bnb = count_current
                    nnz_current_bnb = int(len(supp_current))
                    print(f"      Current BnB completed: obj={obj_current_bnb:.8f}, nodes={count_current}, time={time_current_bnb:.4f}s")

                    # --- Improved BnB Solver ---
                    print("    Running improved BnB...")
                    start_time = time.time()
                    start_cpu = time.process_time()
                    x_improved, obj_improved, supp_improved, count_improved = mainRMVP1BnB_improved(
                        D, tau, tau_bar, gamma, beta_scaled, traverse_rule='bfs'
                    )
                    time_improved_bnb = time.time() - start_time
                    time_cpu_improved_bnb = time.process_time() - start_cpu
                    
                    # Pad solution to full dimension (same as current BnB)
                    x_improved = zeropadding(x_improved, supp_improved, n)
                    
                    obj_improved_bnb = float(x_improved.T @ D @ x_improved + beta_scaled * len(supp_improved))
                    nodes_improved_bnb = count_improved
                    nnz_improved_bnb = int(len(supp_improved))
                    print(f"      Improved BnB completed: obj={obj_improved_bnb:.8f}, nodes={count_improved}, time={time_improved_bnb:.4f}s")

                    # Compute comparison metrics
                    if obj_current_bnb != "ALL_DROPPED" and obj_improved_bnb != "ALL_DROPPED":
                        obj_diff = abs(obj_current_bnb - obj_improved_bnb)
                        obj_rel_diff_pct = (obj_diff / max(abs(obj_current_bnb), abs(obj_improved_bnb), 1e-10)) * 100
                        
                        if nodes_current_bnb > 0:
                            node_reduction_pct = ((nodes_current_bnb - nodes_improved_bnb) / nodes_current_bnb) * 100
                        else:
                            node_reduction_pct = 0.0
                        
                        if time_improved_bnb > 0:
                            speedup = time_current_bnb / time_improved_bnb
                        else:
                            speedup = 0.0

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
                    "obj_Current_BnB": obj_current_bnb,
                    "time_Current_BnB": time_current_bnb,
                    "time_cpu_Current_BnB": time_cpu_current_bnb,
                    "nodes_Current_BnB": nodes_current_bnb,
                    "nnz_Current_BnB": nnz_current_bnb,
                    "obj_Improved_BnB": obj_improved_bnb,
                    "time_Improved_BnB": time_improved_bnb,
                    "time_cpu_Improved_BnB": time_cpu_improved_bnb,
                    "nodes_Improved_BnB": nodes_improved_bnb,
                    "nnz_Improved_BnB": nnz_improved_bnb,
                    "obj_diff": obj_diff,
                    "obj_rel_diff_pct": obj_rel_diff_pct,
                    "node_reduction_pct": node_reduction_pct,
                    "speedup": speedup
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

    print(f"\nComparison experiments completed. Results saved to {output_file}")
    
    # Print summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        successful_comparisons = [r for r in results if r.get("obj_Current_BnB") != "ALL_DROPPED" and r.get("obj_Improved_BnB") != "ALL_DROPPED"]
        
        if successful_comparisons:
            print(f"Successful comparisons: {len(successful_comparisons)} / {len(results)}")
            
            obj_diffs = [r["obj_diff"] for r in successful_comparisons if r["obj_diff"] != "ALL_DROPPED"]
            if obj_diffs:
                print(f"Objective differences: min={min(obj_diffs):.2e}, max={max(obj_diffs):.2e}, mean={np.mean(obj_diffs):.2e}")
            
            node_reductions = [r["node_reduction_pct"] for r in successful_comparisons if r["node_reduction_pct"] != "ALL_DROPPED"]
            if node_reductions:
                print(f"Node reduction: min={min(node_reductions):.2f}%, max={max(node_reductions):.2f}%, mean={np.mean(node_reductions):.2f}%")
            
            speedups = [r["speedup"] for r in successful_comparisons if r["speedup"] != "ALL_DROPPED" and r["speedup"] > 0]
            if speedups:
                print(f"Speedup: min={min(speedups):.2f}x, max={max(speedups):.2f}x, mean={np.mean(speedups):.2f}x")


if __name__ == "__main__":
    run_bnb_comparison_experiments()
