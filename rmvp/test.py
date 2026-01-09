import numpy as np
import pandas as pd
import time
import glob
from itertools import product
from datetime import datetime
from main import mainRMVP1BnB, RMVP1_mipCVXPY, RMVP1_mipGUROBI, RMVP1_mipMOSEK, mainRMVP2BnB, RMVP2_mipCVXPY, RMVP2_mipGUROBI, zeropadding

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
    ratios = np.abs(tau) / np.maximum(D_diag_sqrt, 1e-12)
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
    
    # Estimate t (D-norm constraint bound)
    # H = sqrt(tau^T D^-1 tau)
    try:
        D_inv = np.linalg.inv(D)
        H = float(np.sqrt(tau.T @ D_inv @ tau))
        
        # Lower bound on t for non-trivial solutions
        D_diag_sqrt = np.sqrt(np.diag(D))
        denominators = (tau / D_diag_sqrt) - gamma
        denominators = np.maximum(denominators, 1e-8)
        min_t_bound = np.min(beta / denominators)
        
        # Heuristic for t: larger than H and the bound
        if t is None:
             t = max(1.1 * H, 1.1 * min_t_bound)
        
    except Exception as e:
        print(f"Error calculating t: {e}")
        t = 1.0 # Fallback
        
    base_data['t'] = t
    return base_data


def run_experiments_rmvp1():
    results = []
    # Search in datasets/ (relative to script) or rmvp/datasets/ or ../datasets
    # Using relative paths assuming script execution from root or rmvp folder
    data_files = glob.glob("rmvp/datasets/*.xlsx") + glob.glob("datasets/*.xlsx")
    
    # Remove duplicates and sort for deterministic order
    data_files = sorted(set(data_files))
    
    if not data_files:
        print("No datasets found.")
        return

    # Timestamp for the output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp1_experiment_results_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    # Parameter ranges for experimentation
    rc_values = [0.0002, 0.0004]
    beta_values = [1e-4, 1e-5, 5e-6, 1e-6]
    target_return_factors = [1.05, 1.1, 1.15]
    
    # 0.0 triggers the dynamic calculation in generate_rmvp1_data
    gamma_inputs = [0.001, 0.01,0.1, 0.0]

    # Generate combinations of parameters excluding gamma
    param_combinations = product(data_files, rc_values, beta_values, target_return_factors)

    for file_path, r_c, beta, tr_factor in param_combinations:
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
            tau_minus_gamma_mip_cvx = "ALL_DROPPED"
            tau_minus_gamma_mip_gurobi = "ALL_DROPPED"
            bnb_time = bnb_time_cpu = 0
            mip_time_cvx = mip_time_cvx_cpu = 0
            mip_time_gurobi = mip_time_gurobi_cpu = 0

            # Check if all assets are dropped
            if num_removed == n or num_removed == n-1:
                print(f"    Skipping solver: All {n} assets dropped.")
                obj_bnb = "ALL_DROPPED"
                obj_mip_cvx = "ALL_DROPPED"
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
                print("bnb completed")

                # --- MIP CVXPY Solver ---
                start_time = time.time()
                start_cpu = time.process_time()
                x_mip, _ = RMVP1_mipCVXPY(D, tau, tau_bar, gamma, beta_scaled)
                mip_time_cvx = time.time() - start_time
                mip_time_cvx_cpu = time.process_time() - start_cpu
                obj_mip_cvx = (x_mip.T @ D @ x_mip + beta_scaled * np.sum(np.abs(x_mip) > 1e-8))[0][0]
                tauTx_mip = (tau.T @ x_mip)[0]
                normD_mip = np.sqrt(x_mip.T @ D @ x_mip)[0]
                tau_minus_gamma_mip_cvx = tauTx_mip - gamma * normD_mip
                print("mip cvxpy completed")

                # --- MIP GUROBI Solver ---
                start_time = time.time()
                start_cpu = time.process_time()
                x_mip_gurobi, _ = RMVP1_mipGUROBI(D, tau, tau_bar, gamma, beta_scaled)
                mip_time_gurobi = time.time() - start_time
                mip_time_gurobi_cpu = time.process_time() - start_cpu
                obj_mip_gurobi = (x_mip_gurobi.T @ D @ x_mip_gurobi + beta_scaled * np.sum(np.abs(x_mip_gurobi) > 1e-8))[0][0]
                tauTx_mip_gurobi = (tau.T @ x_mip_gurobi)[0]
                normD_mip_gurobi = np.sqrt(x_mip_gurobi.T @ D @ x_mip_gurobi)[0]
                tau_minus_gamma_mip_gurobi = tauTx_mip_gurobi - gamma * normD_mip_gurobi
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
                "obj_MIP_CVXPY": obj_mip_cvx,
                "time_MIP_CVXPY": mip_time_cvx,
                "time_cpu_MIP_CVXPY": mip_time_cvx_cpu,
                "obj_MIP_GUROBI": obj_mip_gurobi,
                "time_MIP_GUROBI": mip_time_gurobi,
                "time_cpu_MIP_GUROBI": mip_time_gurobi_cpu,
                "tauTx_minus_gammaNormD_BnB": tau_minus_gamma_bnb if num_removed != n else "ALL_DROPPED",
                "tauTx_minus_gammaNormD_MIP_CVXPY": tau_minus_gamma_mip_cvx if num_removed != n else "ALL_DROPPED",
                "tauTx_minus_gammaNormD_MIP_GUROBI": tau_minus_gamma_mip_gurobi if num_removed != n else "ALL_DROPPED"
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
    # Search in datasets/ (relative to script) or rmvp/datasets/ or ../datasets
    # Using relative paths assuming script execution from root or rmvp folder
    data_files = glob.glob("rmvp/datasets/*.xlsx") + glob.glob("datasets/*.xlsx")
    
    # Remove duplicates if any
    data_files = list(set(data_files))
    
    if not data_files:
        print("No datasets found.")
        return

    # Timestamp for the output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"rmvp2_experiment_results_{timestamp}.xlsx"
    print(f"Results will be saved to: {output_file}")

    # Parameter ranges for experimentation
    rc_values = [0.0, 0.0001, 0.0005]
    beta_values = [1e-5, 1e-4, 1e-3]
    target_return_factors = [1.05, 1.1]
    
    # 0.0 triggers the dynamic calculation in generate_rmvp1_data (called by generate_rmvp2_data)
    gamma_inputs = [0.001, 0.01, 0.0]

    # Generate combinations of parameters excluding gamma
    param_combinations = product(data_files, rc_values, beta_values, target_return_factors)

    for file_path, r_c, beta, tr_factor in param_combinations:
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

            D = problem_data['D']
            tau = problem_data['tau']
            tau_bar = problem_data['tau_bar']
            gamma = problem_data['gamma'] # The actual used gamma
            beta_scaled = problem_data['beta']
            t = problem_data['t']
            n = problem_data['n']
            
            # Count assets violating assumption
            D_diag_sqrt = np.sqrt(np.diag(D))
            feasibility_margin = np.abs(tau) - (gamma * D_diag_sqrt)
            num_removed = np.sum(feasibility_margin <= 0)
            
            print(f"    Running gamma={gamma:.6f} | t={t:.4f} (Dropped: {num_removed})")

            # Check if all assets are dropped
            if num_removed == n or num_removed == n-1:
                print(f"    Skipping solver: All {n} assets dropped.")
                obj_bnb, bnb_time = "ALL_DROPPED", 0
                obj_mip_cvx, mip_time_cvx = "ALL_DROPPED", 0
                obj_mip_gurobi, mip_time_gurobi = "ALL_DROPPED", 0
            else:
                # --- BnB Solver ---
                start_time = time.time()
                x_bnb, _, supp_bnb, _ = mainRMVP2BnB(D, tau, tau_bar, gamma, beta_scaled, t)
                bnb_time = time.time() - start_time
                x_bnb = zeropadding(x_bnb, supp_bnb, n)
                # Recalculate objective: -tau^T x + tau_bar + gamma * ||x||_D + beta * ||x||_0
                norm_D_bnb = np.sqrt(x_bnb.T @ D @ x_bnb)[0,0]
                obj_bnb = -(tau.T @ x_bnb)[0,0] + tau_bar + gamma * norm_D_bnb + beta_scaled * np.sum(np.abs(x_bnb) > 1e-8)
                print("bnb completed")

                # --- MIP CVXPY Solver ---
                start_time = time.time()
                x_mip_cvx, _ = RMVP2_mipCVXPY(D, tau, tau_bar, gamma, beta_scaled, t)
                mip_time_cvx = time.time() - start_time
                norm_D_cvx = np.sqrt(x_mip_cvx.T @ D @ x_mip_cvx)[0,0]
                obj_mip_cvx = -(tau.T @ x_mip_cvx)[0,0] + tau_bar + gamma * norm_D_cvx + beta_scaled * np.sum(np.abs(x_mip_cvx) > 1e-8)
                print("mip cvxpy completed")

                # --- MIP GUROBI Solver ---
                start_time = time.time()
                x_mip_gurobi, _ = RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta_scaled, t)
                mip_time_gurobi = time.time() - start_time
                norm_D_gurobi = np.sqrt(x_mip_gurobi.T @ D @ x_mip_gurobi)[0,0]
                obj_mip_gurobi = -(tau.T @ x_mip_gurobi)[0,0] + tau_bar + gamma * norm_D_gurobi + beta_scaled * np.sum(np.abs(x_mip_gurobi) > 1e-8)
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
                "t": t,
                "dropped_assets": num_removed,
                "tau_bar": tau_bar,
                "obj_BnB": obj_bnb,
                "time_BnB": bnb_time,
                "obj_MIP_CVXPY": obj_mip_cvx,
                "time_MIP_CVXPY": mip_time_cvx,
                "obj_MIP_GUROBI": obj_mip_gurobi,
                "time_MIP_GUROBI": mip_time_gurobi
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
    #run_experiments_rmvp2()



    """
    problem_data = generate_rmvp1_data(file_path="rmvp/datasets/DowJones_returns.xlsx", sheet_name=0, r_c=2e-4, gamma=0, beta=1e-4, target_return_factor=1.05, scale_returns=100)
    D = problem_data['D']
    tau = problem_data['tau']
    tau_bar = problem_data['tau_bar']
    gamma = problem_data['gamma']
    beta = problem_data['beta']
    beta = 5e-6
    print("tau_bar: ", tau_bar)

    #x_mip1, obj_mip1 = RMVP1_mipCVXPY(D, tau, tau_bar, gamma, beta)
    #print("x_mip1: ", x_mip1)
    #print("mip cvxpy completed")
    #print("obj_mip1: ", (x_mip1.T @ D @ x_mip1 + beta * np.sum(np.abs(x_mip1) > 1e-9))[0][0])
    #print("const: ", tau.T @ x_mip1 - gamma * np.sqrt(x_mip1.T @ D @ x_mip1))
    x_mip2, obj_mip2 = RMVP1_mipGUROBI(D, tau, tau_bar, gamma, beta)
    print("mip gurobi completed")
    print("x_mip2: ", x_mip2)
    print("obj_mip2: ", (x_mip2.T @ D @ x_mip2 + beta * np.sum(np.abs(x_mip2) > 1e-9))[0][0])
    print("const: ", tau.T @ x_mip2 - gamma * np.sqrt(x_mip2.T @ D @ x_mip2))
    #x_mip3, obj_mip3 = RMVP1_mipMOSEK(D, tau, tau_bar, gamma, beta)
    #print("mip mosek completed")
    #print("obj_mip3: ", (x_mip3.T @ D @ x_mip3 + beta * np.sum(np.abs(x_mip3) > 1e-9))[0][0])
    #print("const: ", tau.T @ x_mip3 - gamma * np.sqrt(x_mip3.T @ D @ x_mip3))
    x_bnb1, obj_bnb1, supp_bnb1, count_bnb1 = mainRMVP1BnB(D, tau, tau_bar, gamma, beta)
    print("bnb completed")
    x_bnb1 = zeropadding(x_bnb1, supp_bnb1, D.shape[0])
    print("x_bnb1: ", x_bnb1)
    print("obj_bnb1: ", (x_bnb1.T @ D @ x_bnb1 + beta * np.sum(np.abs(x_bnb1) > 1e-9))[0][0])
    print("const: ", tau.T @ x_bnb1 - gamma * np.sqrt(x_bnb1.T @ D @ x_bnb1))

    """
    