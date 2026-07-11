"""
Shared helpers for the config-driven BnB / Gurobi run scripts (2 models x 2 methods).

All four run_*.py scripts import from here. This module:
  * bootstraps sys.path so main / test / warm_start (in the parent rmvp/ dir) import,
  * provides a crash-safe incremental Excel writer (one row per completed run),
  * fixes the drop-fraction lookup (match config keys by float, not str(beta)),
  * applies the RMVP1 assumption filter |tau_i| > gamma*sqrt(D_ii) (RMVP2 already
    filters inside generate_rmvp2_data),
  * wraps the four solvers with wall- AND cpu-time measurement.

Timing note: BnB keeps numpy/BLAS at default threads (we report wall time); Gurobi
runs single-threaded (RMVP*_mipGUROBI default Threads=1).
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# --- make the parent rmvp/ package importable ------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RMVP_DIR = os.path.dirname(HERE)
if RMVP_DIR not in sys.path:
    sys.path.insert(0, RMVP_DIR)

# --- Gurobi license -------------------------------------------------------
# The project-root gurobi.lic points at the working token server
# (139.179.39.137:41954). Because these scripts run from config_runs/, Gurobi's
# default cwd search would otherwise miss it and fall through to the expired
# /opt/gurobi/gurobi.lic. Point GRB_LICENSE_FILE at the project-root file
# explicitly (unless the user already set one). Harmless for the BnB scripts.
_proj_lic = os.path.join(os.path.dirname(RMVP_DIR), "gurobi.lic")
if os.path.exists(_proj_lic) and not os.environ.get("GRB_LICENSE_FILE"):
    os.environ["GRB_LICENSE_FILE"] = _proj_lic

from main import (mainRMVP1BnB, mainRMVP2BnB, RMVP1_mipGUROBI, RMVP2_mipGUROBI,
                  zeropadding)                                    # noqa: E402

NNZ_TOL = 1e-8


# ---------------------------------------------------------------------------
# Incremental, crash-safe Excel writer (one row per completed run)
# ---------------------------------------------------------------------------
def append_row(output_file, row):
    """Append a single result dict as one row to output_file, immediately.
    First write creates the header; later writes append without header. If the
    process dies mid-run, every previously written row is already on disk."""
    df_row = pd.DataFrame([row])
    try:
        existing = pd.read_excel(output_file)
        start_row = len(existing) + 1
        with pd.ExcelWriter(output_file, mode="a", if_sheet_exists="overlay",
                            engine="openpyxl") as writer:
            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
    except FileNotFoundError:
        df_row.to_excel(output_file, index=False)


def timestamped_output(prefix):
    """Path in this folder like <prefix>_YYYY-MM-DD_HH-MM.xlsx."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return os.path.join(HERE, f"{prefix}_{ts}.xlsx")


# ---------------------------------------------------------------------------
# drop_fractions_by_beta lookup -- match config keys by FLOAT value.
# str(0.00005) == '5e-05' would NOT match a config key '0.00005'; the paper's
# test_warmstart.py uses that str() lookup and thus silently skips such betas.
# ---------------------------------------------------------------------------
def resolve_drop_fractions(drop_map, beta):
    if not drop_map:
        return []
    for k, v in drop_map.items():
        try:
            if abs(float(k) - float(beta)) < 1e-15:
                return list(v)
        except (TypeError, ValueError):
            continue
    return []


# ---------------------------------------------------------------------------
# Timed solvers -- each returns wall AND cpu time.
# (Assumption 1 filtering now lives inside generate_rmvp1_data; RMVP2 filters
#  inside generate_rmvp2_data -- so no separate filter helper is needed here.)
# ---------------------------------------------------------------------------
def solve_bnb_rmvp1(D, tau, tau_bar, gamma, beta):
    n = D.shape[0]
    w0, c0 = time.time(), time.process_time()
    x_bnb, _, supp_bnb, nodes, cS, cP = mainRMVP1BnB(D, tau, tau_bar, gamma, beta,
                                                     collect_collapse=True)
    wall, cpu = time.time() - w0, time.process_time() - c0
    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    obj = float((x_bnb.T @ D @ x_bnb + beta * np.sum(np.abs(x_bnb) > NNZ_TOL))[0][0])
    nnz = int(np.sum(np.abs(x_bnb) > NNZ_TOL))
    return (obj, nnz, int(nodes), wall, cpu,
            ",".join(map(str, cS)), ",".join(map(str, cP)))


def solve_bnb_rmvp2(D, tau, tau_bar, gamma, beta, t):
    n = D.shape[0]
    w0, c0 = time.time(), time.process_time()
    x_bnb, _, supp_bnb, nodes, cS, cP = mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t,
                                                     collect_collapse=True)
    wall, cpu = time.time() - w0, time.process_time() - c0
    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    quad = float((x_bnb.T @ D @ x_bnb)[0][0])
    norm_D = np.sqrt(quad)
    tauTx = float((tau.T @ x_bnb)[0])
    obj = -tauTx + tau_bar + gamma * norm_D + beta * int(np.sum(np.abs(x_bnb) > NNZ_TOL))
    nnz = int(np.sum(np.abs(x_bnb) > NNZ_TOL))
    return (float(obj), nnz, int(nodes), wall, cpu,
            ",".join(map(str, cS)), ",".join(map(str, cP)))


def solve_gurobi_rmvp1(D, tau, tau_bar, gamma, beta):
    w0, c0 = time.time(), time.process_time()
    x_g, _, gap = RMVP1_mipGUROBI(D, tau, tau_bar, gamma, beta)   # Threads=1 by default
    wall, cpu = time.time() - w0, time.process_time() - c0
    obj = float((x_g.T @ D @ x_g + beta * np.sum(np.abs(x_g) > NNZ_TOL))[0][0])
    nnz = int(np.sum(np.abs(x_g) > NNZ_TOL))
    return obj, nnz, gap, wall, cpu


def solve_gurobi_rmvp2(D, tau, tau_bar, gamma, beta, t):
    w0, c0 = time.time(), time.process_time()
    x_g, _, gap = RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta, t)  # Threads=1 by default
    wall, cpu = time.time() - w0, time.process_time() - c0
    quad = float((x_g.T @ D @ x_g)[0][0])
    norm_D = np.sqrt(quad)
    tauTx = float((tau.T @ x_g)[0])
    obj = -tauTx + tau_bar + gamma * norm_D + beta * int(np.sum(np.abs(x_g) > NNZ_TOL))
    nnz = int(np.sum(np.abs(x_g) > NNZ_TOL))
    return float(obj), nnz, gap, wall, cpu
