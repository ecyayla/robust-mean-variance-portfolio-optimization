"""
EuroBonds robust-vs-sparse out-of-sample study (self-contained).

Rolling-window backtest comparing three portfolios estimated on the SAME window:
  * Markowitz    - dense, closed-form (all assets, no sparsity, no robustness)
  * Sparse       - CP-RMVP1 with gamma=0,  beta>0  (BnB + warm-start drop)
  * SparseRobust - CP-RMVP1 with gamma>0,  beta>0  (BnB + warm-start drop)

For each rebalance we estimate (tau, D) from the in-sample window, solve the three
models, then evaluate realised out-of-sample excess returns on the next H days. We
report pooled / per-window excess Sharpe, a paired t-stat (Robust - Sparse), support
sizes, and solver time.

The claim (see eurobonds_robust_section.md): in a WIDE volatility-dispersion universe
the robust term reshapes the selection toward low-volatility assets and beats sparse
out of sample. This script reproduces the numbers behind that section.

Run:
    python3 eurobonds_experiment.py                      # quick default grid
    python3 eurobonds_experiment.py --window 60 --test 15 \
        --betas 5e-6 2e-6 1e-6 --gammas 0.0 0.05 0.10 0.15 --drop 0.5 --max-windows 0
"""
import os
# BLAS/OpenMP left at default (multi-thread) for speed on the full backtest.
import sys
import time
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from main import solveRMVP1, mainRMVP1BnB          # noqa: E402
from warm_start import warm_start_rmvp1            # noqa: E402

SCALE = 100.0            # returns scaled x100 (matches the paper pipeline)
RIDGE_COEF = 1e-6        # covariance ridge = ridge_coef * scale^2 * I (see --ridge)
ANNUAL = 252             # trading days / year
NNZ_TOL = 1e-8


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_returns(path):
    """Load a returns matrix (rows=days, cols=assets), no header, numeric only."""
    df = pd.read_excel(path, header=None)
    return df.values.astype(float)


def build_problem(R_window, r_c, tr_factor, ridge_coef=RIDGE_COEF):
    """From an in-sample window build (D, tau, tau_bar) exactly like the paper pipeline:
    returns scaled x100, D = sample covariance + ridge, tau = excess mean return.
    ridge = ridge_coef * scale^2 * I  (needed for invertibility when window <= n_assets;
    with window > n_assets the sample covariance is already full-rank and a tiny/zero
    ridge suffices)."""
    R = R_window * SCALE
    n = R.shape[1]
    r_hat = R.mean(axis=0)
    Rc = R - r_hat
    D = (Rc.T @ Rc) / (R.shape[0] - 1) + ridge_coef * SCALE**2 * np.eye(n)

    r_c_s = r_c * SCALE
    tau = r_hat - r_c_s
    tau_bar = r_c_s * (tr_factor - 1.0)     # bar_r - r_c, with bar_r = r_c * tr_factor
    return D, tau, tau_bar


# ---------------------------------------------------------------------------
# Solving one model -> full-length weight vector over the ORIGINAL universe
# ---------------------------------------------------------------------------
def solve_model(model, D, tau, tau_bar, gamma, beta, drop, timeout, restrict_idx=None):
    """Return {status, x_full, n_selected, support, wall, cpu}.
    x_full is length-n (original universe); unselected assets are 0.

    restrict_idx: if given, Sparse/SparseRobust are solved ONLY within this subset of
    the original universe (apples-to-apples: both models see the SAME assets). We pass
    the robust feasibility set here so gamma=0 (Sparse) and gamma>0 (Robust) differ only
    in the robust penalty, not in the asset menu. None = use the full universe."""
    n = D.shape[0]
    w0, c0 = time.time(), time.process_time()

    if model == "Markowitz":
        # dense closed-form: min x'Dx s.t. tau'x - 0 >= tau_bar, all assets.
        x, _, _ = solveRMVP1(D, tau_bar, tau, 0.0)
        x_full = np.asarray(x).reshape(-1)
        return {"status": "ok", "x_full": x_full,
                "n_selected": int(np.sum(np.abs(x_full) > NNZ_TOL)),
                "support": np.where(np.abs(x_full) > NNZ_TOL)[0],
                "wall": time.time() - w0, "cpu": time.process_time() - c0}

    # Sparse / SparseRobust ---------------------------------------------------
    # Optionally restrict to a shared universe, then apply this model's own
    # Assumption 1 filter |tau_i| > gamma*sqrt(D_ii) on top (a no-op if restrict_idx
    # already equals the gamma feasibility set).
    base = np.arange(n) if restrict_idx is None else np.asarray(restrict_idx, dtype=int)
    keep_local = np.abs(tau[base]) > gamma * np.sqrt(np.diag(D)[base])
    idx_keep = base[keep_local]
    if idx_keep.size < 2:
        return {"status": "too_few", "x_full": np.zeros(n), "n_selected": 0,
                "support": np.array([], dtype=int),
                "wall": time.time() - w0, "cpu": time.process_time() - c0}

    D_f = D[np.ix_(idx_keep, idx_keep)]
    tau_f = tau[idx_keep]

    # Warm-start elimination, then branch-and-bound on the reduced problem.
    warm = warm_start_rmvp1(D_f, tau_f, tau_bar, gamma, beta,
                            drop_fraction=drop, drop_farthest=True)
    D_r, tau_r, keep_r = warm["D_reduced"], warm["tau_reduced"], warm["keep_idx"]

    x_bnb, _, supp, _ = mainRMVP1BnB(D_r, tau_r, tau_bar, gamma, beta,
                                     time_limit=timeout)
    x_bnb = np.asarray(x_bnb).reshape(-1)
    supp = np.asarray(supp, dtype=int)

    # Map reduced support -> filtered -> original indices.
    orig_idx = idx_keep[keep_r[supp]]
    x_full = np.zeros(n)
    x_full[orig_idx] = x_bnb
    return {"status": "ok", "x_full": x_full,
            "n_selected": int(orig_idx.size), "support": orig_idx,
            "wall": time.time() - w0, "cpu": time.process_time() - c0}


# ---------------------------------------------------------------------------
# Out-of-sample evaluation
# ---------------------------------------------------------------------------
def oos_returns(R_test, x_full, r_c):
    """Realised out-of-sample daily excess-return series: (r_t - r_c) @ x."""
    return (R_test - r_c) @ x_full


def sharpe(rets):
    """Annualised excess Sharpe of a daily return series (nan if degenerate)."""
    rets = np.asarray(rets)
    if rets.size < 2:
        return np.nan
    sd = rets.std(ddof=1)
    if sd <= 0:
        return np.nan
    return np.sqrt(ANNUAL) * rets.mean() / sd


def paired_t(diffs):
    """Paired t-statistic of per-window Sharpe differences (Robust - Sparse)."""
    d = np.asarray([x for x in diffs if np.isfinite(x)])
    if d.size < 2:
        return np.nan
    se = d.std(ddof=1) / np.sqrt(d.size)
    return np.nan if se == 0 else d.mean() / se


# ---------------------------------------------------------------------------
# Rolling grid
# ---------------------------------------------------------------------------
def run_grid(R, args):
    """Rolling backtest over the (beta, gamma) grid. Returns one dict per cell."""
    W, H = args.window, args.test
    starts = list(range(W, R.shape[0] - H, H))
    if args.max_windows:
        starts = starts[:args.max_windows]
    print(f"[grid] {len(starts)} windows  W={W} H={H}  drop={args.drop}  "
          f"betas={args.betas}  gammas={args.gammas}")

    # accumulator per (beta, gamma)
    cells = {(b, g): _new_cell() for b in args.betas for g in args.gammas}

    for wi, t in enumerate(starts):
        Rin, Rtest = R[t - W:t], R[t:t + H]
        D, tau, tau_bar = build_problem(Rin, args.r_c, args.tr_factor, args.ridge)

        mk = solve_model("Markowitz", D, tau, tau_bar, 0.0, 0.0, args.drop, args.timeout)
        mk_rets = oos_returns(Rtest, mk["x_full"], args.r_c)

        diag_sqrt = np.sqrt(np.diag(D))
        for b in args.betas:
            for g in args.gammas:
                # Shared universe: the robust feasibility set for THIS gamma. Both
                # models solve on the same assets, so they differ only in the robust
                # penalty (g=0 vs g>0), not in the asset menu.
                if g == 0.0:
                    shared_idx = None                      # no filter -> full universe
                else:
                    shared_idx = np.where(np.abs(tau) > g * diag_sqrt)[0]

                sp = solve_model("Sparse", D, tau, tau_bar, 0.0, b,
                                 args.drop, args.timeout, restrict_idx=shared_idx)
                sp_rets = oos_returns(Rtest, sp["x_full"], args.r_c)
                sp_sh = sharpe(sp_rets)

                if g == 0.0:
                    ro, ro_rets, ro_sh = sp, sp_rets, sp_sh
                else:
                    ro = solve_model("SparseRobust", D, tau, tau_bar, g, b,
                                     args.drop, args.timeout, restrict_idx=shared_idx)
                    ro_rets = oos_returns(Rtest, ro["x_full"], args.r_c)
                    ro_sh = sharpe(ro_rets)

                c = cells[(b, g)]
                c["mk"].append(mk_rets)
                c["sp"].append(sp_rets)
                c["ro"].append(ro_rets)
                c["sp_sh"].append(sp_sh)
                c["ro_sh"].append(ro_sh)
                c["sp_supp"].append(sp["n_selected"])
                c["ro_supp"].append(ro["n_selected"])
                c["sp_wall"] += sp["wall"]
                c["ro_wall"] += ro["wall"]
                c["ro_cpu"] += ro["cpu"]
                # selection change + robust win-rate among changed windows
                if set(sp["support"].tolist()) != set(ro["support"].tolist()):
                    c["diff_sel"] += 1
                    if np.isfinite(ro_sh) and np.isfinite(sp_sh) and ro_sh > sp_sh:
                        c["ro_win"] += 1

        if (wi + 1) % 10 == 0 or wi == len(starts) - 1:
            print(f"  window {wi + 1}/{len(starts)} done")

    return [_finalize(b, g, cells[(b, g)]) for b in args.betas for g in args.gammas]


def _new_cell():
    return {"mk": [], "sp": [], "ro": [], "sp_sh": [], "ro_sh": [],
            "sp_supp": [], "ro_supp": [], "sp_wall": 0.0, "ro_wall": 0.0,
            "ro_cpu": 0.0, "diff_sel": 0, "ro_win": 0}


def _finalize(beta, gamma, c):
    diffs = [r - s for r, s in zip(c["ro_sh"], c["sp_sh"])]
    n_win = len(c["sp_sh"])
    return {
        "beta": beta, "gamma": gamma,
        "Sparse_supp": float(np.mean(c["sp_supp"])),
        "SparseRobust_supp": float(np.mean(c["ro_supp"])),
        "Markowitz_pooledSharpe": sharpe(np.concatenate(c["mk"])),
        "Sparse_pooledSharpe": sharpe(np.concatenate(c["sp"])),
        "SparseRobust_pooledSharpe": sharpe(np.concatenate(c["ro"])),
        "paired_meanDiff": float(np.nanmean(diffs)) if diffs else np.nan,
        "paired_t": paired_t(diffs),
        "diff_selection_pct": 100.0 * c["diff_sel"] / n_win if n_win else np.nan,
        "robust_win_rate_pct": (100.0 * c["ro_win"] / c["diff_sel"]
                                if c["diff_sel"] else np.nan),
        "Sparse_wall_s": c["sp_wall"],
        "SparseRobust_wall_s": c["ro_wall"],
        "SparseRobust_cpu_s": c["ro_cpu"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EuroBonds robust-vs-sparse OOS study")
    p.add_argument("--data", default=os.path.join(HERE, "datasets", "EuroBondsRet.xlsx"))
    p.add_argument("--window", type=int, default=60, help="in-sample window (days)")
    p.add_argument("--test", type=int, default=15, help="out-of-sample block (days)")
    p.add_argument("--r_c", type=float, default=2e-4, help="risk-free / cash return")
    p.add_argument("--tr_factor", type=float, default=1.05, help="target = tr*r_c")
    p.add_argument("--betas", type=float, nargs="+", default=[2e-6, 1e-6])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15])
    p.add_argument("--drop", type=float, default=0.5, help="warm-start drop fraction")
    p.add_argument("--ridge", type=float, default=RIDGE_COEF,
                   help="covariance ridge coefficient (ridge = coef*scale^2*I)")
    p.add_argument("--timeout", type=float, default=90.0, help="BnB time limit (s)")
    p.add_argument("--max-windows", type=int, default=25,
                   help="cap number of windows (0 = full backtest)")
    p.add_argument("--tag", default="", help="output filename tag; empty = no xlsx")
    return p.parse_args()


def main():
    args = parse_args()
    R = load_returns(args.data)
    vols = R.std(axis=0)
    print(f"EuroBonds: {R.shape[0]} days x {R.shape[1]} assets | "
          f"vol dispersion p90/p10 = {np.percentile(vols, 90) / np.percentile(vols, 10):.1f}x")

    rows = run_grid(R, args)
    df = pd.DataFrame(rows)
    cols = ["beta", "gamma", "Sparse_supp", "SparseRobust_supp",
            "Markowitz_pooledSharpe", "Sparse_pooledSharpe", "SparseRobust_pooledSharpe",
            "paired_meanDiff", "paired_t", "diff_selection_pct", "robust_win_rate_pct",
            "Sparse_wall_s", "SparseRobust_wall_s", "SparseRobust_cpu_s"]
    print("\n" + df[cols].round(3).to_string(index=False))

    if args.tag:
        out = os.path.join(HERE, f"eurobonds_exp_{args.tag}.xlsx")
        df.to_excel(out, index=False)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
