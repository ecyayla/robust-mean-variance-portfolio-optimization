#!/usr/bin/env python3
"""
rvs.py — the single robust-vs-sparse OOS experiment runner.

ONE script for every experiment. Sweep any of r_c / beta / gamma (each a list) on any
dataset, with any window / holding / drop / ridge / target-return. It solves, per rolling
window, the Sparse (gamma=0) and Robust (gamma>0) CP-RMVP portfolios on the SAME
gamma-feasible asset set (apples-to-apples), runs the OOS backtest, and reports for each
(r_c, gamma, beta) cell:

  suppS/suppR  average support (number of held assets), Sparse / Robust
  ShrpS/ShrpR  annualized (sqrt252) pooled OOS Sharpe, Sparse / Robust
  retS/retR%   annualized mean OOS excess return (%)
  netS/netR    average net exposure 1'x  (negative = net short)
  tFull/pFull  one-sided paired t & p on OOS excess Sharpe difference (H1: robust>sparse)
  ndiff        # windows where the two models pick a different support
  rWin%        % of those windows where robust's realized OOS return beats sparse's

Rows stream to --out (append + flush) and stdout as each cell finishes, so partial results
survive interruption. A trailing 'DONE' marks completion.

drop is the warm-start elimination fraction. drop=0 is EXACT branch-and-bound (recommended
for final numbers). drop>0 drops the "farthest" assets before the BnB — it is a HEURISTIC
that speeds up the solve but CAN change the optimum. Default is 0 (exact).

Examples (run from the rmvp/ directory)
---------------------------------------
# EuroBonds r_c x beta grid at gamma=0.10, exact:
python3 reproduce/rvs.py --data datasets/EuroBondsRet.xlsx --r_c 2e-4 1e-4 5e-5 --beta 1e-6 5e-7 1e-7

# gamma sweep at fixed r_c:
python3 reproduce/rvs.py --data datasets/EuroBondsRet.xlsx --r_c 1e-4 --gamma 0.10 0.15 --beta 1e-6 5e-7

# a different dataset, warm-start on for a quick look:
python3 reproduce/rvs.py --data datasets/NASDAQ100_returns.xlsx --r_c 1e-4 5e-5 --beta 1e-4 5e-5 --drop 0.5
"""
import argparse
import itertools
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RMVP_DIR = os.path.dirname(HERE)                   # parent rmvp/ holds the core modules
sys.path.insert(0, RMVP_DIR)
from main import solveRMVP1, mainRMVP1BnB          # noqa: E402
from warm_start import warm_start_rmvp1            # noqa: E402

TOL = 1e-8              # solver absolute tolerance (relErr / MIPGapAbs in main.py)
SCALE = 100.0           # returns scaled x100 (matches the paper pipeline)
RIDGE_COEF = 1e-6       # default covariance ridge = ridge_coef * scale^2 * I
ANNUAL = 252            # trading days / year
NNZ_TOL = 1e-8          # nonzero-weight threshold when counting support


# ---------------------------------------------------------------------------
# Problem construction / solving / OOS evaluation
# (inlined so this module is self-contained)
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
                "n_filtered": n, "n_bnb_input": n,   # dense: no filter, all assets
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
                "n_filtered": int(idx_keep.size), "n_bnb_input": 0,
                "wall": time.time() - w0, "cpu": time.process_time() - c0}

    D_f = D[np.ix_(idx_keep, idx_keep)]
    tau_f = tau[idx_keep]

    # Warm-start elimination, then branch-and-bound on the reduced problem.
    warm = warm_start_rmvp1(D_f, tau_f, tau_bar, gamma, beta,
                            drop_fraction=drop, drop_farthest=True)
    D_r, tau_r, keep_r = warm["D_reduced"], warm["tau_reduced"], warm["keep_idx"]
    n_bnb_input = int(len(tau_r))          # assets actually fed into the BnB tree

    x_bnb, _, supp, _ = mainRMVP1BnB(D_r, tau_r, tau_bar, gamma, beta,
                                     time_limit=timeout, collect_collapse=False)
    x_bnb = np.asarray(x_bnb).reshape(-1)
    supp = np.asarray(supp, dtype=int)

    # Map reduced support -> filtered -> original indices.
    orig_idx = idx_keep[keep_r[supp]]
    x_full = np.zeros(n)
    x_full[orig_idx] = x_bnb
    return {"status": "ok", "x_full": x_full,
            "n_selected": int(orig_idx.size), "support": orig_idx,
            "n_filtered": int(idx_keep.size), "n_bnb_input": n_bnb_input,
            "wall": time.time() - w0, "cpu": time.process_time() - c0}


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
    d = np.array([x for x in diffs if np.isfinite(x)])
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return t, 1 - stats.t.cdf(t, len(d) - 1)


def wsharpe(a):
    return sharpe(a) if a.std(ddof=1) > 0 else np.nan


def run_cell(R, starts, W, H, rc, tr, gamma, beta, drop, timeout, ridge):
    """Backtest Sparse vs Robust for one (r_c, gamma, beta) cell. Returns a metrics dict."""
    spf, rof, dF = [], [], []
    ssupp, rsupp, snet, rnet = [], [], [], []
    ndiff, rwin, risk_terms = 0, 0, []
    for t in starts:
        try:
            Rin, Rtest = R[t - W:t], R[t:t + H]
            D, tau, tb = build_problem(Rin, rc, tr, ridge)
            ds = np.sqrt(np.diag(D))
            sh = np.where(np.abs(tau) > gamma * ds)[0]
            if sh.size < 2:
                continue
            sp = solve_model("Sparse", D, tau, tb, 0.0, beta, drop, timeout, restrict_idx=sh)
            ro = solve_model("SparseRobust", D, tau, tb, gamma, beta, drop, timeout, restrict_idx=sh)
            xs, xr = sp["x_full"], ro["x_full"]
            fs, fr = oos_returns(Rtest, xs, rc), oos_returns(Rtest, xr, rc)
            spf.append(fs); rof.append(fr)
            dF.append(wsharpe(fr) - wsharpe(fs))
            ssupp.append(sp["n_selected"]); rsupp.append(ro["n_selected"])
            snet.append(xs.sum()); rnet.append(xr.sum())
            risk_terms.append(xs @ D @ xs)
            if set(sp["support"].tolist()) != set(ro["support"].tolist()):
                ndiff += 1
                if fr.mean() > fs.mean():
                    rwin += 1
        except Exception as e:
            print(f"    window {t} failed: {type(e).__name__}: {e}", flush=True)
            continue
    if not spf:
        return None
    tF, pF = paired_t(dF)
    med_risk = float(np.median(risk_terms)) if risk_terms else float("nan")
    return dict(
        suppS=np.mean(ssupp), suppR=np.mean(rsupp),
        ShrpS=sharpe(np.concatenate(spf)), ShrpR=sharpe(np.concatenate(rof)),
        retS=np.mean(np.concatenate(spf)) * 252 * 100,
        retR=np.mean(np.concatenate(rof)) * 252 * 100,
        netS=np.mean(snet), netR=np.mean(rnet),
        tF=tF, pF=pF, ndiff=ndiff,
        rWin=(100.0 * rwin / ndiff) if ndiff else float("nan"),
        med_risk=med_risk, nwin=len(spf),
    )


HEADER = (f'{"r_c":>8s} {"gamma":>5s} {"beta":>8s} {"suppS":>7s} {"suppR":>7s} '
          f'{"ShrpS":>8s} {"ShrpR":>8s} {"retS%":>8s} {"retR%":>8s} '
          f'{"netS":>7s} {"netR":>7s} {"tFull":>6s} {"pFull":>7s} '
          f'{"ndiff":>5s} {"rWin%":>5s} {"flag":>5s}')


def main():
    ap = argparse.ArgumentParser(description="Robust-vs-Sparse OOS grid runner",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="path to returns .xlsx")
    ap.add_argument("--r_c", type=float, nargs="+", required=True, help="risk-free rate(s)")
    ap.add_argument("--beta", type=float, nargs="+", required=True, help="sparsity price(s)")
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.10], help="robustness radius/radii")
    ap.add_argument("--tr", type=float, default=1.05, help="target-return factor (target=r_c*tr)")
    ap.add_argument("--drop", type=float, default=0.0, help="warm-start drop fraction (0=exact BnB)")
    ap.add_argument("--ridge", type=float, default=0.0, help="covariance ridge coefficient")
    ap.add_argument("--window", type=int, default=252, help="in-sample window (days)")
    ap.add_argument("--test", type=int, default=21, help="OOS holding period (days)")
    ap.add_argument("--max-windows", type=int, default=0, help="cap #rebalances (0=all)")
    ap.add_argument("--timeout", type=int, default=90, help="per-window BnB time limit (s)")
    ap.add_argument("--out", default=None, help="results file (default: rvs_<dataset><tag>.txt)")
    ap.add_argument("--tag", default="", help="suffix for the default output filename")
    args = ap.parse_args()

    R = load_returns(args.data)
    W, H = args.window, args.test
    starts = list(range(W, R.shape[0] - H, H))
    if args.max_windows and args.max_windows > 0:
        starts = starts[:args.max_windows]

    name = os.path.splitext(os.path.basename(args.data))[0]
    out = args.out or os.path.join(HERE, f"rvs_{name}{('_' + args.tag) if args.tag else ''}.txt")

    cells = list(itertools.product(args.r_c, args.gamma, args.beta))
    meta = (f"# rvs | data={name} shape={R.shape} dailyvol={R.std():.2e} | "
            f"W={W} H={H} tr={args.tr} drop={args.drop} ridge={args.ridge} | "
            f"{len(starts)} rebalances | {len(cells)} cells")
    tolnote = ("# flag: 'tol!' = beta or median risk-term within 10x the 1e-8 solver "
               "tolerance -> numerically fragile. tFull/pFull = paired t & p on OOS excess Sharpe.")
    with open(out, "w") as f:
        f.write(meta + "\n" + tolnote + "\n" + HEADER + "\n"); f.flush()
    print(meta, flush=True); print(HEADER, flush=True)

    t0 = time.time()
    for rc, gamma, beta in cells:
        m = run_cell(R, starts, W, H, rc, args.tr, gamma, beta, args.drop, args.timeout, args.ridge)
        if m is None:
            row = f'{rc:8.1e} {gamma:5.3f} {beta:8.1e}  (no feasible windows)'
        else:
            flag = "tol!" if (beta < 10 * TOL or m["med_risk"] < 10 * TOL) else ""
            row = (f'{rc:8.1e} {gamma:5.3f} {beta:8.1e} {m["suppS"]:7.4f} {m["suppR"]:7.4f} '
                   f'{m["ShrpS"]:8.4f} {m["ShrpR"]:8.4f} {m["retS"]:8.4f} {m["retR"]:8.4f} '
                   f'{m["netS"]:+7.3f} {m["netR"]:+7.3f} {m["tF"]:6.2f} {m["pF"]:7.4f} '
                   f'{m["ndiff"]:5d} {m["rWin"]:5.0f} {flag:>5s}')
        with open(out, "a") as f:
            f.write(row + "\n"); f.flush()
        print(row, flush=True)
    with open(out, "a") as f:
        f.write(f"DONE ({time.time() - t0:.0f}s)\n"); f.flush()
    print(f"DONE ({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
