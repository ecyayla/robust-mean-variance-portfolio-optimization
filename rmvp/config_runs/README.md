# config_runs — guide

Re-runs the paper's computational study (Tables 1 & 2) driven by
`rmvp/experiment_config.json`. Two models × two solvers = **4 independent scripts**,
each writing its own timestamped Excel. This file explains the flow, where each piece
lives, and what every Excel column means.

## Models

- **RMVP1**: `min xᵀDx + β‖x‖₀`  s.t.  `τᵀx − γ‖x‖_D ≥ τ̄`.
- **RMVP2**: `min −τᵀx + τ̄ + γ‖x‖_D + β‖x‖₀`  s.t.  `‖x‖_D ≤ t`.

`D` = diag of variances, `τ` = returns vector, `τ̄` = target, `γ` = robustness radius,
`β` = sparsity penalty, `t` = risk budget (RMVP2 only).

## Files

```
config_runs/
    README.md            <- this file
    _common.py           <- shared helpers, imported by all 4 scripts
    run_rmvp1_bnb.py     <- RMVP1, Branch-and-Bound  (warm-start + drop loop)
    run_rmvp1_gurobi.py  <- RMVP1, Gurobi MIP        (single solve, no warm-start)
    run_rmvp2_bnb.py     <- RMVP2, Branch-and-Bound  (warm-start + drop loop)
    run_rmvp2_gurobi.py  <- RMVP2, Gurobi MIP        (single solve, no warm-start)
    <prefix>_YYYY-MM-DD_HH-MM.xlsx   <- outputs, one per run
```

Supporting code lives one level up in `rmvp/`:

- `test.py` — `generate_rmvp1_data`, `generate_rmvp2_data` (build `D, τ, τ̄`, apply
  assumption filters), `load_experiment_config`, `get_dataset_files`.
- `warm_start.py` — `warm_start_rmvp1/2` (drop-fraction elimination before BnB).
- `main.py` — `mainRMVP1BnB`, `mainRMVP2BnB` (BnB solvers), `RMVP1_mipGUROBI`,
  `RMVP2_mipGUROBI` (Gurobi), `zeropadding`.

## How to run

```bash
cd rmvp/config_runs
python3 run_rmvp1_bnb.py       # each script is standalone
```

Datasets and parameter grids come from `experiment_config.json` (`rmvp1` / `rmvp2`
sections). A script only processes datasets with `"enabled": true`. Edit the config to
choose datasets, `beta_values`, `gamma_inputs`, `t_values`, `drop_fractions_by_beta`,
etc. The four scripts run independently (e.g. in separate tmux panes).

## Flow (per script)

1. Load config + list dataset files.
2. For each enabled dataset, iterate the parameter grid:
   `product(rc_values, beta_values, target_return_factors[, t_values])`, then over
   `gamma_inputs`.
3. Build the problem with `generate_rmvp*_data`. **The assumption filter runs inside
   these functions** — the returned `D, τ` are already filtered:
   - RMVP1: keep asset `i` iff `|τ_i| > γ·√D_ii` (Assumption 1).
   - RMVP2: Assumption 1, then Assumption 2 `|τ_i|/√D_ii > γ + β/t`.
   Both solvers of a model therefore see the **same** asset set.
4. Skip if fewer than 2 assets survive.
5. **BnB scripts only:** for each drop fraction (`resolve_drop_fractions`), warm-start
   eliminate, then solve BnB. **Gurobi scripts:** one solve on the full filtered set.
6. `append_row` writes one Excel row immediately after each solve.

## Incremental / crash-safe writing

`append_row` (in `_common.py`) writes each result the moment it is produced: first call
creates the file with a header, later calls append without a header. If a script dies
mid-run, every row already computed is on disk and the `.xlsx` opens fine.

## Timing

Two clocks per solve:

- **wall** (`time.time`) — real elapsed time. This is the headline number.
- **cpu** (`time.process_time`) — sums CPU across threads.

BnB keeps numpy/BLAS at **default threads** (multi-thread) because we report wall time;
Gurobi runs **single-thread** (`Threads=1` inside `RMVP*_mipGUROBI`). Note: wall of a
multi-thread BnB can exceed its cpu/threads; and running a BnB job and a Gurobi job at
the same time on this shared 8-core host distorts both wall times via CPU contention —
stagger them for clean numbers.

## drop_fraction (BnB warm-start)

Before BnB, `warm_start_rmvp*` removes a fraction of assets least likely to be selected
(the "farthest") to shrink the tree. `drop_fractions_by_beta` in the config maps each
`beta` to a list of fractions to try. Keys are matched **by float value**, so
scientific-notation betas like `0.00005` work (a plain `str(beta)` lookup would render
that as `'5e-05'` and silently skip it). Gurobi scripts do no warm-start / dropping.

## Right-subtree collapse — the two collapse columns

`mainRMVP1BnB` / `mainRMVP2BnB` include an inline **right-subtree collapse**: when a
branch's right child cannot beat the incumbent (`right_lb + β ≥ global_ub − relErr`),
the whole right subtree is pruned in place. Every time this fires we record the sizes of
that node's two sets:

- `S` (`Ssupp`) = variables already **forced in**.
- `P` (`Psupp`) = variables still **free to branch** (the branched variable already
  removed).

Called with `collect_collapse=True`, the BnB returns two lists (one entry per collapse
firing). The scripts join them into comma-separated strings:

- `collapse_S_sizes = "1,5,9"` — `|S|` at the 1st, 2nd, 3rd collapse …
- `collapse_P_sizes = "4,7,11"` — `|P|` at the same firings.

The two strings always have the **same number of entries** (one per collapse). Empty
string = no collapse fired for that run. Gurobi has no tree, so its scripts have no
collapse columns.

## Excel columns

### Shared identity / problem-size columns (all 4 files)

| Column | Meaning |
|---|---|
| `Dataset` | Full path of the input `.xlsx`. |
| `n` | Full asset count before any filter (`n_full`). |
| `n_filtered` | Assets surviving the assumption filter(s); this is what the solver saw. |
| `gamma` | Robustness radius actually used (dynamic γ resolved). |
| `r_c` | Risk-free / cash return parameter. |
| `beta` | Sparsity penalty as given in the config. |
| `beta_scaled` | `beta` after internal return scaling (the value the solver used). |
| `tr_factor` | Target-return factor (sets `τ̄`). |
| `t` | Risk budget. RMVP2: real value. RMVP1: `None`. |
| `dropped_assets` | Total assets removed by filters. RMVP1: `= dropped_gamma`. RMVP2: `dropped_gamma + dropped_t`. |
| `dropped_gamma` | Assets removed by Assumption 1 (`|τ_i| ≤ γ·√D_ii`). |
| `dropped_t` | Assets removed by RMVP2 Assumption 2. Always `0` for RMVP1. |

### BnB-only columns (`run_rmvp1_bnb`, `run_rmvp2_bnb`)

| Column | Meaning |
|---|---|
| `drop_fraction` | Warm-start elimination fraction for this row. |
| `drop_farthest` | Warm-start strategy flag (`True` = drop farthest assets). |
| `n_warm_start` | Assets left after warm-start elimination (what BnB solved). |
| `warm_start_dropped_assets` | `n_filtered − n_warm_start`. |
| `time_cpu_warm_start` | CPU seconds spent in warm-start. |
| `obj_BnB_warm` | Objective from BnB (formula per model above; includes `β·nnz`). |
| `nnz_BnB_warm` | Number of selected assets (non-zeros in `x`). |
| `bnb_nodes` | Nodes explored by BnB (search-effort proxy). |
| `time_BnB_warm` | **Wall** seconds of the BnB solve. |
| `time_cpu_BnB_warm` | **CPU** seconds of the BnB solve. |
| `time_cpu_total` | `time_cpu_warm_start + time_cpu_BnB_warm`. |
| `collapse_S_sizes` | Comma-separated `|S|` at each right-subtree collapse. |
| `collapse_P_sizes` | Comma-separated `|P|` at each right-subtree collapse. |

### Gurobi-only columns (`run_rmvp1_gurobi`, `run_rmvp2_gurobi`)

| Column | Meaning |
|---|---|
| `obj_MIP_GUROBI` | Objective from Gurobi (same formula as BnB → directly comparable). |
| `nnz_MIP_GUROBI` | Number of selected assets. |
| `time_MIP_GUROBI` | **Wall** seconds of the Gurobi solve. |
| `time_cpu_MIP_GUROBI` | **CPU** seconds (Gurobi is single-thread). |
| `gap_MIP_GUROBI` | Final MIP optimality gap reported by Gurobi. |

## Comparing BnB vs Gurobi

BnB and Gurobi results land in separate files. Join on
`(Dataset, r_c, beta, tr_factor, gamma[, t])` and compare:

- correctness: `obj_BnB_warm − obj_MIP_GUROBI` (should be ~0; matching `nnz` too),
- speed: wall ratio `time_MIP_GUROBI / time_BnB_warm`.

On the EuroStoxx50 check (no drop) the two agree to `1e-9`–`1e-11` and match the paper's
Tables 1 & 2 (`nnz` = 1, 3, 14, 26).

## Gurobi license

`_common.py` sets `GRB_LICENSE_FILE` to the project-root `gurobi.lic` (token server
`139.179.39.137:41954`) if that file exists and no license env var is already set —
otherwise Gurobi would fall through to an expired `/opt/gurobi/gurobi.lic`. If you move
this to another machine, point `GRB_LICENSE_FILE` at a valid license or the Gurobi
scripts will fail (the BnB scripts don't need Gurobi).
