# Sparsity Regularized and Robust Mean–Variance Portfolio Selection

Code accompanying the paper *"Sparsity Regularized and Robust Mean Variance Portfolio
Selection Under Ellipsoidal Uncertainty."*

We study mean–variance portfolio selection that combines an ℓ₀ sparsity penalty with
ellipsoidal uncertainty in the mean return vector, in two formulations:

- **CP-RMVP** (risk minimization): `min xᵀDx + β‖x‖₀`  s.t. `𝔯ᵀx − γ‖x‖_D ≥ 𝔯̄`
- **CP-RMVP2** (return maximization under a variance budget): `min γ‖x‖_D − 𝔯ᵀx + β‖x‖₀`  s.t. `‖x‖_D ≤ T`

where `D` is the covariance, `𝔯 = r̂ − r_c·1` the estimated excess return, `γ` the
robustness radius, and `β` the sparsity penalty. Both are solved with a tailored
queue-based branch-and-bound algorithm, benchmarked against a Gurobi MISOCP reformulation.

## Repository layout

```
rmvp/
├── main.py                  core branch-and-bound + Gurobi MISOCP solvers
├── warm_start.py            warm-start elimination heuristic
├── test.py                  data generation / experiment-config helpers
├── preprocess_rmvp.py       price → returns (price-based datasets)
├── preprocess_ff100.py      raw Fama–French CSV → datasets/FF100_SizeBM.xlsx
├── reproduce/               experiment drivers — see reproduce/README.md
│   ├── run_rmvp*_{bnb,gurobi}.py   Tables 2–3 (config-driven)
│   └── rvs.py                       §6.1 out-of-sample robust-vs-sparse runner
├── datasets/                returns panels (.xlsx); raw/ holds source files
└── experiments/             result files
```

## Installation

```bash
pip install -r requirements.txt
```

`gurobipy` is only required for the Gurobi benchmark scripts in `rmvp/reproduce/`
(and needs a valid Gurobi license). The branch-and-bound method itself does not depend
on Gurobi.

## Reproducing the results

**Computational study (Tables 2–3).** See [`rmvp/reproduce/README.md`](rmvp/reproduce/README.md);
runs are driven by `rmvp/experiment_config.json`.

**Out-of-sample study (Section 6.1).** The Fama–French 100 Size×Book-to-Market panel is
built from the raw daily CSV in `rmvp/datasets/raw/` (available from the
[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)):

```bash
cd rmvp
python preprocess_ff100.py          # -> datasets/FF100_SizeBM.xlsx
python reproduce/rvs.py --data datasets/FF100_SizeBM.xlsx \
    --r_c 5e-5 --gamma 0.10 0.15 --beta 1e-6 5e-7 --drop 0.3
```

Each `(r_c, γ, β)` cell reports the sparse (γ=0) and robust (γ>0) out-of-sample Sharpe
ratios, mean excess returns, average cardinality, and the one-sided paired p-value on
the per-window Sharpe difference. `--drop 0` runs exact branch-and-bound (recommended for
final numbers); `--drop > 0` enables the warm-start elimination heuristic.

## Data

Price/return panels for the index and bond datasets are taken from the sources cited in
the paper. The Fama–French 100 portfolios are from the Kenneth French Data Library (see
link above).

## License

See [LICENSE](LICENSE).
