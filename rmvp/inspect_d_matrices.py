from pathlib import Path

import numpy as np
import pandas as pd


def load_returns(file_path, sheet_name=0, scale_returns=100.0):
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    returns = data.values * scale_returns
    return returns


def compute_d_matrix(returns):
    n_samples = returns.shape[0]
    r_hat = returns.mean(axis=0)
    returns_centered = returns - r_hat
    return (returns_centered.T @ returns_centered) / (n_samples - 1)


def sparsity_stats(matrix, eps):
    total = matrix.size
    near_zero = np.sum(np.abs(matrix) <= eps)
    return near_zero / total if total else 0.0


def describe_matrix(name, D, eps):
    diag = np.diag(D)
    off_diag_mask = ~np.eye(D.shape[0], dtype=bool)
    off_diag = D[off_diag_mask]
    eigvals = np.linalg.eigvalsh(D)
    cond = np.inf if np.min(eigvals) <= 0 else np.max(eigvals) / np.min(eigvals)

    stats = {
        "matrix": name,
        "n": D.shape[0],
        "min_eig": float(np.min(eigvals)),
        "max_eig": float(np.max(eigvals)),
        "cond": float(cond),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "diag_mean": float(np.mean(diag)),
        "offdiag_min": float(np.min(off_diag)) if off_diag.size else 0.0,
        "offdiag_max": float(np.max(off_diag)) if off_diag.size else 0.0,
        "offdiag_mean": float(np.mean(off_diag)) if off_diag.size else 0.0,
        "sparsity_all": sparsity_stats(D, eps),
        "sparsity_offdiag": sparsity_stats(off_diag, eps) if off_diag.size else 0.0,
    }
    return stats


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "datasets"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls")))
    if not files:
        print(f"No datasets found in {data_dir}")
        return

    rows = []
    for file_path in files:
        returns = load_returns(file_path, sheet_name=0, scale_returns=100.0)
        D = compute_d_matrix(returns)
        stats = describe_matrix(file_path.name, D, 1e-8)
        rows.append(stats)

    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
