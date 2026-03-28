import numpy as np
from math import ceil
from main import solveRMVP1, solveRMVP2


def warm_start_rmvp1(D, tau, tau_bar, gamma, beta, drop_fraction=0.1, drop_farthest=True):
    """
    Warm-start strategy for RMVP1 based on Theorem 3 values:
      1) Solve the unrestricted closed-form RMVP1 to get x_hat.
      2) Compute Theorem 3 values (eta, rho_i, bound1, bound2, lower_bound).
      3) Score each asset by deficit = lower_bound - |x_hat[i]|.
      4) Drop the top drop_fraction assets by deficit (farthest by default,
         closest if drop_farthest=False).

    Assumptions: |tau_i| > gamma * sqrt(D_ii) for all assets.
    """
    D = np.asarray(D)
    tau = np.asarray(tau).reshape(-1)
    n = D.shape[0]

    # Step 1: closed-form unrestricted solution
    x_hat, _, _ = solveRMVP1(D, tau_bar, tau, gamma)
    x_hat = x_hat.reshape(-1)

    # Step 2: Theorem 3 values
    diag_D = np.diag(D)
    diag_sqrt = np.sqrt(diag_D)
    abs_tau = np.abs(tau)

    denom_eta = (tau - gamma * diag_sqrt) ** 2
    eta_vals = (tau_bar ** 2) * diag_D / denom_eta
    eta = float(np.min(eta_vals))

    denom_all = abs_tau - gamma * diag_sqrt
    bound1_scale = np.sqrt(eta + beta) - np.sqrt(eta)

    deficits = np.zeros(n)
    for i in range(n):
        denom_i = denom_all[i]
        bound2 = tau_bar / denom_i

        numer_i = abs_tau[i] + gamma * diag_sqrt[i]
        ratios = numer_i / denom_all

        D_ii = diag_D[i]
        D_i = D[i]
        quad_vals = D_ii + (ratios ** 2) * diag_D + 2.0 * np.abs(ratios * D_i)
        quad_vals[i] = -np.inf
        rho_i = float(np.sqrt(np.max(quad_vals)))

        bound1 = bound1_scale / rho_i
        lower_bound = min(bound1, bound2)

        deficits[i] = lower_bound - abs(x_hat[i])

    # Step 3: drop top-k by deficit
    # drop_fraction=0 keeps only assumption filtering (no extra drops)
    k = 0 if drop_fraction == 0 else ceil(drop_fraction * n)
    if k > 0:
        if drop_farthest:
            topk = np.argpartition(deficits, -k)[-k:]
        else:
            topk = np.argpartition(deficits, k)[:k]
        dropped = topk.tolist()
    else:
        dropped = []

    mask = np.ones(n, dtype=bool)
    if dropped:
        mask[dropped] = False

    D_reduced = D[mask][:, mask]
    tau_reduced = tau[mask]
    x_reduced = x_hat[mask]

    return {
        "D_reduced": D_reduced,
        "tau_reduced": tau_reduced,
        "x_reduced": x_reduced,
    }


def warm_start_rmvp2(D, tau, tau_bar, gamma, beta, t, drop_fraction=0.1, drop_farthest=True):
    """
    Warm-start strategy for RMVP2 based on Proposition 4:
      1) Solve the unrestricted closed-form RMVP2 to get x_hat.
      2) Compute H = sqrt(tau^T D^{-1} tau).
      3) Score each asset by deficit = lower_bound - |x_hat[i]|,
         where lower_bound = min(beta / (|tau_i| + H * sqrt(D_ii)), t / sqrt(D_ii)).
      4) Drop the top drop_fraction assets by deficit (farthest by default,
         closest if drop_farthest=False).
    """
    D = np.asarray(D)
    tau = np.asarray(tau).reshape(-1)
    n = D.shape[0]

    x_hat, _, _ = solveRMVP2(D, tau_bar, tau, gamma, beta, t)
    x_hat = x_hat.reshape(-1)

    diag_D = np.diag(D)
    diag_sqrt = np.sqrt(diag_D)
    diag_sqrt_safe = np.maximum(diag_sqrt, 1e-8)

    D_inv = np.linalg.inv(D)
    H = float(np.sqrt(tau.T @ D_inv @ tau))

    abs_tau = np.abs(tau)
    denom_a = np.maximum(abs_tau + H * diag_sqrt, 1e-12)
    bound_a = beta / denom_a
    bound_b = t / diag_sqrt_safe
    lower_bound = np.minimum(bound_a, bound_b)

    deficits = lower_bound - np.abs(x_hat)

    # drop_fraction=0 keeps only assumption filtering (no extra drops)
    k = 0 if drop_fraction == 0 else ceil(drop_fraction * n)
    if k > 0:
        if drop_farthest:
            topk = np.argpartition(deficits, -k)[-k:]
        else:
            topk = np.argpartition(deficits, k)[:k]
        dropped = topk.tolist()
    else:
        dropped = []

    mask = np.ones(n, dtype=bool)
    if dropped:
        mask[dropped] = False

    D_reduced = D[mask][:, mask]
    tau_reduced = tau[mask]
    x_reduced = x_hat[mask]

    return {
        "D_reduced": D_reduced,
        "tau_reduced": tau_reduced,
        "x_reduced": x_reduced,
    }


if __name__ == "__main__":
    print("Provide D, tau, tau_bar, gamma, beta to use warm_start_rmvp1.")

