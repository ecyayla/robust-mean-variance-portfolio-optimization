import numpy as np
from math import ceil
from main import solveRMVP1
from check_theorem import rmvp1_global_lower_bound_check


def warm_start_rmvp1(D, tau, tau_bar, gamma, beta, drop_fraction=0.1, tol=1e-8):
    """
    Warm-start strategy for RMVP1:
      1) Solve the unrestricted closed-form RMVP1 to get x_hat.
      2) Evaluate the global lower-bound check; compute deficits = max(0, lower_bound - |x_i|).
      3) Drop the top drop_fraction of assets with the largest deficits.
      4) Return the reduced problem data with those assets removed.

    Args:
        D: (n,n) PSD matrix.
        tau: (n,) vector.
        tau_bar: scalar.
        gamma: scalar.
        beta: scalar.
        drop_fraction: fraction (0–1) of assets to drop; clipped to [0,1).
        tol: threshold to define support and numerical zeroing.

    Returns:
        {
          "D_reduced": filtered D,
          "tau_reduced": filtered tau,
          "x_reduced": filtered x_hat (1D),
          "kept_indices": list of kept asset indices,
          "dropped_indices": list of dropped asset indices,
          "check_result": theorem check on the unrestricted x_hat
        }
    """
    n = D.shape[0]

    # Step 1: closed-form unrestricted solution
    x_hat, _, _ = solveRMVP1(D, tau_bar, tau, gamma)
    x_hat = x_hat.reshape(-1)

    # Step 2: theorem-based deficits
    chk = rmvp1_global_lower_bound_check(D, tau, tau_bar, gamma, beta, x_hat, tol=tol)
    deficits = np.zeros(n)
    for d in chk["details"]:
        i = d["i"]
        deficits[i] = max(0.0, d["lower_bound"] - d["abs_xi"])

    # Step 3: drop top-k by deficit
    k = ceil(drop_fraction * n)
    if k > 0:
        drop_order = np.argsort(deficits)[::-1]  # descending deficits
        dropped_indices = drop_order[:k].tolist()
    else:
        dropped_indices = []

    mask = np.ones(n, dtype=bool)
    if dropped_indices:
        mask[dropped_indices] = False
    kept_indices = np.where(mask)[0].tolist()

    D_reduced = D[np.ix_(mask, mask)]
    tau_reduced = tau[mask]
    x_reduced = x_hat[mask]

    return {
        "D_reduced": D_reduced,
        "tau_reduced": tau_reduced,
        "x_reduced": x_reduced,
        "kept_indices": kept_indices,
        "dropped_indices": dropped_indices,
        "check_result": chk,
    }


if __name__ == "__main__":
    print("Provide D, tau, tau_bar, gamma, beta to use warm_start_rmvp1.")

