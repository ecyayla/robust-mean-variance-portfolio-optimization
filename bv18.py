import numpy as np
from numpy.linalg import norm


def _objective(x, A, b, c):
    return float(x.T @ (A @ x) + 2.0 * (b.T @ x) + c)


def _project_onto_ball(x, r):
    nrm = norm(x)
    if nrm <= r or nrm == 0.0:
        return x
    return (r / nrm) * x


def _get_warm_start(meta_in, n, r):
    if meta_in is None or not isinstance(meta_in, dict):
        return np.zeros((n, 1))
    x0 = meta_in.get("x0", None)
    if x0 is None:
        return np.zeros((n, 1))
    x0 = np.asarray(x0, float).reshape(-1, 1)
    if x0.shape[0] != n:
        return np.zeros((n, 1))
    return _project_onto_ball(x0, r)


def solveTRP_BV18_PG(A, b, c=0.0, r=1.0, max_iter=1000, tol=1e-8, step_size=None, meta_in=None):
    """
    Beck-Vaisbourd (2018) style projected gradient (PG) for TRS.
    Warm-start uses meta_in["x0"] when provided.
    Returns (x_opt, f_opt, meta_out).
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1, 1)
    n = A.shape[0]

    x = _get_warm_start(meta_in, n, r)

    if step_size is None:
        L = float(np.linalg.norm(A, 2))
        step_size = 1.0 if L <= 0.0 else 1.0 / L

    for _ in range(max_iter):
        grad = A @ x + b
        x_new = x - step_size * grad
        x_new = _project_onto_ball(x_new, r)
        if norm(x_new - x) <= tol * max(1.0, norm(x)):
            x = x_new
            break
        x = x_new

    f_opt = _objective(x, A, b, c)
    meta_out = {"x": x.reshape(-1)}
    return x, f_opt, meta_out


def solveTRP_BV18_CG(A, b, c=0.0, r=1.0, max_iter=1000, tol=1e-8, meta_in=None):
    """
    Beck-Vaisbourd (2018) style conditional gradient (CG / Frank-Wolfe) for TRS.
    Warm-start uses meta_in["x0"] when provided.
    Returns (x_opt, f_opt, meta_out).
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1, 1)
    n = A.shape[0]

    x = _get_warm_start(meta_in, n, r)

    for _ in range(max_iter):
        grad = A @ x + b
        gnorm = norm(grad)
        if gnorm <= tol:
            break

        s = (-r / gnorm) * grad
        d = s - x
        if norm(d) <= tol * max(1.0, norm(x)):
            x = s
            break

        Ad = A @ d
        denom = float(d.T @ Ad)
        numer = float(d.T @ (A @ x) + b.T @ d)

        if denom > 0.0:
            gamma = -numer / denom
            if gamma < 0.0:
                gamma = 0.0
            elif gamma > 1.0:
                gamma = 1.0
        else:
            gamma = 1.0 if numer < 0.0 else 0.0

        x_new = x + gamma * d
        if norm(x_new - x) <= tol * max(1.0, norm(x)):
            x = x_new
            break
        x = x_new

    f_opt = _objective(x, A, b, c)
    meta_out = {"x": x.reshape(-1)}
    return x, f_opt, meta_out
