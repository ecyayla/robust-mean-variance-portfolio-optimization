import numpy as np
import cvxpy as cp
from queue import PriorityQueue, LifoQueue
import time
import gurobipy as gp
from gurobipy import GRB
import mosek
import mosek.fusion as mf
import math

# Import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 


def zeropadding(u,w,n):
    
    U =  np.zeros((n,1))  #Makes u with 4 significant level!!
    cnt = 0
    for i in range(n):
        if np.isin(i,w):
            U[i] = u[cnt]
            cnt = cnt+1
        else:
            U[i] = 0
    return U


def is_symmetric(A, tol=1e-10):
    """
    Check if a matrix is symmetric.
    
    Parameters
    ----------
    A : (n, n) array_like
        Matrix to check.
    tol : float, optional
        Tolerance for checking symmetry. Default is 1e-10.
    
    Returns
    -------
    is_sym : bool
        True if A is symmetric (within tolerance).
    max_asymmetry : float
        Maximum absolute difference between A and A.T.
    """
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        return False, None
    max_asymmetry = np.max(np.abs(A - A.T))
    is_sym = max_asymmetry < tol
    return is_sym, max_asymmetry


def dataset_to_Q(X, y, reg=1e-3):
    """
    Build the SVM Q matrix from feature matrix X and label vector y.
    Q[i, j] = y_i * y_j * <x_i, x_j>.

    Parameters
    ----------
    X : (m, d) array_like
        Feature matrix.
    y : (m,) or (m, 1) array_like
        Label vector taking values in {-1, +1}.
    reg : float, optional
        Diagonal regularization added for numerical stability.

    Returns
    -------
    Q : (m, m) ndarray
        Kernel/Hessian matrix.
    y_vec : (m,) ndarray
        Flattened label vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    K = X @ X.T
    Q = (y @ y.T) * K
    Q = Q + np.eye(Q.shape[0]) * reg
    return Q, y

def normalize_features(X, method='standardize'):
    """
    Normalize feature matrix for SVM preprocessing.
    
    Parameters
    ----------
    X : (m, d) ndarray
        Feature matrix.
    method : str, optional
        Normalization method:
        - 'standardize': Zero mean, unit variance (default, most common for SVM)
        - 'minmax': Scale to [0, 1]
        - 'l2': L2 normalization (unit norm per sample)
        - None: No normalization
    
    Returns
    -------
    X_norm : (m, d) ndarray
        Normalized feature matrix.
    """
    X = np.asarray(X, dtype=float)
    
    if method is None or method == 'none':
        return X
    
    if method == 'standardize':
        # Standardization: zero mean, unit variance
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True)
        # Avoid division by zero for constant features
        X_std = np.where(X_std < 1e-10, 1.0, X_std)
        return (X - X_mean) / X_std
    
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        X_min = np.min(X, axis=0, keepdims=True)
        X_max = np.max(X, axis=0, keepdims=True)
        X_range = X_max - X_min
        X_range = np.where(X_range < 1e-10, 1.0, X_range)
        return (X - X_min) / X_range
    
    elif method == 'l2':
        # L2 normalization: each sample has unit norm
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return X / norms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                        f"Choose from: 'standardize', 'minmax', 'l2', None")


def getDataset_breastcancerwisconsin(normalize=True):
    """
    Load the breast cancer dataset from scikit-learn.
    
    Parameters
    ----------
    normalize : bool, optional
        If True, standardize features to zero mean and unit variance (default: True).
        This is standard practice for SVM preprocessing.
    
    Returns
    -------
    X : (m, d) ndarray
        Feature matrix (normalized if normalize=True).
    y : (m,) ndarray
        Label vector in {-1, +1}.
    """
    
    meta = load_breast_cancer()
    X = np.asarray(meta.data, dtype=float)
    y = np.asarray(meta.target, dtype=float)  # 0/1 labels
    
    # Feature normalization (standardization: zero mean, unit variance)
    if normalize:
        X = normalize_features(X, method='standardize')
    
    # Map to {-1, +1}
    y = np.where(y == 1.0, 1.0, -1.0)
    
    return X, y

def getDataset_heartdisease(normalize=True):
    """
    Load the heart disease dataset from UCI.
    
    Parameters
    ----------
    normalize : bool, optional
        If True, standardize features to zero mean and unit variance (default: True).
        This is standard practice for SVM preprocessing.
    """

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features 
    y = heart_disease.data.targets
    
    # Feature normalization (standardization: zero mean, unit variance)
    if normalize:
        X = normalize_features(X, method='standardize')

    return X, y
    

def lagrangian_gradient(alpha_s, Q, Ssupp, Psupp):
    """
    Raw gradient of the dual quadratic term (NO MULTIPLIERS).

        grad_P = Q_{P,S} alpha_S - 1_P

    Parameters
    ----------
    alpha_s : (|S|,) or (|S|,1)
        Current alpha values on support S.
    Q       : (n, n)
        Kernel/Hessian matrix.
    Ssupp   : 1-D array
        Indices of active support.
    Psupp   : 1-D array
        Indices of candidate (still-zero) coordinates.

    Returns
    -------
    grad_p : (|Psupp|, 1)
        Raw gradient components for Psupp.
    """
    Ssupp = np.asarray(Ssupp, dtype=np.int32)
    Psupp = np.asarray(Psupp, dtype=np.int32)

    if len(Ssupp) > 0:
        alpha_s = alpha_s.reshape(-1, 1)
        Q_ps = Q[Psupp][:, Ssupp]
        grad_p = (Q_ps @ alpha_s).reshape(-1) - 1.0
    else:
        grad_p = -np.ones(len(Psupp))

    return grad_p.reshape(-1, 1)


def solveSVM_DualMosek(Q, y, precision_tol=1e-8):
    """
    Solve the SVM dual problem using MOSEK.
    
    min_α  (1/2)α^T Q α - ⟨1,α⟩
    s.t.   ⟨y,α⟩ = 0
           α ≥ 0
    
    Parameters
    ----------
    Q : (m, m) array_like
        Kernel/Hessian matrix Q where Q[i,j] = y_i * y_j * <x_i, x_j>.
    y : (m,) or (m, 1) array_like
        Label vector.
    precision_tol : float, optional
        Numerical precision tolerance for MOSEK solver. Default is 1e-10.
        Lower values = higher precision (but slower). Typical range: 1e-8 to 1e-12.
    
    Returns
    -------
    alpha_opt : (m, 1) ndarray
        Optimal dual variables.
    obj_val : float
        Optimal objective value.
    
    Raises
    ------
    Exception
        If the problem is not solved to optimality.
    """
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float).flatten()
    m = Q.shape[0]
    
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if len(y) != m:
        raise ValueError("y must have the same length as Q dimensions")
    
    M = mf.Model("SVM_Dual")
    
    # Variable: α ≥ 0
    alpha = M.variable("alpha", m, mf.Domain.greaterThan(0.0))
    
    # Constraint: ⟨y,α⟩ = 0
    M.constraint("equality", mf.Expr.dot(y, alpha), mf.Domain.equalsTo(0.0))
    
    # Objective: (1/2)α^T Q α - 1^T α
    # Since Q is PSD, factorize Q = L L^T using Cholesky
    # Then α^T Q α = α^T L L^T α = ||L^T α||^2
    # We use auxiliary variable t and rotated quadratic cone: t >= (1/2) ||L^T α||^2
    try:
        L = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        # If Cholesky fails, Q might not be exactly PSD due to numerical issues
        # Add small regularization and try again
        Q_reg = Q + np.eye(m) * 1e-8
        L = np.linalg.cholesky(Q_reg)
    
    # Auxiliary variable t for the quadratic term
    t = M.variable("t", 1, mf.Domain.greaterThan(0.0))
    
    # L^T as MOSEK Matrix
    L_T = mf.Matrix.dense(L.T)
    
    # Compute L^T * α
    L_T_alpha = mf.Expr.mul(L_T, alpha)
    
    # Rotated quadratic cone constraint: t >= (1/2) ||L^T α||^2
    # RQC: (t, u, x) means t*u >= ||x||^2, t>=0, u>=0
    # We want: t >= (1/2) ||L^T α||^2
    # This means: t * 0.5 >= ||L^T α||^2 is NOT what we want
    # Actually: t >= (1/2) ||L^T α||^2 means 2t >= ||L^T α||^2
    # So: (2t, 1, L^T α) in RQC means 2t * 1 >= ||L^T α||^2, which is correct
    # But let's try: (t, 2, L^T α) means t * 2 >= ||L^T α||^2, so t >= (1/2) ||L^T α||^2
    # This is equivalent and might be more numerically stable
    M.constraint("quad_cone", 
                 mf.Expr.vstack(t, 1, L_T_alpha), 
                 mf.Domain.inRotatedQCone())
    
    # Linear term: -1^T α = -sum(alpha)
    linear_term = mf.Expr.sum(alpha)
    
    # Objective: t - 1^T α 
    # At optimality, t will be at its lower bound: t = (1/2) ||L^T α||^2 = (1/2) α^T Q α
    # So the objective becomes: (1/2) α^T Q α - 1^T α
    objective = mf.Expr.sub(t, linear_term)
    
    M.objective("obj", mf.ObjectiveSense.Minimize, objective)
    
    # Set high precision parameters for MOSEK
    # In MOSEK Fusion, parameters are set on the task object
    # Tighter tolerances for higher numerical precision
    task = M.getTask()
    task.putdouparam(mosek.dparam.intpnt_tol_pfeas, precision_tol)      # Primal feasibility tolerance
    task.putdouparam(mosek.dparam.intpnt_tol_dfeas, precision_tol)      # Dual feasibility tolerance
    task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, precision_tol)    # Relative gap tolerance
    task.putdouparam(mosek.dparam.intpnt_tol_infeas, precision_tol)     # Infeasibility tolerance
    task.putdouparam(mosek.dparam.intpnt_tol_mu_red, precision_tol)     # Mu reduction tolerance
    task.putdouparam(mosek.dparam.basis_tol_x, precision_tol)           # Basis tolerance for variables
    task.putdouparam(mosek.dparam.basis_tol_s, precision_tol)           # Basis tolerance for slacks
    
    M.solve()
    
    solsta = M.getPrimalSolutionStatus()
    if solsta != mf.SolutionStatus.Optimal:
        M.dispose()
        raise Exception(f"Unexpected solution status: {solsta}")
    
    alpha_opt = M.getVariable('alpha').level().reshape(-1, 1)
    obj_val = M.primalObjValue()
    M.dispose()
    
    return alpha_opt, obj_val


def solveSVM_DualCvxpy(Q, y, max_iter=10000, solver="SCS", alpha0=None):
    """
    Solve the SVM dual problem using CVXPY.
    
    Parameters
    ----------
    Q : (m, m) array_like
        Kernel matrix.
    y : (m,) or (m, 1) array_like
        Label vector.
    max_iter : int, optional
        Maximum number of iterations. Default is 10000.
    solver : str, optional
        Solver to use. If None, CVXPY chooses automatically.
        Options: 'ECOS', 'SCS', 'OSQP', 'MOSEK', etc.
    
    Returns
    -------
    alpha_opt : (m,) ndarray
        Optimal dual variables.
    obj_val : float
        Optimal objective value.
    """
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float).flatten()
    m = Q.shape[0]
    alpha = cp.Variable(m)
    if alpha0 is not None:
        try:
            alpha.value = np.asarray(alpha0, dtype=float).reshape(-1)
        except Exception:
            alpha.value = None
    constraints = [y.T @ alpha == 0, alpha >= 0]
    objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))
    problem = cp.Problem(objective, constraints)
    
    # Set solver-specific parameters for max_iter
    solver_kwargs = {}
    if solver is None:
        # Default: use max_iters for ECOS/SCS
        solver_kwargs = {'max_iters': max_iter}
    elif solver.upper() == 'ECOS':
        solver_kwargs = {'solver': cp.ECOS, 'max_iters': max_iter}
    elif solver.upper() == 'SCS':
        solver_kwargs = {'solver': cp.SCS, 'max_iters': max_iter}
    elif solver.upper() == 'OSQP':
        solver_kwargs = {'solver': cp.OSQP, 'max_iter': max_iter}
    elif solver.upper() == 'MOSEK':
        solver_kwargs = {
            'solver': cp.MOSEK,
            'mosek_params': {'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iter}
        }
    else:
        solver_kwargs = {'solver': solver, 'max_iters': max_iter}
    
    problem.solve(verbose=False, warm_start=alpha0 is not None, **solver_kwargs)

    if problem.status == cp.OPTIMAL:
        return alpha.value, problem.value
    else:
        print(f"Warning: Problem not optimal, status: {problem.status}")
        print("Q: ", Q)
        print("y: ", y)
        return np.zeros((m, 1)), np.inf


def solveSVM(Q, y, method='mosek', meta_in=None):
    alpha0 = None
    if meta_in is not None:
        alpha0 = meta_in.get("alpha0", None)

    if method == 'mosek':
        alpha_opt, obj_val = solveSVM_DualMosek(Q, y)
    elif method == 'cvxpy':
        alpha_opt, obj_val = solveSVM_DualCvxpy(Q, y, alpha0=alpha0)
    else:
        raise ValueError(f"Invalid method: {method}")

    meta_out = {"solver": method, "status": "OK", "alpha": alpha_opt}
    return alpha_opt, obj_val, meta_out




def branchVariable(alpha, Q, y, Ssupp, Psupp, branch_rule='max_lagrangian_grad'):
    if branch_rule == 'max_lagrangian_grad':
        grad = lagrangian_gradient(alpha, Q, Ssupp, Psupp)
        bb_ind = np.argmax(abs(grad[:,0]))
        ind = Psupp[bb_ind]
    else:
        raise ValueError(f"Invalid branch rule: {branch_rule}")

    return ind


def _slice_alpha_to_child(alpha_parent, w_parent, w_child):
    """
    Map parent alpha (length |w_parent|) to child ordering w_child.
    Returns None if dimensions mismatch or indices are inconsistent.
    """
    if alpha_parent is None or w_parent is None:
        return None
    w_to_pos = {idx: pos for pos, idx in enumerate(w_parent)}
    alpha_parent = np.asarray(alpha_parent).reshape(-1)
    if alpha_parent.shape[0] != len(w_parent):
        return None
    alpha_child = np.zeros((len(w_child), 1))
    for j, idx in enumerate(w_child):
        pos = w_to_pos.get(idx)
        if pos is not None:
            alpha_child[j, 0] = alpha_parent[pos]
        else:
            alpha_child[j, 0] = 0.0
    return alpha_child


def _build_meta_for_child(meta_parent, w_child, transmit_payload):
    """
    Build meta_in for a child node; returns None if parent meta is missing.
    transmit_payload: string or None, supports "primal".
    """
    if meta_parent is None:
        return None
    meta_in = {}
    w_parent = meta_parent.get("w_child", None)
    meta_in["w_parent"] = w_parent

    if transmit_payload == "primal":
        alpha_parent = meta_parent.get("alpha", None)
        alpha_child = _slice_alpha_to_child(alpha_parent, w_parent, w_child)
        if alpha_child is not None:
            meta_in["alpha0"] = alpha_child

    return meta_in

def mainBnB(
    Q,
    y,
    beta,
    method='mosek',
    branch_rule='max_lagrangian_grad',
    traverse_rule='bfs',
    time_limit=None,
    transmit_meta_left=None,
    transmit_meta_right=None,
):
    relErr = 1e-10

    # When transmit_meta_left/right is None, no warm-start info is sent.

    if traverse_rule == 'bfs':
        q = PriorityQueue()
    elif traverse_rule == 'dfs':
        q = LifoQueue()
    else:
        raise ValueError(f"Invalid traverse rule: {traverse_rule}")




    ub = 10e10
    global_ub = ub + 1e-4
    alpha_init, lb, meta_init = solveSVM(Q, y, method, meta_in=None)

    n = Q.shape[0]
    global_supp = np.array([], dtype=np.int32)
    Ssupp = np.array([], dtype=np.int32)
    Psupp = np.arange(n, dtype=np.int32)
    lambda_init = 0  # TODO: lambda_init değişecek
    if transmit_meta_left or transmit_meta_right:
        if meta_init is None:
            meta_init = {}
        meta_init["w_child"] = list(range(n))
    else:
        meta_init = None
    q.put([lb, ub, 0, Ssupp, Psupp, alpha_init, lambda_init, meta_init])

    count = 0
    while q.qsize() >= 1:
        [lb, ub, _, Ssupp, Psupp, alpha1, lambda_val, meta_parent] = q.get()
        print("count: ", count)
        count += 1
        if ub - global_ub < relErr:
            global_ub = ub
            global_supp = Ssupp
            global_alpha = alpha1

        if global_ub <= lb:
            #print("global_ub <= lb")
            break # dfs için breakten continue yapıldı, paperde termination kısmında break var.
        if abs(ub - lb) < relErr:
            print("abs(ub - lb) < relErr")
            continue
        if Psupp.size == 0:
            continue
        if Ssupp.size >= 1:
            ind = branchVariable(alpha1, Q, y, Ssupp, Psupp, branch_rule)
        else:
            # Select the most negative diagonal element
            diag = Q.diagonal()
            bb_ind = np.argmin(diag)
            bb_ind = 0 # TODO: Q'nun submatrisi olacak burası
            ind = Psupp[bb_ind]

        left_supp = np.sort(np.append(Ssupp, ind)).astype(np.int32)
        Psupp = np.delete(Psupp, bb_ind)

        if (Ssupp.size + Psupp.size) >= 1: # Since right_sup = Ssupp + Psupp
            w = np.concatenate((Ssupp, Psupp)).astype(np.int32)
            Qw = Q[w][:, w]
            yw = y[w,0:1]

            meta_in = _build_meta_for_child(meta_parent, w, transmit_meta_right) if transmit_meta_right else None
            alpha_w_opt, right_lb, meta_right = solveSVM(Qw, yw, method, meta_in=meta_in)
            right_lb = right_lb + beta * Ssupp.size

            if transmit_meta_right:
                if meta_right is None:
                    meta_right = {}
                meta_right["w_child"] = w
            else:
                meta_right = None
            q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, alpha1, lambda_val, meta_right]) # TODO: x1 değişecek
        else: # alpha = 0 case
            alpha_w_opt = None # this line just added for clarity
            right_lb = 0 # empty set yields trivial solution alpha = 0, also beta * len(Ssupp) = 0
            q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, alpha1, lambda_val, meta_parent])

        if left_supp.size >= 1:
            w = left_supp
            Qw = Q[w][:, w]
            yw = y[w,0:1]
            meta_in = _build_meta_for_child(meta_parent, w, transmit_meta_left) if transmit_meta_left else None
            alpha_w_opt, left_ub, meta_left = solveSVM(Qw, yw, method, meta_in=meta_in)
            left_ub = left_ub + beta * left_supp.size

            if transmit_meta_left:
                if meta_left is None:
                    meta_left = {}
                meta_left["w_child"] = w
            else:
                meta_left = None
            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, alpha_w_opt, lambda_val, meta_left])

    global_supp = np.array(sorted(global_supp.tolist()), dtype=np.int32)
    #print("count: ", count)
    return global_alpha, global_ub, global_supp, count

# Old implementation (commented)
# def mainBnB(Q, y, beta, method='mosek', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None):
#     relErr = 1e-10
#
#     if traverse_rule == 'bfs':
#         q = PriorityQueue()
#     elif traverse_rule == 'dfs':
#         q = LifoQueue()
#     else:
#         raise ValueError(f"Invalid traverse rule: {traverse_rule}")
#
#     ub = 10e10
#     global_ub = ub + 1e-4
#     alpha_init, lb = solveSVM(Q, y, method)
#
#     global_supp = []
#     Ssupp = []
#     Psupp = list(range(Q.shape[0]))
#     lambda_init = 0 # TODO: lambda_init değişecek
#     q.put([lb,ub,0,Ssupp,Psupp,alpha_init,lambda_init])
#
#     count = 0
#     while q.qsize() >= 1:
#         [lb,ub,_,Ssupp,Psupp,alpha1,lambda_val] = q.get()
#         print("count: ", count)
#
#         count += 1
#         if ub - global_ub < relErr:
#             global_ub = ub
#             global_supp = Ssupp
#             global_alpha = alpha1
#
#         if global_ub <= lb:
#             break # dfs için breakten continue yapıldı, paperde termination kısmında break var.
#         if abs(ub - lb) < relErr:
#             print("abs(ub - lb) < relErr")
#             continue
#         if len(Psupp) == 0:
#             continue
#         else:
#             if len(Ssupp) >= 1:
#                 ind = branchVariable(alpha1, Q, y, Ssupp, Psupp, branch_rule)
#
#             else:
#                 # Select the most negative diagonal element
#                 diag = Q.diagonal()
#                 bb_ind = np.argmin(diag)
#                 bb_ind = 0 # TODO: Q'nun submatrisi olacak burası
#                 ind = Psupp[bb_ind]
#
#             left_supp = sorted(Ssupp + [ind])
#             Psupp.remove(ind)
#             Psupp = sorted(Psupp)
#
#             if len(Ssupp) + len(Psupp) >= 1: # Since right_sup = Ssupp + Psupp
#                 w = Ssupp + Psupp
#                 Qw = Q[w,:][:,w]
#                 yw = y[w,0:1]
#
#                 alpha_w_opt, right_lb = solveSVM(Qw, yw, method)
#                 right_lb = right_lb + beta * len(Ssupp)
#
#                 q.put([right_lb,ub,np.random.rand(),Ssupp,Psupp,alpha1,lambda_val]) # TODO: x1 değişecek
#             else: # alpha = 0 case
#                 alpha_w_opt = None # this line just added for clarity
#                 right_lb = 0 # empty set yields trivial solution alpha = 0, also beta * len(Ssupp) = 0
#                 q.put([right_lb,ub,np.random.rand(),Ssupp,Psupp,alpha1,lambda_val])
#
#             if len(left_supp) >= 1:
#                 w = left_supp
#                 Qw = Q[w,:][:,w]
#                 yw = y[w,0:1]
#                 alpha_w_opt, left_ub = solveSVM(Qw, yw, method)
#                 left_ub = left_ub + beta * len(left_supp)
#
#                 q.put([lb+beta,left_ub,np.random.rand(),left_supp,Psupp,alpha_w_opt,lambda_val])
#
#     global_supp = sorted(global_supp)
#     return global_alpha, global_ub, global_supp, count


def SVMDual_mip(Q, y, beta):
    """
    Solve the sparse SVM dual problem using Gurobi MIP.
    
    min_α  (1/2)α^T Q α - ⟨1,α⟩ + β * |supp(α)|
    s.t.   ⟨y,α⟩ = 0
           α ≥ 0
    
    Parameters
    ----------
    Q : (m, m) array_like
    y : (m,) or (m, 1) array_like
        Label vector.
    beta : float
        Sparsity penalty parameter.
    
    Returns
    -------
    alpha_opt : (m, 1) ndarray
        Optimal dual variables.
    obj_val : float
        Optimal objective value.
    """
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float).flatten()
    m = Q.shape[0]
    
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if len(y) != m:
        raise ValueError("y must have the same length as Q dimensions")
    
    # Big-M value: use a large enough bound for alpha
    M = 100000
    
    model = gp.Model("SVM_Dual_MIP")
    
    # Variables: α ≥ 0 (continuous), z ∈ {0,1} (binary)
    alpha = model.addVars(m, lb=0.0, ub=M, name="alpha")
    z = model.addVars(m, vtype=GRB.BINARY, name="z")
    
    # Quadratic term: (1/2)α^T Q α
    quad_term = gp.quicksum(Q[i, j] * alpha[i] * alpha[j] for i in range(m) for j in range(m))
    
    # Linear term: -⟨1,α⟩ = -sum(alpha[i])
    linear_term = -gp.quicksum(alpha[i] for i in range(m))
    
    # Sparsity penalty: β * sum(z[i])
    sparsity_term = beta * gp.quicksum(z[i] for i in range(m))
    #sparsity_term = beta * np.ones((m,1)).T @ z
    
    model.setObjective(0.5 * quad_term + linear_term + sparsity_term, GRB.MINIMIZE)
    
    # Constraint: ⟨y,α⟩ = 0
    model.addConstr(gp.quicksum(float(y[i]) * alpha[i] for i in range(m)) == 0.0, name="equality")
    
    # Linking constraints between alpha and z
    # - Upper big-M: alpha[i] <= M * z[i]
    # - Indicator: if z[i] = 0 then alpha[i] <= 0  (with alpha >= 0 ⇒ alpha = 0)
    # - Lower epsilon: alpha[i] >= eps * z[i] (avoid z[i]=1 with tiny alpha)
    eps_link = 1e-6
    for i in range(m):
        model.addConstr(alpha[i] <= M * z[i], name=f"bigM_{i}")
        #model.addGenConstrIndicator(z[i], 0, alpha[i], GRB.LESS_EQUAL, 0.0, name=f"ind_zero_{i}")
        #model.addConstr(alpha[i] >= eps_link * z[i], name=f"eps_link_{i}")
    
    # Solver parameters for higher precision on integrality and numerics
    model.setParam('OutputFlag', 0)
    model.setParam('IntFeasTol', 1e-8)
    #model.setParam('MIPGap', 0)
    #model.setParam('MIPGapAbs', 0)
    #model.setParam('NumericFocus', 3)
    #model.setParam('IntegralityFocus', 1)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        alpha_opt = np.array([[alpha[i].x] for i in range(m)])
        z_opt = np.array([z[i].x for i in range(m)])
        print("z_opt: ", z_opt)
        
        return alpha_opt, model.objVal
    else:
        return np.zeros((m, 1)), np.inf


def generate_SVM_test_case(m=10, d=5, beta=0.1, seed=42, normalize=True):
    """
    Generate a test case for SVM dual problem.
    
    Parameters
    ----------
    m : int, optional
        Number of samples. Default is 10.
    d : int, optional
        Feature dimension. Default is 5.
    beta : float, optional
        Sparsity penalty parameter. Default is 0.1.
    seed : int, optional
        Random seed. Default is 42.
    normalize : bool, optional
        If True, standardize features to zero mean and unit variance (default: True).
        This is standard practice for SVM preprocessing.
    
    Returns
    -------
    Q : (m, m) ndarray
        Kernel matrix Q where Q[i,j] = y_i * y_j * <x_i, x_j>.
    y : (m,) ndarray
        Label vector.
    beta : float
        Sparsity penalty parameter.
    """
    np.random.seed(seed)
    
    # Generate random features between -1 and 1
    X = np.random.uniform(-1, 1, size=(m, d)) * 10
    
    # Feature normalization (standard practice for SVM)
    if normalize:
        X = normalize_features(X, method='standardize')
    
    # Generate random labels (-1 or +1)
    y = np.random.choice([-1, 1], size=m)
    
    # Build Q matrix using dataset_to_Q function
    Q, y_vec = dataset_to_Q(X, y)
    
    return Q, y_vec, beta


def solve_SVM_both_methods(Q, y, beta, method='mosek'):
    """
    Solve the sparse SVM dual problem using both MIP and BnB methods.
    
    Parameters
    ----------
    Q : (m, m) array_like
        Kernel matrix.
    y : (m,) or (m, 1) array_like
        Label vector.
    beta : float
        Sparsity penalty parameter.
    
    Returns
    -------
    alpha_mip : (m, 1) ndarray
        Optimal solution from MIP method.
    obj_mip : float
        Objective value from MIP method.
    alpha_bnb : (m, 1) ndarray
        Optimal solution from BnB method (reconstructed to full size).
    obj_bnb : float
        Objective value from BnB method.
    """
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(y, dtype=float)
    m = Q.shape[0]
    
    # Solve with MIP
    alpha_mip, obj_mip = SVMDual_mip(Q, y, beta)
    
    # Solve with BnB
    # Note: mainBnB expects y to be 2D based on y[w,0:1] usage
    y_2d = y.reshape(-1, 1) if len(y.shape) == 1 else y
    alpha_bnb_supp, obj_bnb, supp_bnb, _ = mainBnB(Q, y_2d, beta, method=method)
    
    # Reconstruct full alpha vector from BnB solution
    alpha_bnb = zeropadding(alpha_bnb_supp, supp_bnb, m)
    
    return alpha_mip, obj_mip, alpha_bnb, obj_bnb


def compute_SVM_objective(alpha, Q, y, beta):
    """
    Compute the objective value for the sparse SVM dual problem.
    
    (1/2)α^T Q α - ⟨1,α⟩ + β * |supp(α)|
    
    Parameters
    ----------
    alpha : (m, 1) array_like
        Dual variables.
    Q : (m, m) array_like
        Kernel matrix.
    y : (m,) or (m, 1) array_like
        Label vector (not used in objective, but kept for consistency).
    beta : float
        Sparsity penalty.
    
    Returns
    -------
    obj_val : float
        Objective value.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1, 1)
    #print("alpha: ", alpha)
    Q = np.asarray(Q, dtype=float)
    
    # Replace values less than 1e-6 with zero
    alpha = np.where(np.abs(alpha) < 1e-6, 0, alpha)
    
    quad_term = 0.5 * (alpha.T @ Q @ alpha)[0, 0]
    linear_term = -np.sum(alpha)
    sparsity_term = beta * np.count_nonzero(alpha)
    
    return quad_term + linear_term + sparsity_term


def calculate_hyperplane(X, y, alpha, tol=1e-6):
    """
    Compute the SVM hyperplane parameters (w, b) from the dual solution (hard-margin case).

    Parameters
    ----------
    X : ndarray of shape (m, n)
        Training data matrix where each row is a feature vector x_i.
    y : ndarray of shape (m,)
        Labels (+1 or -1) for each training example.
    alpha : ndarray of shape (m,)
        Dual coefficients obtained from the SVM dual optimization.
    tol : float, optional
        Numerical tolerance to decide if alpha_i > 0.

    Returns
    -------
    w : ndarray of shape (n,)
        Weight vector defining the separating hyperplane.
    b : float
        Bias term (intercept) of the hyperplane.
    support_indices : ndarray
        Indices of support vectors.
    """

    # Compute w = Σ_i α_i y_i x_i
    w = np.sum((alpha * y)[:, np.newaxis] * X, axis=0)

    # Identify support vectors (α_i > 0)
    support_indices = np.where(alpha > tol)[0]

    if len(support_indices) == 0:
        raise ValueError("No valid support vectors found. Check α or tolerance.")

    # Compute b = y_i - wᵀx_i for each support vector
    b_values = [y[i] - np.dot(w, X[i]) for i in support_indices]
    b = np.mean(b_values)

    return w, b, support_indices


def train_and_test_svm(X_train, X_test, y_train, y_test, beta, method='mosek', 
                       branch_rule='max_lagrangian_grad', traverse_rule='bfs', 
                       time_limit=None, verbose=True):
    """
    Train SVM using Branch and Bound and evaluate on test set.
    
    Parameters
    ----------
    X_train : (m_train, d) ndarray
        Training feature matrix.
    X_test : (m_test, d) ndarray
        Test feature matrix.
    y_train : (m_train,) ndarray
        Training labels (should be in {-1, +1}).
    y_test : (m_test,) ndarray
        Test labels (should be in {-1, +1}).
    beta : float
        Sparsity penalty parameter for the sparse SVM dual problem.
    method : str, optional
        Solver method for subproblems: 'mosek' or 'cvxpy' (default: 'mosek').
    branch_rule : str, optional
        Branching rule for BnB (default: 'max_lagrangian_grad').
    traverse_rule : str, optional
        Traversal rule: 'bfs' or 'dfs' (default: 'bfs').
    time_limit : float, optional
        Time limit in seconds (default: None).
    verbose : bool, optional
        If True, print progress information (default: True).
    
    Returns
    -------
    train_accuracy : float
        Classification accuracy on training set.
    test_accuracy : float
        Classification accuracy on test set.
    w : (d,) ndarray
        Weight vector of the hyperplane.
    b : float
        Bias term of the hyperplane.
    alpha : (m_train,) ndarray
        Dual coefficients (full vector, padded with zeros).
    support_indices : ndarray
        Indices of support vectors.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    
    # Ensure labels are in {-1, +1}
    if not np.all(np.isin(y_train, [-1, 1])):
        # Map 0/1 to -1/+1 if needed
        y_train = np.where(y_train == 0, -1, 1)
    if not np.all(np.isin(y_test, [-1, 1])):
        y_test = np.where(y_test == 0, -1, 1)
    
    if verbose:
        print(f"Training SVM with {m_train} samples, {X_train.shape[1]} features")
        print(f"Beta (sparsity penalty): {beta}")
    
    # Build Q matrix from training data
    Q, y_train_vec = dataset_to_Q(X_train, y_train)
    #y_train_vec = y_train_vec.flatten()  # Ensure 1D array for mainBnB
    
    # Solve sparse SVM dual problem using Branch and Bound
    if verbose:
        print("Running Branch and Bound...")
    alpha_supp, obj_val, support_indices, count = mainBnB(
        Q, y_train_vec, beta, method=method, 
        branch_rule=branch_rule, traverse_rule=traverse_rule, 
        time_limit=time_limit
    )
    
    # Pad alpha to full size (alpha_supp is only on support)
    alpha = zeropadding(alpha_supp, support_indices, m_train).flatten()
    
    if verbose:
        print(f"Found {len(support_indices)} support vectors")
        print(f"Objective value: {obj_val:.6f}")
    
    # Compute hyperplane parameters (w, b)
    w, b, support_indices_hyperplane = calculate_hyperplane(X_train, y_train, alpha)
    
    # Classification function: sign(w^T x + b)
    def classify(X):
        """Classify samples using the hyperplane."""
        scores = X @ w + b
        return np.sign(scores)
    
    # Predict on training set
    y_train_pred = classify(X_train)
    train_accuracy = np.mean(y_train_pred == y_train)
    
    # Predict on test set
    y_test_pred = classify(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    
    if verbose:
        print(f"Train accuracy: {train_accuracy:.4f} ({np.sum(y_train_pred == y_train)}/{m_train})")
        print(f"Test accuracy: {test_accuracy:.4f} ({np.sum(y_test_pred == y_test)}/{m_test})")
    
    return train_accuracy, test_accuracy, w, b, alpha, support_indices_hyperplane



if __name__ == "__main__":
    # Example 1: synthetic instance
    #Q, y, beta = generate_SVM_test_case(m=10, d=5, beta=0.5, seed=2)

    X_hd, y_hd = getDataset_breastcancerwisconsin(normalize=True)

    X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(X_hd, y_hd, test_size=0.2, random_state=42)
    print("y_hd_train: ", y_hd_train.shape)
    y_hd_train = y_hd_train.reshape(-1, 1)
    y_hd_test = y_hd_test.reshape(-1, 1)
    print("y_hd_train: ", y_hd_train.shape)
    train_accuracy, test_accuracy, w, b, alpha, support_indices_hyperplane = train_and_test_svm(X_hd_train, X_hd_test, y_hd_train, y_hd_test, beta=0.1, method='cvxpy')


    """

    # Example 2: real dataset (Breast Cancer Wisconsin)
    X_bc, y_bc = getDataset_breastcancerwisconsin()
    Q, y = dataset_to_Q(X_bc, y_bc)
    beta = 0.1
    
    # Solve with both methods
    #alpha_mip, obj_mip, alpha_bnb, obj_bnb = solve_SVM_both_methods(Q, y, beta, method='cvxpy')

    
    alpha_bnb, obj_bnb, supp_bnb, _ = mainBnB(Q, y, beta, method='cvxpy')
    alpha_bnb = zeropadding(alpha_bnb, supp_bnb, Q.shape[0])
    
    # Print results
    #print("MIP Solution:")
    #print(alpha_mip)
    #print(f"MIP Objective: {obj_mip}")
    #print(f"MIP Objective: {compute_SVM_objective(alpha_mip, Q, y, beta)}")
    #print("MIP Equality: ", np.dot(y.T, alpha_mip))
    print("\nBnB Solution:")
    print(alpha_bnb)
    print(f"BnB Objective: {obj_bnb}")
    print(f"BnB Objective: {compute_SVM_objective(alpha_bnb, Q, y, beta)}")
    print("BnB Equality: ", np.dot(y.T, alpha_bnb))


     # test case


    w = [4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 18, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568]
    Qw = Q[w,:][:,w]
    yw = y[w,0:1]

    # Check symmetry
    is_sym, max_asym = is_symmetric(Qw)
    print(f"Qw is symmetric: {is_sym}")
    if not is_sym:
        print(f"Max asymmetry: {max_asym}")
        # Symmetrize
        Qw = 0.5 * (Qw + Qw.T)
        print("Qw has been symmetrized")
    
    eigvals = np.linalg.eigvalsh(Qw)
    print("cond(Q) ~", eigvals[-1]/eigvals[0])
    min_eigval = np.min(eigvals)
    print(f"min_eigval: {min_eigval}")
    
    # Ensure PSD
    if min_eigval < 1e-10:
        reg = max(1e-10 - min_eigval, 1e-10)
        Qw = Qw + np.eye(Qw.shape[0]) * reg
        print(f"Added regularization: {reg}, new min_eigval: {np.min(np.linalg.eigvalsh(Qw))}")

    Qw = 0.5 * (Qw + Qw.T)

    mu = 1e-1 * np.eye(len(w))
    Qw = Qw + mu

    alpha_w_opt, right_lb, _ = solveSVM(Qw, yw, "cvxpy")

    print("right_lb: ", right_lb)
    """

    

    