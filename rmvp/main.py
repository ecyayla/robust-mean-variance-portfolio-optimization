import numpy as np
import pandas as pd
import cvxpy as cp
from queue import PriorityQueue, LifoQueue
import time
import gurobipy as gp
from gurobipy import GRB
import math



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



def solveRMVP1(D, tau_bar, tau, gamma, dual=True):
    """
    Closed-form optimizer for RMVP1 (robust mean-variance, model (2) in paper)
    Implements Proposition: The unique optimal solution is
        u_hat = (tau_bar / (H_omega * (H_omega - gamma))) * (D_omega)^{-1} * tau_omega
    where H_omega = sqrt(tau_omega^T * (D_omega)^{-1} * tau_omega).
    
    Assumes Slater condition is satisfied and D ≻ 0. Allows short sales.

    Parameters
    ----------
    D : (m, m) array_like
        Positive semidefinite symmetric matrix D (or D_omega for support omega).
    tau_bar : float
        Scalar parameter tau_bar (bar{ccr} in the proposition).
    tau : (m,) array_like
        The excess mean return vector estimate tau (ccr_omega in the proposition).
    gamma : float
        Ellipsoidal uncertainty radius γ.

    Returns
    -------
    u_hat : (m,) ndarray
        Optimal solution u_hat.

    Raises
    ------
    ValueError
        If H_omega ≤ γ (infeasible or degenerate) or if matrix is not PD.
    """
    #D = np.asarray(D, dtype=float)
    #tau = np.asarray(tau, dtype=float)


    # Precompute D^{-1}
    D_inv = np.linalg.inv(D)
    
    # Compute H_omega = sqrt(tau^T * D^{-1} * tau)
    H = np.sqrt(tau.T @ D_inv @ tau)

    # Compute scale factor: tau_bar / (H_omega * (H_omega - gamma))
    scale = tau_bar / (H * (H - gamma))
    # Compute optimal solution: u_hat = scale * (D^{-1} @ tau)
    u_hat = scale * (D_inv @ tau)
    obj_val = float(u_hat.T @ D @ u_hat)

    if dual:
        lambda_val = 2 * tau_bar / ((H - gamma)**2)
    else:
        lambda_val = 0
    return u_hat, obj_val, lambda_val



def rmvp1_lagrangian_gradient(x_s, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val):
    """
    Gradient of RMVP1 Lagrangian w.r.t. the still-zero coordinates (Psupp).
    
    For problem: min x^T D x  s.t.  tau^T x - gamma * ||x||_D >= tau_bar
    
    The gradient on Psupp is:
        ∇_Z L = 2 D_{Z,S} x_S - λ * (tau_Z - gamma * (D_{Z,S} x_S) / ||x_S||_D)
    where:
        λ = 2 * tau_bar / ((H - γ)^2),
        H = sqrt(tau_S^T D_SS^{-1} tau_S),
        ||x_S||_D = sqrt(x_S^T D_SS x_S)

    Parameters
    ----------
    x_s    : (|S|,1) current weights on active support S
    D      : (n,n)   positive semidefinite symmetric matrix D
    tau    : (n,)    excess mean return vector estimate tau
    tau_bar: float   scalar parameter tau_bar
    gamma  : float   uncertainty radius
    Ssupp  : array   active support indices
    Psupp  : array   still-zero support indices
    lambda_val: float   Lagrange multiplier
    Returns
    -------
    grad_z : (|Psupp|,1) gradient of L on Psupp
    """

    S = np.array(Ssupp, dtype=int)
    Z = np.array(Psupp, dtype=int)

    # Extract blocks
    D_SS = D[np.ix_(S, S)]
    D_ZS = D[np.ix_(Z, S)]
    tau_S = tau[S]
    tau_Z = tau[Z]

    # Compute H_omega = sqrt(tau_S^T D_SS^{-1} tau_S)
    y = np.linalg.solve(D_SS, tau_S)
    H = float(np.sqrt(tau_S @ y))
    
    # Compute ||x_S||_D = sqrt(x_S^T D_SS x_S) and D_ZS x_S
    DZS_xs = D_ZS @ x_s
    norm_x_D = float(np.sqrt(x_s.T @ (D_SS @ x_s)))
    
    if norm_x_D < 1e-8:
        norm_x_D = 1e-8  # Avoid division by zero

    # Gradient on Psupp: ∇_Z L = 2 D_{Z,S} x_S - λ * (tau_Z - gamma * (D_{Z,S} x_S) / ||x_S||_D)
    grad_z = 2 * DZS_xs - lambda_val * (tau_Z.reshape(-1, 1) - gamma * (DZS_xs / norm_x_D))
    return grad_z



def branchVariable_rmvp1(x, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule='max_lagrangian_grad'):
    """
    Selects a variable to branch on based on the branching rule.
    
    Parameters
    ----------
    x         : (n,1) current solution
    D         : (n,n) positive semidefinite symmetric matrix D
    tau       : (n,)  excess mean return vector estimate tau
    tau_bar   : float scalar parameter tau_bar
    gamma     : float uncertainty radius
    Ssupp     : array active support indices
    Psupp     : array still-zero support indices
    branch_rule: str  branching rule ('max_lagrangian_grad')
    
    Returns
    -------
    ind : int index of variable to branch on
    """
    if branch_rule == 'max_lagrangian_grad':
        grad = rmvp1_lagrangian_gradient(x, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val)
        bb_ind = np.argmax(abs(grad[:,0]))
        ind = Psupp[bb_ind]
    else:
        raise ValueError(f"Invalid branch rule: {branch_rule}")
    
    return ind, bb_ind




def mainRMVP1BnB(D, tau, tau_bar, gamma, beta, method='auto', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None, collect_collapse=True, enable_collapse=True):
    """
    Branch-and-bound algorithm for CP-RMVP problem:
        min_{x in R^n} x^T D x  s.t.  tau^T x - gamma * ||x||_D >= tau_bar

    If collect_collapse=True, also returns two lists (collapse_S, collapse_P) recording
    |S| and |P| at every right-subtree collapse; the return becomes a 6-tuple. Default
    False keeps the historical 4-tuple return for existing callers.
    """
    np.random.seed(42)
    relErr = 1e-8
    collapse_S, collapse_P = [], []   # |S|, |P| at each right-subtree collapse

    if traverse_rule == 'bfs':
        q = PriorityQueue()
    elif traverse_rule == 'dfs':
        q = LifoQueue()
    else:
        raise ValueError(f"Invalid traverse rule: {traverse_rule}")

    ub = 10e10
    global_ub = ub + 1e2

    x_init, lb, lambda_init = solveRMVP1(D, tau_bar, tau, gamma)

    n = D.shape[1]
    global_supp = np.array([], dtype=np.int32)
    Ssupp = np.array([], dtype=np.int32)
    Psupp = np.arange(n, dtype=np.int32)
    DD = np.diag(D)

    q.put([lb, ub, 0, Ssupp, Psupp, x_init, lambda_init])

    count = 0
    while q.qsize() >= 1:
        [lb, ub, _, Ssupp, Psupp, x1, lambda_val] = q.get()
        count += 1

        if ub - global_ub < relErr:
            global_ub = ub
            global_supp = Ssupp
            global_x = x1

        if global_ub - lb <= relErr:
            #print("global_ub <= lb")
            break
        if Psupp.size == 0:
            continue

        if Ssupp.size >= 1:
            ind, bb_ind = branchVariable_rmvp1(x1, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule)
        else:
            tau_Psupp = tau[Psupp]
            var = DD[Psupp]
            bb_dec = var / (tau_Psupp + 1e-8)
            bb_ind = int(np.argmin(bb_dec))
            ind = int(Psupp[bb_ind])

        left_supp = np.sort(np.append(Ssupp, ind)).astype(np.int32)
        Psupp = np.delete(Psupp, bb_ind)

        if (Ssupp.size + Psupp.size) >= 1:
            w = np.concatenate((Ssupp, Psupp)).astype(np.int32)
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, right_lb, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
            right_lb = right_lb + beta * Ssupp.size

            if (right_lb + beta >= global_ub - relErr) and enable_collapse:          # lb^R + beta >= ub* - eps
                collapse_S.append(int(Ssupp.size))             # record |S|, |P| at collapse
                collapse_P.append(int(Psupp.size))
            else:
                q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, x1, lambda_val])  # Enqueue (x, lb^R, ub, Psupp, S)


        if left_supp.size >= 1:
            w = left_supp
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, left_ub, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
            left_ub = left_ub + beta * left_supp.size

            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, x_w_opt, lambda_val])

    global_supp = np.array(sorted(global_supp.tolist()), dtype=np.int32)
    if collect_collapse:
        return global_x, global_ub, global_supp, count, collapse_S, collapse_P
    return global_x, global_ub, global_supp, count




def RMVP1_mipGUROBI(D, tau, tau_bar, gamma, beta, threads=1):
    """
    Mixed-integer programming formulation for CP-RMVP problem:
        min_{x in R^n} x^T D x + beta * ||x||_0  s.t.  tau^T x - gamma * ||x||_D >= tau_bar
    
    Parameters
    ----------
    D : (n, n) array_like
        Positive semidefinite symmetric matrix D.
    tau : (n,) array_like
        The excess mean return vector estimate tau.
    tau_bar : float
        Scalar parameter tau_bar.
    gamma : float
        Ellipsoidal uncertainty radius γ.
    beta : float
        Sparsity penalty parameter.
    
    Returns
    -------
    x_opt : (n, 1) ndarray
        Optimal solution.
    obj_val : float
        Optimal objective value.
    """
    n = D.shape[0]
    # Big-M value for sparsity constraints
    M = 1e1  # Large enough bound for x

    model = gp.Model("RMVP1_mipGUROBI")

    # Decision variables
    x = model.addVars(n, lb=-M, ub=M, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # Auxiliary variable for ||x||_D = sqrt(x^T D x)
    t = model.addVar(lb=0, name="t")

    # Objective: x^T D x + beta * sum(z_i)
    quad_term = gp.quicksum(D[i,j] * x[i] * x[j] for i in range(n) for j in range(n))
    sparsity_term = beta * gp.quicksum(z[i] for i in range(n))
    model.setObjective(quad_term + sparsity_term, GRB.MINIMIZE)

    # Constraint: tau^T x - gamma * ||x||_D >= tau_bar
    # This is equivalent to: tau^T x - gamma * t >= tau_bar, where t >= sqrt(x^T D x)
    linear_part = gp.quicksum(tau[i] * x[i] for i in range(n))
    model.addConstr(linear_part - gamma * t >= tau_bar, name="robust_constraint")
    
    # Constraint: t >= sqrt(x^T D x), which is equivalent to: x^T D x <= t^2
    # This is a rotated second-order cone constraint
    # In Gurobi, we model this as: x^T D x - t^2 <= 0
    # Build the quadratic expression for x^T D x
    quad_expr = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            quad_expr += D[i,j] * x[i] * x[j]
    
    # Build quadratic expression for t^2
    #t_sq_expr = gp.QuadExpr()
    #t_sq_expr += t * t
    
    # Add constraint: x^T D x <= t^2, i.e., x^T D x - t^2 <= 0
    model.addQConstr(quad_expr - t * t <= 0, name="norm_constraint")

    # Sparsity constraints: |x_i| <= M * z_i
    model.addConstrs(x[i] <= M * z[i] for i in range(n))
    model.addConstrs(x[i] >= -M * z[i] for i in range(n))

    # Tighter tolerances for higher precision
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', threads)
    model.setParam('IntFeasTol', 1e-8)
    model.setParam('FeasibilityTol', 1e-8)
    model.setParam('OptimalityTol', 1e-8)
    model.setParam('MIPGap', 1e-8)
    model.setParam('MIPGapAbs', 1e-8)
    # Time limit (seconds) ~ 1 hour
    model.setParam('TimeLimit', 43200)

    model.optimize()
    print("MIP GUROBI status: ", model.status)

    if model.status == GRB.OPTIMAL:
        x_opt = np.array([[x[i].x] for i in range(n)])
        z_opt = np.array([z[i].x for i in range(n)])
        return x_opt, model.objVal, model.MIPGap
    elif model.SolCount > 0:
        # Return incumbent with its gap if not proven optimal
        x_opt = np.array([[x[i].X] for i in range(n)])
        return x_opt, model.objVal, model.MIPGap
    else:
        return np.zeros((n, 1)), np.inf, np.inf

    
    


def solveRMVP2(D, tau_bar, tau, gamma, beta, t, dual=True):
    """
    Closed-form optimizer for RMVP2 (maximization model with D-norm constraint).

    Model on a given support (omega):
        max_x   tau_omega^T x - tau_bar - gamma * ||x||_D + beta * ||x||_0
        s.t.    ||x||_D <= t

    Under the assumptions in the paper the optimal solution on support omega is
        x_hat = (t / H_omega) * D_omega^{-1} * tau_omega
    where
        H_omega = sqrt(tau_omega^T * D_omega^{-1} * tau_omega).

    Parameters
    ----------
    D : (m, m) array_like
        Positive definite symmetric matrix (or submatrix D_omega).
    tau_bar : float
        Scalar parameter bar{ccr} (appears as a constant shift in the objective).
    tau : (m,) array_like
        Excess mean return vector on the current support tau_omega.
    gamma : float
        Uncertainty radius γ.
    beta : float
        Sparsity parameter (used here only in the objective value).
    t : float
        D-norm radius in the constraint ||x||_D <= t.
    dual : bool, optional
        If True, also return a placeholder Lagrange multiplier.

    Returns
    -------
    x_hat : (m, 1) ndarray
        Optimal solution on the given support.
    obj_val : float
        Objective value at x_hat.
    lambda_val : float
        Placeholder dual / Lagrange multiplier (0 if dual=False).
    """
    # Invert D
    D_inv = np.linalg.inv(D)

    # Compute H_omega = sqrt(tau^T D^{-1} tau)
    H = float(np.sqrt(tau.T @ D_inv @ tau))

    # Closed-form optimizer: x_hat = (t / H) * D^{-1} tau
    x_vec = (t / H) * (D_inv @ tau)
    x_hat = x_vec.reshape(-1, 1)

    # Compute objective value: tau^T x - tau_bar - gamma ||x||_D + beta ||x||_0
    quad = x_hat.T @ D @ x_hat
    norm_D = np.sqrt(quad)

    obj_val = tau.T @ x_hat - tau_bar - gamma * norm_D

    if dual:
        lambda_val = H - gamma
    else:
        lambda_val = 0.0

    return x_hat, -obj_val, lambda_val

def rmvp2_lagrangian_gradient(x_s, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val):
    """
    Gradient of RMVP2 Lagrangian w.r.t. the still-zero coordinates (Psupp).
    
    For problem: max_{x in R^N} tau^T x - tau_bar - gamma * ||x||_D  s.t. ||x||_D <= t
    
    The Lagrangian is:
        L(x, lambda) = tau^T x - tau_bar - gamma * ||x||_D - lambda * (||x||_D - t)
                     = tau^T x - tau_bar - (gamma + lambda) * ||x||_D + lambda * t
    
    The gradient w.r.t. x is:
        ∇_x L = tau - (gamma + lambda) * (D x) / ||x||_D
    
    For the still-zero coordinates (Psupp), where x_Z = 0:
        ∇_Z L = tau_Z - (gamma + lambda) * (D_{Z,S} x_S) / ||x_S||_D
    where:
        ||x_S||_D = sqrt(x_S^T D_{SS} x_S)

    Parameters
    ----------
    x_s       : (|S|,1) current weights on active support S
    D         : (n,n) positive semidefinite symmetric matrix D
    tau       : (n,)  excess mean return vector estimate tau
    tau_bar   : float scalar parameter tau_bar (not used in gradient, but kept for consistency)
    gamma     : float uncertainty radius
    Ssupp     : array active support indices
    Psupp     : array still-zero support indices
    lambda_val: float Lagrange multiplier for constraint ||x||_D <= t
    Returns
    -------
    grad_z : (|Psupp|,1) gradient of L on Psupp
    """
    S = np.array(Ssupp, dtype=int)
    Z = np.array(Psupp, dtype=int)
    

    # Extract blocks
    D_SS = D[np.ix_(S, S)]
    D_ZS = D[np.ix_(Z, S)]
    tau_Z = tau[Z]
    
    # Compute ||x_S||_D = sqrt(x_S^T D_SS x_S) and D_ZS x_S
    DZS_xs = D_ZS @ x_s
    norm_x_D = np.sqrt(x_s.T @ (D_SS @ x_s))
    
    if norm_x_D < 1e-8:
        norm_x_D = 1e-8  # Avoid division by zero

    # Gradient on Psupp: ∇_Z L = tau_Z - (gamma + lambda) * (D_{Z,S} x_S) / ||x_S||_D
    grad_z = tau_Z.reshape(-1, 1) - (gamma + lambda_val) * (DZS_xs / norm_x_D)
    return grad_z


def branchVariable_rmvp2(x, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule='max_lagrangian_grad'):
    """
    Selects a variable to branch on based on the branching rule for RMVP2.
    
    Parameters
    ----------
    x         : (n,1) current solution
    D         : (n,n) positive semidefinite symmetric matrix D
    tau       : (n,)  excess mean return vector estimate tau
    tau_bar   : float scalar parameter tau_bar
    gamma     : float uncertainty radius
    Ssupp     : array active support indices
    Psupp     : array still-zero support indices
    lambda_val: float Lagrange multiplier
    branch_rule: str  branching rule ('max_lagrangian_grad')
    
    Returns
    -------
    ind : int index of variable to branch on
    """
    if branch_rule == 'max_lagrangian_grad':
        grad = rmvp2_lagrangian_gradient(x, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val)
        bb_ind = np.argmax(abs(grad[:,0]))
        ind = Psupp[bb_ind]
    else:
        raise ValueError(f"Invalid branch rule: {branch_rule}")
    
    return ind, bb_ind


def mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t, method='auto', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None, collect_collapse=True, enable_collapse=True):
    """
    Branch-and-bound algorithm for CP-RMVP2 problem (same structure as mainRMVP1BnB).

    If collect_collapse=True, also returns two lists (collapse_S, collapse_P) recording
    |S| and |P| at every right-subtree collapse; the return becomes a 6-tuple. Default
    False keeps the historical 4-tuple return for existing callers.
    """
    np.random.seed(42)
    relErr = 1e-8
    collapse_S, collapse_P = [], []   # |S|, |P| at each right-subtree collapse

    if traverse_rule == 'bfs':
        q = PriorityQueue()
    elif traverse_rule == 'dfs':
        q = LifoQueue()
    else:
        raise ValueError(f"Invalid traverse rule: {traverse_rule}")

    ub = 10e10
    global_ub = ub + 1e-4

    x_init, lb, lambda_init = solveRMVP2(D, tau_bar, tau, gamma, beta, t)

    n = D.shape[1]
    global_supp = np.array([], dtype=np.int32)
    Ssupp = np.array([], dtype=np.int32)
    Psupp = np.arange(n, dtype=np.int32)
    DD = np.diag(D)

    q.put([lb, ub, 0, Ssupp, Psupp, x_init, lambda_init])

    count = 0
    while q.qsize() >= 1:
        [lb, ub, _, Ssupp, Psupp, x1, lambda_val] = q.get()

        count += 1
        if ub - global_ub < relErr:
            global_ub = ub
            global_supp = Ssupp
            global_x = x1

        if global_ub - lb <= relErr:
            #print("global_ub <= lb")
            break
        if Psupp.size == 0:
            continue

        if Ssupp.size >= 1:
            ind, bb_ind = branchVariable_rmvp2(x1, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule)
        else:
            tau_Psupp = tau[Psupp]
            var = DD[Psupp]
            bb_dec = var / (tau_Psupp + 1e-8)
            bb_ind = int(np.argmin(bb_dec))
            ind = int(Psupp[bb_ind])

        left_supp = np.sort(np.append(Ssupp, ind)).astype(np.int32)
        Psupp = np.delete(Psupp, bb_ind)

        if (Ssupp.size + Psupp.size) >= 1:  # Since right_sup = Ssupp + Psupp
            w = np.concatenate((Ssupp, Psupp)).astype(np.int32)
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, right_lb, lambda_val = solveRMVP2(D_w, tau_bar, tau_w, gamma, beta, t)
            right_lb = right_lb + beta * Ssupp.size
            
            if (right_lb + beta >= global_ub - relErr) and enable_collapse:          # lb^R + beta >= ub* - eps
                collapse_S.append(int(Ssupp.size))             # record |S|, |P| at collapse
                collapse_P.append(int(Psupp.size))
            else:
                q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, x1, lambda_val])  # Enqueue (x, lb^R, ub, Psupp, S)

        if left_supp.size >= 1:
            w = left_supp
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, left_ub, lambda_val = solveRMVP2(D_w, tau_bar, tau_w, gamma, beta, t)
            left_ub = left_ub + beta * left_supp.size

            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, x_w_opt, lambda_val])

    global_supp = np.array(sorted(global_supp.tolist()), dtype=np.int32)
    if collect_collapse:
        return global_x, global_ub, global_supp, count, collapse_S, collapse_P
    return global_x, global_ub, global_supp, count


def RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta, t, threads=1):
    """
    Mixed-integer programming formulation for CP-RMVP2 problem:
        min_{x in R^N} -tau^T x + tau_bar + gamma * ||x||_D + beta * ||x||_0  s.t. ||x||_D <= t
    
    Parameters
    ----------
    D : (n, n) array_like
        Positive semidefinite symmetric matrix D.
    tau : (n,) array_like
        The excess mean return vector estimate tau.
    tau_bar : float
        Scalar parameter tau_bar.
    gamma : float
        Ellipsoidal uncertainty radius γ.
    beta : float
        Sparsity penalty parameter.
    t : float
        D-norm constraint bound.
    
    Returns
    -------
    x_opt : (n, 1) ndarray
        Optimal solution.
    obj_val : float
        Optimal objective value.
    """
    n = D.shape[0]
    # Big-M value for sparsity constraints
    M = 1e2
    
    model = gp.Model("RMVP2_mipGUROBI")
    
    # Decision variables
    x = model.addVars(n, lb=-M, ub=M, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # Auxiliary variable for ||x||_D = sqrt(x^T D x)
    s = model.addVar(lb=0, name="s")
    
    # Objective: min -tau^T x + tau_bar + gamma * ||x||_D + beta * sum(z_i)
    # Note: beta * ||x||_0 is a reward in maximization, so we add it
    linear_part = gp.quicksum(tau[i] * x[i] for i in range(n))
    sparsity_term = beta * gp.quicksum(z[i] for i in range(n))
    model.setObjective(-linear_part + tau_bar + gamma * s + sparsity_term, GRB.MINIMIZE)
    
    # Constraint: ||x||_D <= t, which is equivalent to: x^T D x <= t^2
    # This is a rotated second-order cone constraint
    # In Gurobi, we model this as: x^T D x - t^2 <= 0
    # But we also need s >= sqrt(x^T D x), so we use: x^T D x <= s^2 and s <= t
    # Build the quadratic expression for x^T D x
    quad_term = gp.quicksum(D[i,j] * x[i] * x[j] for i in range(n) for j in range(n))

    # Constraint: x^T D x <= s^2, i.e., x^T D x - s^2 <= 0
    model.addQConstr(quad_term - s * s <= 0, name="norm_constraint")
    
    # Constraint: s <= t (since ||x||_D <= t)
    model.addConstr(s <= t, name="bound_constraint")
    
    # Sparsity constraints: |x_i| <= M * z_i
    model.addConstrs(x[i] <= M * z[i] for i in range(n))
    model.addConstrs(x[i] >= -M * z[i] for i in range(n))
    
    # Tighter tolerances for higher precision
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', threads)
    model.setParam('IntFeasTol', 1e-8)
    model.setParam('FeasibilityTol', 1e-8)
    model.setParam('OptimalityTol', 1e-8)
    model.setParam('MIPGap', 1e-8)
    model.setParam('MIPGapAbs', 1e-8)
    # Time limit (seconds) ~ 1 hour
    model.setParam('TimeLimit', 43200)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        x_opt = np.array([[x[i].x] for i in range(n)])
        # Post-process: set near-zero entries to exactly zero
        x_opt[np.abs(x_opt) < 1e-8] = 0.0
        return x_opt, model.objVal, model.MIPGap
    elif model.SolCount > 0:
        x_opt = np.array([[x[i].X] for i in range(n)])
        x_opt[np.abs(x_opt) < 1e-8] = 0.0
        return x_opt, model.objVal, model.MIPGap
    else:
        return np.zeros((n, 1)), np.inf, np.inf  # Use +inf for minimization failure
    
# def RMVP2_mipCVXPY(D, tau, tau_bar, gamma, beta, t):
#     """
#     Mixed-integer programming formulation for CP-RMVP2 problem using CVXPY:
#         min_{x in R^N} -tau^T x + tau_bar + gamma * ||x||_D + beta * ||x||_0  s.t. ||x||_D <= t
#     
#     Parameters
#     ----------
#     D : (n, n) array_like
#         Positive semidefinite symmetric matrix D.
#     tau : (n,) array_like
#         The excess mean return vector estimate tau.
#     tau_bar : float
#         Scalar parameter tau_bar.
#     gamma : float
#         Ellipsoidal uncertainty radius γ.
#     beta : float
#         Sparsity penalty parameter.
#     t : float
#         D-norm constraint bound.
#     
#     Returns
#     -------
#     x_opt : (n, 1) ndarray
#         Optimal solution.
#     obj_val : float
#         Optimal objective value.
#     """
#     n = D.shape[0]
#     # Big-M value for sparsity constraints
#     M = 1e5  # Large enough bound for x
#     
#     # Convert to numpy arrays
#     D = np.asarray(D, dtype=float)
#     tau = np.asarray(tau, dtype=float)
#     
#     # Decision variables
#     x = cp.Variable(n, name="x")
#     z = cp.Variable(n, boolean=True, name="z")
#     s = cp.Variable(nonneg=True, name="s")  # Auxiliary variable for ||x||_D
#     
#     # Objective: min -tau^T x + tau_bar + gamma * ||x||_D + beta * sum(z_i)
#     # Note: beta * ||x||_0 is a reward in maximization, so we add it
#     linear_part = tau.T @ x
#     sparsity_term = beta * cp.sum(z)
#     objective = cp.Minimize(-linear_part + tau_bar + gamma * s + sparsity_term)
#     
#     # Constraint: ||x||_D <= t, which is equivalent to: x^T D x <= t^2
#     # Using Cholesky decomposition: D = L L^T, then ||L^T x|| <= t
#     # This is a second-order cone constraint: (t, L^T x) in SOC
#     L = np.linalg.cholesky(D)
#     L_T = L.T
#     norm_constraint = cp.SOC(t, L_T @ x)
#     
#     # Also need s >= ||x||_D for the objective
#     # Constraint: s >= ||x||_D, which is: ||L^T x|| <= s
#     s_norm_constraint = cp.SOC(s, L_T @ x)
#     
#     # Sparsity constraints: |x_i| <= M * z_i
#     # This is: -M * z_i <= x_i <= M * z_i
#     sparsity_upper = x <= M * z
#     sparsity_lower = x >= -M * z
#     
#     # Formulate problem
#     constraints = [norm_constraint, s_norm_constraint, sparsity_upper, sparsity_lower]
#     problem = cp.Problem(objective, constraints)
#     
#     # Solve with default available solver
#     problem.solve(verbose=False)
#     
#     # Extract solution
#     if problem.status == cp.OPTIMAL:
#         x_opt = x.value.reshape(-1, 1)
#         # Post-process: set near-zero entries to exactly zero
#         x_opt[np.abs(x_opt) < 1e-8] = 0.0
#
#         obj_val = problem.value
#
#         
#         return x_opt, obj_val
#     else:
#         return np.zeros((n, 1)), np.inf  # Use +inf for minimization failure
    



