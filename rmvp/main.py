from asyncio import taskgroups
import numpy as np
import pandas as pd
import cvxpy as cp
from queue import PriorityQueue, LifoQueue
import time
import gurobipy as gp
from gurobipy import GRB
import mosek.fusion as mf
import mosek
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




def mainRMVP1BnB(D, tau, tau_bar, gamma, beta, method='mosek', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None):
    """
    Branch-and-bound algorithm for CP-RMVP problem:
        min_{x in R^n} x^T D x  s.t.  tau^T x - gamma * ||x||_D >= tau_bar
    """
    relErr = 1e-8

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
            print("global_ub <= lb")
            break
        if abs(ub - lb) < relErr:
            print("abs(ub - lb) < relErr")
            continue
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

            q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, x1, lambda_val])

        if left_supp.size >= 1:
            w = left_supp
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, left_ub, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
            left_ub = left_ub + beta * left_supp.size

            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, x_w_opt, lambda_val])

    global_supp = np.array(sorted(global_supp.tolist()), dtype=np.int32)
    return global_x, global_ub, global_supp, count

# Old implementation (commented)
# def mainRMVP1BnB(D, tau, tau_bar, gamma, beta, method='mosek', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None):
#     """
#     Branch-and-bound algorithm for CP-RMVP problem:
#         min_{x in R^n} x^T D x  s.t.  tau^T x - gamma * ||x||_D >= tau_bar
#     
#     Parameters
#     ----------
#     D : (n, n) array_like
#         Positive semidefinite symmetric matrix D.
#     tau : (n,) array_like
#         The excess mean return vector estimate tau (ccr in the problem).
#     tau_bar : float
#         Scalar parameter tau_bar (bar{ccr} in the problem).
#     gamma : float
#         Ellipsoidal uncertainty radius γ.
#     beta : float
#         Sparsity penalty parameter.
#     method : str, optional
#         Solver method (not used currently).
#     branch_rule : str, optional
#         Branching rule ('max_lagrangian_grad').
#     traverse_rule : str, optional
#         Traversal rule ('bfs' or 'dfs').
#     time_limit : float, optional
#         Time limit for the algorithm.
#     
#     Returns
#     -------
#     global_x : (n,) ndarray
#         Optimal solution.
#     global_ub : float
#         Global upper bound (optimal objective value).
#     global_supp : list
#         Support of the optimal solution.
#     count : int
#         Number of nodes explored.
#     """
#     relErr = 1e-8
# 
#     if traverse_rule == 'bfs':
#         q = PriorityQueue()
#     elif traverse_rule == 'dfs':
#         q = LifoQueue()
#     else:
#         raise ValueError(f"Invalid traverse rule: {traverse_rule}")
# 
#     ub = 10e10
#     global_ub = ub + 1e2
#     
#     # Solve initial relaxed problem
#     x_init, lb, lambda_init = solveRMVP1(D, tau_bar, tau, gamma)
# 
# 
#     global_supp = []
#     Ssupp = []
#     Psupp = list(range(D.shape[1]))
#     DD = np.diag(D)
#     
#     q.put([lb,ub,0,Ssupp,Psupp,x_init,lambda_init])
# 
#     count = 0
#     while q.qsize() >= 1:
#         [lb,ub,_,Ssupp,Psupp,x1,lambda_val] = q.get()
#         #print("count: ", count)
#         #print("lb: ", lb)
#         #print("global_ub: ", global_ub)
#         #print("Ssupp: ", Ssupp)
#         #print("Psupp: ", Psupp)
#         #print("x1: ", x1)
#         #print("lambda_val: ", lambda_val)
# 
#         count += 1
#         if ub - global_ub < relErr:
#             global_ub = ub
#             global_supp = Ssupp
#             global_x = x1
# 
# 
#         if global_ub - lb <= relErr:
#             print("global_ub <= lb")
#             break
#         if abs(ub - lb) < relErr:
#             print("abs(ub - lb) < relErr")
#             continue
#         if len(Psupp) == 0:
#             continue
#         else:
#             if len(Ssupp) >= 1:
#                 ind, bb_ind = branchVariable_rmvp1(x1, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule)
#             else:
#                 tau_Psupp = tau[Psupp]
#                 var = DD[Psupp]
#                 bb_dec = var/(tau_Psupp+1e-8)
#                 bb_ind = np.argmin(bb_dec)
#                 ind = Psupp[bb_ind]
# 
#             left_supp = sorted(Ssupp + [ind])
#             Psupp = list(np.delete(np.array(Psupp), bb_ind))
# 
# 
# 
#             if len(Ssupp) + len(Psupp) >= 1: # Since right_sup = Ssupp + Psupp
#                 w = Ssupp + Psupp
#                 D_w = D[w,:][:,w]
#                 tau_w = tau[w]
# 
#                 x_w_opt, right_lb, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
#                 right_lb = right_lb + beta * len(Ssupp)
# 
# 
#                 q.put([right_lb,ub,np.random.rand(),Ssupp,Psupp,x1,lambda_val])
#                 
# 
#             
#             if len(left_supp) >= 1:
#                 w = left_supp
#                 D_w = D[w,:][:,w]
#                 tau_w = tau[w]
# 
# 
#                 x_w_opt, left_ub, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma)
#                 left_ub = left_ub + beta * len(left_supp)
# 
#                 q.put([lb+beta,left_ub,np.random.rand(),left_supp,Psupp,x_w_opt,lambda_val])
#             
#             else:
#                 q.put([lb+beta,ub,np.random.rand(),left_supp,Psupp,x1,lambda_val])
# 
#     global_supp = sorted(global_supp)
#     #print("count: ", count)
#     return global_x, global_ub, global_supp, count



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
    #model.setParam('Threads', threads)
    model.setParam('IntFeasTol', 1e-8)
    model.setParam('FeasibilityTol', 1e-8)
    model.setParam('OptimalityTol', 1e-8)
    model.setParam('MIPGap', 1e-8)
    model.setParam('MIPGapAbs', 1e-8)
    # Improve numeric robustness
    model.setParam('NumericFocus', 3)
    model.setParam('BarConvTol', 1e-8)
    model.setParam('BarQCPConvTol', 1e-8)
    # Time limit (seconds) ~ 1 hour
    model.setParam('TimeLimit', 3600)

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


def RMVP1_mipMOSEK(D, tau, tau_bar, gamma, beta):
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
    
    # Convert to numpy arrays
    D = np.asarray(D, dtype=float)
    tau = np.asarray(tau, dtype=float)
    
    # Create MOSEK model
    M_model = mf.Model("RMVP1_mipMOSEK")
    
    # Decision variables
    # x: continuous variables (n-dimensional)
    x = M_model.variable("x", n, mf.Domain.unbounded())
    # z: binary variables (n-dimensional) for sparsity
    z = M_model.variable("z", n, mf.Domain.binary())
    # t: auxiliary variable for ||x||_D = sqrt(x^T D x), t >= 0
    t = M_model.variable("t", mf.Domain.greaterThan(0.0))
    # q: auxiliary variable for x^T D x in the objective, q >= 0
    q = M_model.variable("q", mf.Domain.greaterThan(0.0))
    
    # Objective: q + beta * sum(z_i)
    # We will add constraint q >= x^T D x separately
    sparsity_term = mf.Expr.mul(beta, mf.Expr.sum(z))
    quad_obj = mf.Expr.add(q, sparsity_term)
    M_model.objective(mf.ObjectiveSense.Minimize, quad_obj)
    
    # Constraint: q >= x^T D x
    # Using Cholesky decomposition: D = L L^T, then ||L^T x||^2 <= q
    # This is modeled as: (q, 0.5, L^T x) in RotatedQCone
    # RotatedQCone (r, s, x) means: 2*r*s >= ||x||^2
    # So (q, 0.5, L^T x) means: 2*q*0.5 >= ||L^T x||^2, i.e., q >= ||L^T x||^2 = x^T D x
    L = np.linalg.cholesky(D)
    L_T_matrix = mf.Matrix.dense(L.T)
    Ltx = mf.Expr.mul(L_T_matrix, x)
    quad_cone_expr = mf.Expr.vstack(q, 0.5, Ltx)
    M_model.constraint("quad_obj_constraint", 
                      quad_cone_expr, 
                      mf.Domain.inRotatedQCone())
    
    # Constraint: tau^T x - gamma * t >= tau_bar
    linear_part = mf.Expr.dot(tau, x)
    M_model.constraint("robust_constraint", 
                      mf.Expr.sub(linear_part, mf.Expr.mul(gamma, t)), 
                      mf.Domain.greaterThan(tau_bar))
    
    # Constraint: t >= sqrt(x^T D x), which is equivalent to: x^T D x <= t^2
    # Using Cholesky decomposition: D = L L^T, then ||L^T x|| <= t
    # This is modeled as: (t, L^T x) in QuadraticCone
    # QuadraticCone (r, x) means: r >= ||x||
    # So (t, L^T x) means: t >= ||L^T x||, which gives t^2 >= ||L^T x||^2 = x^T D x
    Ltx_norm = mf.Expr.mul(L_T_matrix, x)
    cone_expr = mf.Expr.vstack(t, Ltx_norm)
    M_model.constraint("norm_constraint", 
                      cone_expr, 
                      mf.Domain.inQCone())
    
    # Sparsity constraints: |x_i| <= M * z_i
    # This is: -M * z_i <= x_i <= M * z_i
    # We add constraints: x <= M * z and x >= -M * z
    M_model.constraint("sparsity_upper", 
                      mf.Expr.sub(x, mf.Expr.mul(M, z)), 
                      mf.Domain.lessThan(0.0))
    M_model.constraint("sparsity_lower", 
                      mf.Expr.add(x, mf.Expr.mul(M, z)), 
                      mf.Domain.greaterThan(0.0))
    
    # Set solver parameters for higher precision
    # Interior point tolerances
    M_model.setSolverParam("intpntCoTolRelGap", 1e-8)
    M_model.setSolverParam("intpntCoTolPfeas", 1e-8)
    M_model.setSolverParam("intpntCoTolDfeas", 1e-8)
    M_model.setSolverParam("intpntCoTolMuRed", 1e-8)
    M_model.setSolverParam("intpntCoTolInfeas", 1e-8)
    
    # Mixed-integer solver tolerances
    M_model.setSolverParam("mioTolRelGap", 1e-8)
    M_model.setSolverParam("mioTolAbsGap", 1e-8)
    M_model.setSolverParam("mioMaxTime", -1)  # No time limit

    # Solve
    M_model.solve()
    
    # Extract solution
    solsta = M_model.getPrimalSolutionStatus()
    if solsta == mf.SolutionStatus.Optimal:
        x_opt = np.array(M_model.getVariable('x').level()).reshape(-1, 1)
        obj_val = M_model.primalObjValue()
        M_model.dispose()
        return x_opt, obj_val
    else:
        M_model.dispose()
        return np.zeros((n, 1)), np.inf


def RMVP1_mipCVXPY(D, tau, tau_bar, gamma, beta):
    """
    Mixed-integer programming formulation for CP-RMVP problem using CVXPY:
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
    
    # Convert to numpy arrays
    D = np.asarray(D, dtype=float)
    tau = np.asarray(tau, dtype=float)

    
    # Decision variables
    x = cp.Variable(n, name="x")
    z = cp.Variable(n, boolean=True, name="z")
    t = cp.Variable(nonneg=True, name="t")
    
    # Objective: x^T D x + beta * sum(z_i)
    quad_term = cp.quad_form(x, D)
    sparsity_term = beta * cp.sum(z)
    objective = cp.Minimize(quad_term + sparsity_term)
    
    # Constraint: tau^T x - gamma * t >= tau_bar
    robust_constraint = tau.T @ x - gamma * t >= tau_bar
    
    # Constraint: t >= sqrt(x^T D x), which is equivalent to: x^T D x <= t^2
    # Using Cholesky decomposition: D = L L^T, then ||L^T x|| <= t
    # This is a second-order cone constraint: (t, L^T x) in SOC
    L = np.linalg.cholesky(D)
    L_T = L.T
    norm_constraint = cp.SOC(t, L_T @ x)
    
    # Sparsity constraints: |x_i| <= M * z_i
    # This is: -M * z_i <= x_i <= M * z_i
    sparsity_upper = x <= M * z
    sparsity_lower = x >= -M * z
    
    # Formulate problem
    constraints = [robust_constraint, sparsity_upper, sparsity_lower]

    u = cp.Variable(nonneg=True)   # represents t^2

    constraints += [
        cp.quad_form(x, cp.psd_wrap(D)) <= u,          # x^T D x <= u   (DCP)
        cp.SOC(u + 0.5, cp.hstack([u - 0.5, t]))       # enforces t^2 <= u (rotated cone)
    ]


    problem = cp.Problem(objective, constraints)
    
    # Solve with MOSEK
    # Set solver parameters for higher precision
    mosek_params = {
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-8,
        "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-8,
        "MSK_DPAR_MIO_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_MIO_TOL_ABS_GAP": 1e-8,
    }
    problem.solve(solver=cp.MOSEK, verbose=False, mosek_params=mosek_params)
    
    # Extract solution
    if problem.status == cp.OPTIMAL:
        x_opt = x.value.reshape(-1, 1)
        # Post-process: set near-zero entries to exactly zero
        #x_opt[np.abs(x_opt) < 1e-8] = 0.0
        obj_val = problem.value
        return x_opt, obj_val
    else:
        return np.zeros((n, 1)), np.inf

    
    


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


def mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t, method='mosek', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None):
    """
    Branch-and-bound algorithm for CP-RMVP2 problem (same structure as mainRMVP1BnB).
    """
    relErr = 1e-8

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

        if global_ub <= lb:
            print("global_ub <= lb")
            break
        if abs(ub - lb) < relErr:
            print("abs(ub - lb) < relErr")
            continue
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

            q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, x1, lambda_val])

        if left_supp.size >= 1:
            w = left_supp
            D_w = D[np.ix_(w, w)]
            tau_w = tau[w]

            x_w_opt, left_ub, lambda_val = solveRMVP2(D_w, tau_bar, tau_w, gamma, beta, t)
            left_ub = left_ub + beta * left_supp.size

            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, x_w_opt, lambda_val])

    global_supp = np.array(sorted(global_supp.tolist()), dtype=np.int32)
    return global_x, global_ub, global_supp, count

# Old implementation (commented)
# def mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t, method='mosek', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None):
#     """
#     Branch-and-bound algorithm for CP-RMVP2 problem (same structure as mainRMVP1BnB).
#     """
#     relErr = 1e-8
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
#     
#     # Solve initial relaxed problem
#     x_init, lb, lambda_init = solveRMVP2(D, tau_bar, tau, gamma, beta, t)
# 
# 
#     global_supp = []
#     Ssupp = []
#     Psupp = list(range(D.shape[1]))
#     DD = np.diag(D)
#     
#     q.put([lb,ub,0,Ssupp,Psupp,x_init,lambda_init])
# 
#     count = 0
#     while q.qsize() >= 1:
#         [lb,ub,_,Ssupp,Psupp,x1,lambda_val] = q.get()
# 
#         count += 1
#         if ub - global_ub < relErr:
#             global_ub = ub
#             global_supp = Ssupp
#             global_x = x1
# 
# 
#         if global_ub <= lb:
#             print("global_ub <= lb")
#             break
#         if abs(ub - lb) < relErr:
#             print("abs(ub - lb) < relErr")
#             continue
#         if len(Psupp) == 0:
#             continue
#         else:
#             if len(Ssupp) >= 1:
#                 ind, bb_ind = branchVariable_rmvp2(x1, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule)
#             else:
#                 #bb_ind = 0
#                 #ind = Psupp[bb_ind]
# 
#                 tau_Psupp = tau[Psupp]
#                 var = DD[Psupp]
#                 bb_dec = var/(tau_Psupp+1e-8)
#                 bb_ind = np.argmin(bb_dec)
#                 ind = Psupp[bb_ind]
# 
#             left_supp = sorted(Ssupp + [ind])
#             Psupp = list(np.delete(np.array(Psupp), bb_ind))
# 
# 
#             #if len(Ssupp) == 0: print("Ssupp is empty")
# 
# 
# 
#             if len(Ssupp) + len(Psupp) >= 1: # Since right_sup = Ssupp + Psupp
#                 w = Ssupp + Psupp
#                 D_w = D[w,:][:,w]
#                 tau_w = tau[w]
# 
#                 x_w_opt, right_lb, lambda_val = solveRMVP2(D_w, tau_bar, tau_w, gamma, beta, t)
#                 right_lb = right_lb + beta * len(Ssupp)
# 
# 
#                 q.put([right_lb,ub,np.random.rand(),Ssupp,Psupp,x1,lambda_val])
#                 
# 
#             
#             if len(left_supp) >= 1: 
#                 w = left_supp
#                 D_w = D[w,:][:,w]
#                 tau_w = tau[w]
# 
# 
#                 x_w_opt, left_ub, lambda_val = solveRMVP2(D_w, tau_bar, tau_w, gamma, beta, t)
#                 left_ub = left_ub + beta * len(left_supp)
# 
#                 q.put([lb+beta,left_ub,np.random.rand(),left_supp,Psupp,x_w_opt,lambda_val])
#             
#             else:
#                 q.put([lb+beta,ub,np.random.rand(),left_supp,Psupp,x1,lambda_val])
# 
#     global_supp = sorted(global_supp)
#     #print("count: ", count)
#     return global_x, global_ub, global_supp, count


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
    M = 1e5  # Large enough bound for x
    
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
    
    model.setParam('OutputFlag', 0)
    #model.setParam('Threads', threads)
    model.setParam('IntFeasTol', 1e-8)
    # Time limit (seconds) ~ 1 hour
    model.setParam('TimeLimit', 3600)
    
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
    
def RMVP2_mipCVXPY(D, tau, tau_bar, gamma, beta, t):
    """
    Mixed-integer programming formulation for CP-RMVP2 problem using CVXPY:
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
    M = 1e5  # Large enough bound for x
    
    # Convert to numpy arrays
    D = np.asarray(D, dtype=float)
    tau = np.asarray(tau, dtype=float)
    
    # Decision variables
    x = cp.Variable(n, name="x")
    z = cp.Variable(n, boolean=True, name="z")
    s = cp.Variable(nonneg=True, name="s")  # Auxiliary variable for ||x||_D
    
    # Objective: min -tau^T x + tau_bar + gamma * ||x||_D + beta * sum(z_i)
    # Note: beta * ||x||_0 is a reward in maximization, so we add it
    linear_part = tau.T @ x
    sparsity_term = beta * cp.sum(z)
    objective = cp.Minimize(-linear_part + tau_bar + gamma * s + sparsity_term)
    
    # Constraint: ||x||_D <= t, which is equivalent to: x^T D x <= t^2
    # Using Cholesky decomposition: D = L L^T, then ||L^T x|| <= t
    # This is a second-order cone constraint: (t, L^T x) in SOC
    L = np.linalg.cholesky(D)
    L_T = L.T
    norm_constraint = cp.SOC(t, L_T @ x)
    
    # Also need s >= ||x||_D for the objective
    # Constraint: s >= ||x||_D, which is: ||L^T x|| <= s
    s_norm_constraint = cp.SOC(s, L_T @ x)
    
    # Sparsity constraints: |x_i| <= M * z_i
    # This is: -M * z_i <= x_i <= M * z_i
    sparsity_upper = x <= M * z
    sparsity_lower = x >= -M * z
    
    # Formulate problem
    constraints = [norm_constraint, s_norm_constraint, sparsity_upper, sparsity_lower]
    problem = cp.Problem(objective, constraints)
    
    # Solve with MOSEK
    problem.solve(solver=cp.MOSEK, verbose=False)
    
    # Extract solution
    if problem.status == cp.OPTIMAL:
        x_opt = x.value.reshape(-1, 1)
        # Post-process: set near-zero entries to exactly zero
        x_opt[np.abs(x_opt) < 1e-8] = 0.0

        obj_val = problem.value

        
        return x_opt, obj_val
    else:
        return np.zeros((n, 1)), np.inf  # Use +inf for minimization failure
    




def generate_test_instance_rmvp1(n, seed=None, beta=None, gamma=None, r_c=None, bar_r=None):
    """
    Generate test instances for CP-RMVP problem satisfying the assumptions.
    
    Assumptions:
    1. bar{r} > r_c
    2. D is positive definite symmetric matrix
    3. For every i: ccr[i] > bar{ccr} * sqrt(d_i[i])
       where ccr = r_hat - r_c * ones, bar{ccr} = bar{r} - r_c
       and (sqrt(d))_i denotes the i-th column of sqrt(D), sqrt(d_i[i]) is the i-th element
       of the i-th column of sqrt(D), i.e., sqrt(D)[i,i]
       sqrt(D) is computed via eigenvalue decomposition: D = Q @ Lambda @ Q.T,
       then sqrt(D) = Q @ sqrt(Lambda) @ Q.T
    
    Parameters
    ----------
    n : int
        Dimension of the problem (number of assets).
    seed : int, optional
        Random seed for reproducibility.
    beta : float, optional
        Sparsity penalty parameter. If None, generated randomly.
    gamma : float, optional
        Uncertainty radius. If None, generated randomly.
    r_c : float, optional
        Riskless return. If None, generated randomly.
    bar_r : float, optional
        Target return parameter. If None, generated randomly.
    
    Returns
    -------
    instance : dict
        Dictionary containing:
        - 'D': (n, n) positive definite symmetric matrix
        - 'tau': (n,) excess mean return vector (ccr = r_hat - r_c * ones)
        - 'tau_bar': float scalar (bar{ccr} = bar{r} - r_c)
        - 'beta': float sparsity penalty parameter
        - 'gamma': float uncertainty radius
        - 'r_hat': (n,) estimated mean return vector
        - 'r_c': float riskless return
        - 'bar_r': float target return parameter
    """
    if seed is not None:
        np.random.seed(seed)


    r_hat = np.random.randn(n) * 5
    
    # Generate random positive definite symmetric matrix D
    # D is not diagonal, it's a general symmetric positive definite matrix
    A = np.random.randn(n, n)
    D = A.T @ A + 0.1 * np.eye(n)  # Add small diagonal for positive definiteness
    
    # Ensure D is symmetric (should already be, but enforce it)
    D = (D + D.T) / 2
    
    # Ensure D is positive definite (add small epsilon to diagonal if needed)
    eigvals = np.linalg.eigvalsh(D)
    if np.min(eigvals) < 1e-8:
        D += (1e-8 - np.min(eigvals)) * np.eye(n)
        # Re-symmetrize after modification
        D = (D + D.T) / 2

    # Generate parameters
    if r_c is None:
        r_c = np.random.uniform(0.001, 0.005)  # Riskless return between 1% and 5%
    
    if bar_r is None:
        bar_r = r_c + np.random.uniform(0.02, 0.10)  # bar{r} > r_c

    
    
    # Calculate Excess parameters
    tau = r_hat - r_c             
    tau_bar = bar_r - r_c    


    # Calculate ratios for gamma logic
    D_diag_sqrt = np.sqrt(np.diag(D))
    # Avoid division by zero
    ratios = np.abs(tau) / np.maximum(D_diag_sqrt, 1e-8)
    min_ratio = np.min(ratios)
    
    

    
    
    # Generate gamma if not provided (move gamma generation earlier)
    if gamma is None:
        # gamma should be positive and typically less than some bound
        gamma = min_ratio - 1e-5
    
    # Construct tau to strictly satisfy the inequality with a margin
    # tau[i] = gamma * sqrt_D[i,i] * (1 + random_buffer)
    # Using random buffer between 0.1 and 0.5 ensures strict inequality
    random_buffer = np.random.uniform(0.1, 0.5, size=n)
    tau = gamma * D_diag_sqrt * (1.0 + random_buffer)

    
    # Generate beta if not provided
    if beta is None:
        beta = np.random.uniform(0.01, 0.1)
    
    return {
        'D': D,
        'tau': tau,
        'tau_bar': tau_bar,
        'beta': beta,
        'gamma': gamma,
        'r_hat': r_hat,
        'r_c': r_c,
        'bar_r': bar_r
    }


def test_rmvp1_solvers(n=10, seed=42, verbose=True):
    """
    Test function that generates a test instance and runs both RMVP1 solvers.
    
    Parameters
    ----------
    n : int
        Dimension of the problem (number of assets).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, print detailed results.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'instance': The generated test instance
        - 'mip_solution': Solution from RMVP1_mip (x_opt, obj_val)
        - 'bnb_solution': Solution from mainRMVP1BnB (global_x, global_ub, global_supp, count)
    """
    print("=" * 80)
    print("Testing RMVP1 Solvers")
    print("=" * 80)
    
    # Generate test instance
    if verbose:
        print(f"\nGenerating test instance with n={n}, seed={seed}...")
    instance = generate_test_instance_rmvp1(n=n, seed=seed)
    
    D = instance['D']
    tau = instance['tau']
    tau_bar = instance['tau_bar']
    beta = instance['beta']
    gamma = instance['gamma']

    
    if verbose:
        print(f"Instance parameters:")
        print(f"  - Dimension: {n}")
        print(f"  - tau_bar: {tau_bar:.6f}")
        print(f"  - beta: {beta:.6f}")
        print(f"  - gamma: {gamma:.6f}")
        print(f"  - r_c: {instance['r_c']:.6f}")
        print(f"  - bar_r: {instance['bar_r']:.6f}")
        print(f"  - D shape: {D.shape}")
        print(f"  - D eigenvalues range: [{np.min(np.linalg.eigvalsh(D)):.6f}, {np.max(np.linalg.eigvalsh(D)):.6f}]")
    
    results = {'instance': instance}
    
    # Run MIP solver
    print("\n" + "-" * 80)
    print("Running RMVP1_mip (Mixed-Integer Programming solver)...")
    print("-" * 80)
    start_time = time.time()
    x_mip, obj_mip = RMVP1_mipCVXPY(D, tau, tau_bar, gamma, beta)
    obj_mip = (x_mip.T @ D @ x_mip + beta * np.sum(np.abs(x_mip) > 1e-8))[0][0]
    
    mip_time = time.time() - start_time
    const = tau.T @ x_mip - gamma * np.sqrt(x_mip.T @ D @ x_mip)
    print("const: ", const)
    
    if verbose:
        print(f"MIP Solver Results:")
        print(f"  - Status: {'Optimal' if obj_mip < np.inf else 'Infeasible/Error'}")
        print(f"  - Objective value: {obj_mip:.6f}")
        print(f"  - Solve time: {mip_time:.4f} seconds")
        print(f"  - Solution x:")
        print(f"    {x_mip.flatten()}")
        print(f"  - Non-zero components: {np.sum(np.abs(x_mip) > 1e-8)}")
        print(f"  - Support: {np.where(np.abs(x_mip.flatten()) > 1e-8)[0].tolist()}")
    
    results['mip_solution'] = {
        'x': x_mip,
        'obj_val': obj_mip,
        'time': mip_time
    }
    
    # Run Branch-and-Bound solver
    print("\n" + "-" * 80)
    print("Running mainRMVP1BnB (Branch-and-Bound solver)...")
    print("-" * 80)
    start_time = time.time()
    x_bnb, obj_bnb, supp_bnb, count_bnb = mainRMVP1BnB(D, tau, tau_bar, gamma, beta)
    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    obj_bnb = (x_bnb.T @ D @ x_bnb + beta * np.sum(np.abs(x_bnb) > 1e-8))[0][0]
    bnb_time = time.time() - start_time
    const = tau.T @ x_bnb - gamma * np.sqrt(x_bnb.T @ D @ x_bnb)
    print("const: ", const)
    if verbose:
        print(f"Branch-and-Bound Solver Results:")
        print(f"  - Objective value: {obj_bnb:.6f}")
        print(f"  - Solve time: {bnb_time:.4f} seconds")
        print(f"  - Nodes explored: {count_bnb}")
        print(f"  - Solution x:")
        if x_bnb.ndim == 2:
            print(f"    {x_bnb.flatten()}")
        else:
            print(f"    {x_bnb}")
        print(f"  - Support: {supp_bnb}")
        print(f"  - Non-zero components: {len(supp_bnb)}")
    
    results['bnb_solution'] = {
        'x': x_bnb,
        'obj_val': obj_bnb,
        'support': supp_bnb,
        'count': count_bnb,
        'time': bnb_time
    }
    
    # Compare solutions
    print("\n" + "-" * 80)
    print("Comparison:")
    print("-" * 80)
    if obj_mip < np.inf and obj_bnb < np.inf:
        obj_diff = abs(obj_mip - obj_bnb)
        obj_rel_diff = obj_diff / max(abs(obj_mip), abs(obj_bnb)) if max(abs(obj_mip), abs(obj_bnb)) > 0 else 0
        print(f"  - Objective difference: {obj_diff:.6f} (relative: {obj_rel_diff*100:.2f}%)")
        
        # Compare solution vectors
        x_mip_flat = x_mip.flatten()
        if x_bnb.ndim == 2:
            x_bnb_flat = x_bnb.flatten()
        else:
            x_bnb_flat = x_bnb
        
        if len(x_mip_flat) == len(x_bnb_flat):
            sol_diff = np.linalg.norm(x_mip_flat - x_bnb_flat)
            print(f"  - Solution vector difference (L2 norm): {sol_diff:.6f}")
        
        print(f"  - Speed comparison:")
        print(f"    MIP: {mip_time:.4f}s")
        print(f"    BnB: {bnb_time:.4f}s")
        print(f"    Ratio: {bnb_time/mip_time:.2f}x" if mip_time > 0 else "    Ratio: N/A")
        const = tau.T @ x_mip - gamma * np.sqrt(x_mip.T @ D @ x_mip)
        print("const MIP: ", const)
        const = tau.T @ x_bnb - gamma * np.sqrt(x_bnb.T @ D @ x_bnb)
        print("const BNB: ", const)
    print("\n" + "=" * 80)
    
    return results


def generate_test_instance_rmvp2(n, seed=None, beta=None, gamma=None, r_c=None, bar_r=None, t=None):
    """
    Generate test instances for CP-RMVP2 problem satisfying the assumptions.
    
    Assumptions:
    1. bar{r} > r_c
    2. D is positive definite symmetric matrix
    3. For every i: ccr[i] > gamma * sqrt(d_i[i])
       where ccr = r_hat - r_c * ones
       and sqrt(d_i[i]) is the i-th element of the i-th column of sqrt(D)
       sqrt(D) is computed via Cholesky decomposition: D = L L^T, sqrt(D) = L
    
    Parameters
    ----------
    n : int
        Dimension of the problem (number of assets).
    seed : int, optional
        Random seed for reproducibility.
    beta : float, optional
        Sparsity penalty parameter. If None, generated randomly.
    gamma : float, optional
        Uncertainty radius. If None, generated randomly.
    r_c : float, optional
        Riskless return. If None, generated randomly.
    bar_r : float, optional
        Target return parameter. If None, generated randomly.
    t : float, optional
        D-norm constraint bound. If None, generated based on problem structure.
    
    Returns
    -------
    instance : dict
        Dictionary containing:
        - 'D': (n, n) positive definite symmetric matrix
        - 'tau': (n,) excess mean return vector (ccr = r_hat - r_c * ones)
        - 'tau_bar': float scalar (bar{ccr} = bar{r} - r_c)
        - 'beta': float sparsity penalty parameter
        - 'gamma': float uncertainty radius
        - 'r_hat': (n,) estimated mean return vector
        - 'r_c': float riskless return
        - 'bar_r': float target return parameter
        - 't': float D-norm constraint bound
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random positive definite symmetric matrix D
    # D is not diagonal, it's a general symmetric positive definite matrix
    A = (np.random.randn(n, n) + 1) * 10
    D = A.T @ A + 0.1 * np.eye(n)  # Add small diagonal for positive definiteness
    D = (D + D.T) / 2

    # Ensure D is positive definite (add small epsilon to diagonal if needed)
    eigvals = np.linalg.eigvalsh(D)
    if np.min(eigvals) < 1e-8:
        D += (1e-8 - np.min(eigvals)) * np.eye(n)
        # Re-symmetrize after modification
        D = (D + D.T) / 2

    # Compute sqrt(D) using Cholesky decomposition: D = L @ L.T
    # sqrt(D) = L (lower triangular matrix) as requested
    try:
        sqrt_D = np.linalg.cholesky(D)
    except np.linalg.LinAlgError:
        # Fallback: ensure PD if numerical issues
        eigvals, eigvecs = np.linalg.eigh(D)
        eigvals = np.maximum(eigvals, 1e-8)
        D = eigvecs @ np.diag(eigvals) @ eigvecs.T
        sqrt_D = np.linalg.cholesky(D)
    
    # Generate parameters
    if r_c is None:
        r_c = np.random.uniform(0.01, 0.05)  # Riskless return between 1% and 5%
    
    if bar_r is None:
        bar_r = r_c + np.random.uniform(0.02, 0.10)  # bar{r} > r_c
    
    # Compute tau_bar = bar{r} - r_c
    tau_bar = bar_r - r_c

    # Generate gamma if not provided (move gamma generation earlier)
    if gamma is None:
        # gamma should be positive and typically less than some bound
        gamma = np.random.uniform(0.01, 0.1)

    # Ensure Assumption 1 (labeled Assumption 2.1 in text) is satisfied by construction:
    #   tau[i] > gamma * sqrt(d_i[i]) for all i,
    # where sqrt(d_i[i]) is the i-th element of the i-th column of sqrt(D).
    # Since sqrt_D is lower triangular (Cholesky), the i-th element of i-th column is sqrt_D[i, i].
    
    diag_sqrt_D = np.diag(sqrt_D)
    
    # Construct tau to strictly satisfy the inequality with a margin
    # tau[i] = gamma * sqrt_D[i,i] * (1 + random_buffer)
    # Using random buffer between 0.1 and 0.5 ensures strict inequality
    random_buffer = np.random.uniform(0.1, 0.5, size=n)
    tau = gamma * diag_sqrt_D * (1.0 + random_buffer)
    
    # Recover r_hat consistent with tau and r_c: tau = r_hat - r_c
    r_hat = tau + r_c
    
    # Generate beta if not provided
    if beta is None:
        beta = np.random.uniform(0.01, 0.1)
    
    # Generate parameter t (D-norm constraint bound) for RMVP2 if not provided
    # t should be large enough to allow feasible solutions
    # We must satisfy the lower bound assumption:
    # t > min_i { beta / (tau[i]/sqrt(D[i,i]) - gamma) }
    
    # Calculate the lower bound
    # Note: Assumption 1 guarantees tau[i] > gamma * sqrt(D[i,i]),
    # so tau[i]/sqrt(D[i,i]) > gamma, so the denominator is positive.
    denominators = (tau / diag_sqrt_D) - gamma
    # Ensure denominators are strictly positive (they should be by construction)
    denominators = np.maximum(denominators, 1e-8)
    
    t_lower_bounds = beta / denominators
    min_t_bound = np.min(t_lower_bounds)
    
    if t is None:
        try:
            # Compute H = sqrt(tau^T D^{-1} tau) to estimate a reasonable t
            D_inv = np.linalg.inv(D)
            H = float(np.sqrt(tau.T @ D_inv @ tau))
            
            # Start with a base value related to H
            # Ensure it strictly satisfies the assumption
            # t = max(1.5 * H, 1.1 * min_t_bound)
            
            # Use a factor between 1.5 and 3.0 relative to max(H, bound)
            base_t = max(H, min_t_bound)
            t = np.random.uniform(1.5, 3.0) * base_t
        except:
            # Fallback
            t = max(1.0, 1.1 * min_t_bound)
            
    # Enforce the lower bound if a specific t was provided or generated
    if t <= min_t_bound:
        t = 1.1 * min_t_bound
    
    return {
        'D': D,
        'tau': tau,
        'tau_bar': tau_bar,
        'beta': beta,
        'gamma': gamma,
        'r_hat': r_hat,
        'r_c': r_c,
        'bar_r': bar_r,
        't': t
    }


def test_rmvp2_solvers(n=10, seed=42, verbose=True):
    """
    Test function that generates a test instance and runs RMVP2 solvers (Gurobi MIP, CVXPY, and BnB).
    
    Parameters
    ----------
    n : int
        Dimension of the problem (number of assets).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, print detailed results.
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'instance': The generated test instance
        - 'mip_gurobi': Solution from RMVP2_mipGUROBI (x_opt, obj_val)
        - 'mip_cvxpy': Solution from RMVP2_mipCVXPY (x_opt, obj_val)
        - 'bnb_solution': Solution from mainRMVP2BnB (global_x, global_ub, global_supp, count)
    """
    print("=" * 80)
    print("Testing RMVP2 Solvers")
    print("=" * 80)
    
    # Generate test instance
    if verbose:
        print(f"\nGenerating test instance with n={n}, seed={seed}...")
    instance = generate_test_instance_rmvp2(n=n, seed=seed)
    
    D = instance['D']
    tau = instance['tau']
    tau_bar = instance['tau_bar']
    beta = instance['beta']
    gamma = instance['gamma']
    t = instance['t']
    
    print("t: ", t)
    print("H: ", tau.T @ np.linalg.inv(D) @ tau)

    print("tau/gamma: ", tau/gamma)
    print("sqrt(D): ", np.diag(np.linalg.cholesky(D)))

    if verbose:
        print(f"Instance parameters:")
        print(f"  - Dimension: {n}")
        print(f"  - tau_bar: {tau_bar:.6f}")
        print(f"  - beta: {beta:.6f}")
        print(f"  - gamma: {gamma:.6f}")
        print(f"  - t (D-norm bound): {t:.6f}")
        print(f"  - D shape: {D.shape}")
    
    results = {'instance': instance}
    
    # Run Gurobi MIP solver
    print("\n" + "-" * 80)
    print("Running RMVP2_mipGUROBI...")
    print("-" * 80)
    start_time = time.time()
    x_gurobi, obj_gurobi = RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta, t)
    
    # Recompute objective to ensure consistency
    if obj_gurobi > -np.inf:
        norm_D_gurobi = np.sqrt(x_gurobi.T @ D @ x_gurobi)[0,0]
        l0_gurobi = np.sum(np.abs(x_gurobi) > 1e-8)
        obj_gurobi_recalc = -(tau.T @ x_gurobi)[0] + tau_bar + gamma * norm_D_gurobi + beta * l0_gurobi
    else:
        obj_gurobi_recalc = -np.inf
        norm_D_gurobi = 0
        
    gurobi_time = time.time() - start_time
    
    if verbose:
        print(f"Gurobi MIP Solver Results:")
        print(f"  - Objective value: {obj_gurobi_recalc:.6f}")
        print(f"  - Solve time: {gurobi_time:.4f} seconds")
        print(f"  - Solution x norm_D: {norm_D_gurobi:.6f} (<= {t:.6f})")
        print(f"  - Non-zero components: {np.sum(np.abs(x_gurobi) > 1e-8)}")
        #print("x_gurobi: ", x_gurobi)
    
    results['mip_gurobi'] = {
        'x': x_gurobi,
        'obj_val': obj_gurobi_recalc,
        'time': gurobi_time
    }

    # Run CVXPY MIP solver
    print("\n" + "-" * 80)
    print("Running RMVP2_mipCVXPY...")
    print("-" * 80)
    start_time = time.time()
    x_cvxpy, obj_cvxpy = RMVP2_mipCVXPY(D, tau, tau_bar, gamma, beta, t)
    
    # Recompute objective
    if obj_cvxpy > -np.inf:

        norm_D_cvxpy = np.sqrt(x_cvxpy.T @ D @ x_cvxpy)[0,0]
        l0_cvxpy = np.sum(np.abs(x_cvxpy) > 1e-8)
        obj_cvxpy_recalc = -(tau.T @ x_cvxpy)[0] + tau_bar + gamma * norm_D_cvxpy + beta * l0_cvxpy
    else:
        obj_cvxpy_recalc = -np.inf
        norm_D_cvxpy = 0
        
    cvxpy_time = time.time() - start_time
    
    if verbose:
        print(f"CVXPY MIP Solver Results:")
        print(f"  - Objective value: {obj_cvxpy_recalc:.6f}")
        print(f"  - Solve time: {cvxpy_time:.4f} seconds")
        print(f"  - Solution x norm_D: {norm_D_cvxpy:.6f} (<= {t:.6f})")
        print(f"  - Non-zero components: {np.sum(np.abs(x_cvxpy) > 1e-8)}")
        #print("x_cvxpy: ", x_cvxpy)
    results['mip_cvxpy'] = {
        'x': x_cvxpy,
        'obj_val': obj_cvxpy_recalc,
        'time': cvxpy_time
    }
    
    # Run Branch-and-Bound solver
    print("\n" + "-" * 80)
    print("Running mainRMVP2BnB (Branch-and-Bound solver)...")
    print("-" * 80)
    start_time = time.time()
    x_bnb, obj_bnb, supp_bnb, count_bnb = mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t)
    
    # Pad and recompute
    x_bnb_full = zeropadding(x_bnb, supp_bnb, n)
    norm_D_bnb = np.sqrt(x_bnb_full.T @ D @ x_bnb_full)[0,0]
    l0_bnb = len(supp_bnb)
    obj_bnb_recalc = -(tau.T @ x_bnb_full)[0] + tau_bar + gamma * norm_D_bnb + beta * l0_bnb
    
    bnb_time = time.time() - start_time
    
    if verbose:
        print(f"Branch-and-Bound Solver Results:")
        print(f"  - Objective value: {obj_bnb_recalc:.6f}")
        print(f"  - Solve time: {bnb_time:.4f} seconds")
        print(f"  - Nodes explored: {count_bnb}")
        print(f"  - Solution x norm_D: {norm_D_bnb:.6f} (<= {t:.6f})")
        print(f"  - Non-zero components: {l0_bnb}")
        #print("x_bnb: ", x_bnb_full)
    results['bnb_solution'] = {
        'x': x_bnb_full,
        'obj_val': obj_bnb_recalc,
        'support': supp_bnb,
        'count': count_bnb,
        'time': bnb_time
    }
    
    # Compare solutions
    print("\n" + "-" * 80)
    print("Comparison:")
    print("-" * 80)
    
    # Compare objectives
    objs = {
        'Gurobi': obj_gurobi_recalc,
        'CVXPY': obj_cvxpy_recalc,
        'BnB': obj_bnb_recalc
    }
    
    for name, obj in objs.items():
        print(f"{name} Objective: {obj:.6f}")
        
    # Check consistency
    best_obj = max(obj for obj in objs.values() if obj > -np.inf)
    print(f"\nBest Objective: {best_obj:.6f}")
    
    for name, obj in objs.items():
        if obj > -np.inf:
            diff = abs(best_obj - obj)
            print(f"{name} Diff from Best: {diff:.6f}")
    
    print("\n" + "=" * 80)
    
    return results

def check_proposition_rmvp1(n=30, seed=42):
    """
    Checks the proposition regarding feasibility sensitivity to zeroing out elements for RMVP1.
    
    1. Generates RMVP1 instance.
    2. Solves it.
    3. For each non-zero element x_i:
       - Checks if setting x_i = 0 renders the solution infeasible.
       - If so, reports |x_i| and sqrt(beta / L_ii), where L is Cholesky factor of D.
    """
    print("=" * 80)
    print(f"Checking Proposition for RMVP1 (n={n}, seed={seed})")
    print("=" * 80)
    
    instance = generate_test_instance_rmvp1(n=n, seed=seed)
    D = instance['D']
    tau = instance['tau']
    tau_bar = instance['tau_bar']
    gamma = instance['gamma']
    beta = instance['beta']
    beta = 0.1
    
    print(f"Parameters: n={n}, beta={beta:.6f}, gamma={gamma:.6f}, tau_bar={tau_bar:.6f}")
    
    # Solve using CVXPY
    print("Solving problem with RMVP1_mipCVXPY...")
    #x_opt, obj_val = RMVP1_mipCVXPY(D, tau, tau_bar, gamma, beta)
    x_opt, obj_val, supp_bnb, count_bnb = mainRMVP1BnB(D, tau, tau_bar, gamma, beta)
    x_opt = zeropadding(x_opt, supp_bnb, n)

    #x_opt = x_opt.flatten()
    L = np.linalg.cholesky(D)
    diag_L = np.diag(L)
    
    # Identify support
    support = np.where(np.abs(x_opt) > 1e-8)[0]
    print("x_opt: ", x_opt)
    print(f"Optimal objective: {obj_val:.6f}")
    print(f"Support size: {len(support)}")
    print(f"Support indices: {support}")
    print("-" * 80)
    print(f"{'Index':<10} | {'|x_i|':<15} | {'sqrt(beta/L_ii)':<20} | {'Feasible if 0?':<20}")
    print("-" * 80)
    
    results = []
    
    for i in support:
        # Create modified x with x[i] = 0
        x_new = x_opt.copy()
        x_new[i] = 0.0
        
        # Check feasibility: tau^T x - gamma ||x||_D >= tau_bar
        # ||x||_D = sqrt(x^T D x)
        norm_D_new = np.sqrt(x_new.T @ D @ x_new)
        robust_lhs = tau.T @ x_new - gamma * norm_D_new
        
        is_feasible = robust_lhs >= tau_bar - 1e-8 # Tolerance
        
        L_ii = diag_L[i]
        bound = np.sqrt(beta / L_ii)
        
        status = "YES" if is_feasible else "NO (Infeasible)"
        
        print(f"{i:<10} | {float(abs(x_opt[i])):<15.6f} | {float(bound):<20.6f} | {status:<20}")
        
        if not is_feasible:
            results.append({
                'index': i,
                'x_i_abs': abs(x_opt[i]),
                'bound': bound
            })
            
    print("-" * 80)
    return results




def generate_rmvp1_data(
    file_path: str, 
    sheet_name: str = 'AssetReturns', 
    r_c: float = 0.00001, 
    gamma: float = 0.01, 
    beta: float = 1.0, 
    target_return_factor: float = 1.1
):
    # 1. Load Data (No header assumed)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    returns = df.values # Matrix of daily returns
    n_samples, n_assets = returns.shape
    
    # 2. Compute Statistics
    # Mean-center the data
    r_hat = returns.mean(axis=0)
    returns_centered = returns - r_hat

    # Sample covariance matrix
    D = (returns_centered.T @ returns_centered) / (n_samples - 1)
    
    # 3. Setup Problem Parameters & Check Assumptions
    # --- Assumption 1: Target Wealth > Risk-free rate ---
    if r_c == 0:
        avg_pos_return = np.mean(r_hat[r_hat > 0]) if np.any(r_hat > 0) else 0.001
        bar_r = avg_pos_return * target_return_factor
    else:
        bar_r = r_c * target_return_factor
    
    if bar_r <= r_c:
        raise ValueError(f"Assumption Violation: Target wealth ({bar_r:.4f}) must be strictly greater than risk-free rate ({r_c}).")
        
    # Calculate Excess parameters
    tau = r_hat - r_c             
    tau_bar = bar_r - r_c      
    
    # --- Assumption 2: Feasibility Check (Robust constraint feasibility) ---
    # Condition: |tau[i]| / sqrt(D[i,i]) > gamma
    # where sqrt(D[i,i]) is the volatility (std dev) of the i-th asset
    D_diag_sqrt = np.sqrt(np.diag(D))
    
    # Calculate ratios
    ratios = np.abs(tau) / D_diag_sqrt
    print("min(|tau| / sqrt(D_ii)): ", np.min(ratios))
    
    # Check condition: |tau| / sqrt(D_ii) > gamma  <=>  |tau| - gamma * sqrt(D_ii) > 0
    feasibility_margin = np.abs(tau) - (gamma * D_diag_sqrt)
    
    infeasible_indices = np.where(feasibility_margin <= 0)[0]
    if len(infeasible_indices) > 0:
        msg = (f"Assumption Violation: {len(infeasible_indices)} assets violate the robust feasibility condition "
                f"(|tau| > gamma * sqrt(D_ii)). Indices: {infeasible_indices}")
        print(f"Warning: {msg}")
        
        


    # 4. Pack into Dictionary
    problem_data = {
        "n": n_assets,
        "D": D,
        "r_hat": r_hat,
        "r_c": r_c,
        "gamma": gamma,
        "beta": beta,
        "bar_r": bar_r,
        "tau": tau,
        "tau_bar": tau_bar,
        "num_samples": n_samples
    }
    
    return problem_data


if __name__ == "__main__":
    #test_rmvp1_solvers(n=20, seed=35, verbose=True)
    #test_rmvp2_solvers(n=80, seed=35, verbose=True)
    #check_proposition_rmvp1(n=25, seed=35)


    """
    n = 4
    instance = generate_test_instance_rmvp2(n=n, seed=42)
    
    D = instance['D']
    tau = instance['tau']
    tau_bar = instance['tau_bar']
    beta = instance['beta']
    gamma = instance['gamma']
    t = instance['t']
    t = 3
    print("t: ", t)
    beta = 0.1


    L = np.linalg.cholesky(D)
    diag_L = np.diag(L)
    
    

    x_bnb, obj_bnb, supp_bnb, count_bnb = mainRMVP2BnB(D, tau, tau_bar, gamma, beta, t)
    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    obj_bnb = (-tau.T @ x_bnb + tau_bar + gamma * np.sqrt(x_bnb.T @ D @ x_bnb) + beta * np.sum(np.abs(x_bnb) > 1e-6))[0][0]
    const = x_bnb.T @ D @ x_bnb
    print("const BNB: ", const)
    print("obj_bnb: ", obj_bnb)
    print("supp_bnb: ", supp_bnb)
    print("count_bnb: ", count_bnb)
    print("x_bnb: ", x_bnb)


    x_mip, obj_mip = RMVP2_mipGUROBI(D, tau, tau_bar, gamma, beta, t)
    obj_mip = (-tau.T @ x_mip + tau_bar + gamma * np.sqrt(x_mip.T @ D @ x_mip) + beta * np.sum(np.abs(x_mip) > 1e-6))[0][0]
    const = x_mip.T @ D @ x_mip
    print("const GUROBI: ", const)
    print("obj_mip GUROBI: ", obj_mip)
    print("x_mip GUROBI: ", x_mip)

    x_mip, obj_mip = RMVP2_mipCVXPY(D, tau, tau_bar, gamma, beta, t)
    obj_mip = (-tau.T @ x_mip + tau_bar + gamma * np.sqrt(x_mip.T @ D @ x_mip) + beta * np.sum(np.abs(x_mip) > 1e-6))[0][0]
    const = x_mip.T @ D @ x_mip
    print("const CVXPY: ", const)
    print("obj_mip CVXPY: ", obj_mip)
    print("x_mip CVXPY: ", x_mip)


    





    """
    n = 30
    instance = generate_test_instance_rmvp1(n=n, seed=4)
    
    D = instance['D']
    tau = instance['tau']
    tau_bar = instance['tau_bar']
    print("tau_bar: ", tau_bar)
    beta = instance['beta']
    gamma = instance['gamma']


    L = np.linalg.cholesky(D)
    diag_L = np.diag(L)
    
    

    x_bnb, obj_bnb, supp_bnb, count_bnb = mainRMVP1BnB(D, tau, tau_bar, gamma, beta)
    x_bnb = zeropadding(x_bnb, supp_bnb, n)
    obj_bnb = (x_bnb.T @ D @ x_bnb + beta * np.sum(np.abs(x_bnb) > 1e-8))[0][0]
    const = tau.T @ x_bnb - gamma * np.sqrt(x_bnb.T @ D @ x_bnb)
    print("const BNB: ", const)
    print("obj_bnb: ", obj_bnb)
    print("supp_bnb: ", supp_bnb)
    print("count_bnb: ", count_bnb)
    print("x_bnb: ", x_bnb)

    x_mip, obj_mip = RMVP1_mipMOSEK(D, tau, tau_bar, gamma, beta)
    obj_mip = (x_mip.T @ D @ x_mip + beta * np.sum(np.abs(x_mip) > 1e-8))[0][0]
    const = tau.T @ x_mip - gamma * np.sqrt(x_mip.T @ D @ x_mip)
    print("const MOSEK: ", const)
    print("obj_mip MOSEK: ", obj_mip)
    print("x_mip MOSEK: ", x_mip)

    x_mip, obj_mip = RMVP1_mipCVXPY(D, tau, tau_bar, gamma, beta)
    obj_mip = (x_mip.T @ D @ x_mip + beta * np.sum(np.abs(x_mip) > 1e-8))[0][0]
    const = tau.T @ x_mip - gamma * np.sqrt(x_mip.T @ D @ x_mip)
    print("const CVXPY: ", const)
    print("obj_mip CVXPY: ", obj_mip)
    print("x_mip CVXPY: ", x_mip)
    
    
