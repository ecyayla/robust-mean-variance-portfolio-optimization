import numpy as np
import cvxpy as cp
from queue import PriorityQueue, LifoQueue
import time
import gurobipy as gp
from gurobipy import GRB
import mosek.fusion as mf
import math
import multiprocessing

from wlk21 import solveTRP_WLK21
from hk16 import solveTRP_HK16
from lkp17 import solveTRP_LKP17
from aint2017 import solveTRP_AINT2017
from bv18 import solveTRP_BV18_PG, solveTRP_BV18_CG

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

def lagrangian_gradient(x_s, g, H, lambda_val, Ssupp, Psupp):
    """
    Gradient of the Lagrangian w.r.t. the still‑zero coordinates (Psupp).

        ∇_Z L = g_Z + H_{Z,S} x_S            (because x_Z = 0)

    Parameters
    ----------
    x_s        : (|supp|, 1)  current point on the active support S
    g          : (n, 1)       gradient vector of the quadratic term
    H          : (n, n)       Hessian matrix  (symmetric)
    lambda_val : float        trust‑region multiplier (drops out for Z)
    Ssupp      : 1‑D array    indices of the active non‑zero set  S
    Psupp      : 1‑D array    indices of the still‑zero set        Z

    Returns
    -------
    grad_z     : (|psupp|, 1) gradient components for Z
    """
    # 1) slice g on the rows we care about
    g_z = g[Psupp]

    # 2) cross block H_{Z,S}: rows in psupp, cols in supp
    if len(Ssupp) > 0:
        H_zs = H[Psupp, :][:, Ssupp]
        grad_z = g_z + H_zs @ x_s
    else:
        # When Ssupp is empty, there's no cross term
        grad_z = g_z
    
    return grad_z

def compute_objective(x, A, b, c):
    return x.T @ A @ x + 2 * x.T @ b + c


def solveTRP_coordinate_descent(A, b, c, r, max_iter=100):
    """Coordinate descent for trust-region subproblem"""
    n = A.shape[0]
    x = np.zeros((n, 1))
    
    for iter in range(max_iter):
        for i in range(n):
            # Update coordinate i
            grad_i = A[i, :] @ x + b[i]
            hess_ii = A[i, i]
            
            # Optimal step for coordinate i
            step = -grad_i / hess_ii
            x_new = x.copy()
            x_new[i] += step
            
            # Project onto trust region
            if np.linalg.norm(x_new) > r:
                x_new = r * x_new / np.linalg.norm(x_new)
            
            x = x_new
    
    return x, compute_objective(x, A, b, c)

def solveTRP_gradient(A, b, c, r, max_iter=100):
    """Solve using projected gradient descent"""
    n = A.shape[0]
    x = np.zeros((n, 1))
    
    for iter in range(max_iter):
        # Compute gradient
        grad = A @ x + b
        
        # Projected gradient step
        x_new = x - 0.01 * grad
        
        # Project onto trust region
        if np.linalg.norm(x_new) > r:
            x_new = r * x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new
    
    return x, compute_objective(x, A, b, c)


def solveTRP_cvxpy(A, b, c, r):
    d, eigenvecs = np.linalg.eigh(A)
    n = len(d)
    
    # Decision variables
    z = cp.Variable(n, nonneg=True)
    t  = cp.Variable(n, nonneg=True)
    
    f = eigenvecs.T @ b
    abs_f = np.abs(f)
    # Objective function
    objective = cp.sum(cp.multiply(d, z)) - 2 * cp.sum(cp.multiply(abs_f, cp.power(z, 0.5))) + c
    
    # Constraint: ||y||^2 <= r
    constraint = [cp.sum(z) <= r]
    constraint += [
        cp.SOC(z[i] + 1e-12, cp.hstack([2*t[i], z[i] - 1e-12]))  # t^2 ≤ z
    for i in range(n)
    ]
    
    objective = cp.sum(cp.multiply(d, z)) - 2*cp.sum(cp.multiply(abs_f, t)) + c
    # Problem
    
    problem = cp.Problem(cp.Minimize(objective), constraint)

    # Solve
    problem.solve(solver=cp.MOSEK)

    if problem.status == cp.OPTIMAL:
        z_opt = z.value.reshape(-1,1)
        sign_f = np.sign(f)

        y_opt = -sign_f * np.sqrt(z_opt) 
        x_opt = eigenvecs @ y_opt
        x_opt = x_opt.reshape(-1,1)

        return x_opt, objective.value
    else:
        print(f"Warning: Problem not optimal, status: {problem.status}")
        return np.zeros((A.shape[0], 1)), np.inf


def solveTRP_mosek(A, b, c, r):
    d, eigenvecs = np.linalg.eigh(A)
    n = len(d)

    M = mf.Model()
    
    z = M.variable("z", n, mf.Domain.greaterThan(0.0))
    t = M.variable("t", n, mf.Domain.greaterThan(0.0))

    f = eigenvecs.T @ b
    abs_f = np.abs(f)
    #objective = mf.Expr.dot(d, z) - 2 * mf.Expr.dot(abs_f, np.sqrt(z)) + c

    # Trust‑region radius: Σ z_i ≤ r
    M.constraint("radius", mf.Expr.sum(z), mf.Domain.lessThan(r))
    for i in range(n):
        M.constraint(                # ( z_i , 0.5 , t_i ) ∈ RQC
            f"cone_{i}",
            mf.Expr.vstack(z.index(i), 0.5, t.index(i)),
            mf.Domain.inRotatedQCone()
        )

    lin1 = mf.Expr.dot(d, z)
    lin2 = mf.Expr.dot(abs_f, t)

    objective = mf.Expr.add(lin1, mf.Expr.mul(-2.0, lin2))
    objective = mf.Expr.add(objective, c)
    M.objective(
        "obj",
        mf.ObjectiveSense.Minimize,
        objective
    )

    M.solve()

    solsta = M.getPrimalSolutionStatus()
    if solsta != mf.SolutionStatus.Optimal:
        M.dispose()
        raise Exception(f"Unexpected solution status: {solsta}")

    z_opt = M.getVariable('z').level().reshape(-1,1)
    sign_f = np.sign(f)
    #sign_f = np.where(f >= 0, 1, -1)
    y_opt = -sign_f * np.sqrt(z_opt) 
    x_opt = (eigenvecs @ y_opt).reshape(-1,1)
    obj_val = M.primalObjValue()
    M.dispose()

    return x_opt, obj_val


# --- Unified solver wrappers returning (x_opt, f_opt, meta) --------------
def solveTRP_cvxpy_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt = solveTRP_cvxpy(A, b, c, r)
    meta_out = {"solver": "cvxpy", "status": "OK"}
    return x_opt, f_opt, meta_out


def solveTRP_mosek_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt = solveTRP_mosek(A, b, c, r)
    meta_out = {"solver": "mosek", "status": "OK"}
    return x_opt, f_opt, meta_out


def solveTRP_coordinate_descent_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt = solveTRP_coordinate_descent(A, b, c, r)
    meta_out = {"solver": "coordinate_descent", "status": "OK"}
    return x_opt, f_opt, meta_out


def solveTRP_gradient_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt = solveTRP_gradient(A, b, c, r)
    meta_out = {"solver": "gradient", "status": "OK"}
    return x_opt, f_opt, meta_out


def solveTRP_WLK21_wrapper(A, b, c, r, meta_in=None):
    A1 = np.eye(A.shape[0])
    b1 = np.zeros_like(b)
    c1 = -r
    theta = 0.5
    xi = 0.1
    gamma_hat = max(0, -np.linalg.eigvalsh(A)[0]) + theta
    zeta = max(1, gamma_hat + 1)
    eps = 1e-6
    p = 1e-6
    x_opt, f_opt = solveTRP_WLK21(A, b, c, A1, b1, c1, xi, zeta, gamma_hat, eps, p)
    meta_out = {
        "solver": "WLK21",
        "params": {
            "theta": theta,
            "xi": xi,
            "gamma_hat": gamma_hat,
            "zeta": zeta,
            "eps": eps,
            "p": p,
        },
    }
    return x_opt, f_opt, meta_out


def solveTRP_HK16_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt = solveTRP_HK16(A, b, c)
    meta_out = {"solver": "HK16", "status": "OK"}
    return x_opt, f_opt, meta_out


def solveTRP_LKP17_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt = solveTRP_LKP17(A, b, c)
    meta_out = {"solver": "LKP17", "status": "OK"}
    return x_opt, f_opt, meta_out


def solveTRP_AINT2017_wrapper(A, b, c, r, meta_in=None):
    # meta_in may contain warm-start hints (e.g., eigenvectors)
    x_opt, f_opt, meta = solveTRP_AINT2017(A, b, c, r, meta_in=meta_in)
    meta_out = {"solver": "AINT2017", "status": "OK"}
    if meta is not None:
        meta_out.update(meta)
    return x_opt, f_opt, meta_out


def solveTRP_BV18_PG_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt, meta = solveTRP_BV18_PG(A, b, c, r, meta_in=meta_in)
    meta_out = {"solver": "BV18_PG", "status": "OK"}
    if meta is not None:
        meta_out.update(meta)
    return x_opt, f_opt, meta_out


def solveTRP_BV18_CG_wrapper(A, b, c, r, meta_in=None):
    x_opt, f_opt, meta = solveTRP_BV18_CG(A, b, c, r, meta_in=meta_in)
    meta_out = {"solver": "BV18_CG", "status": "OK"}
    if meta is not None:
        meta_out.update(meta)
    return x_opt, f_opt, meta_out


SOLVER_REGISTRY = {
    "cvxpy": solveTRP_cvxpy_wrapper,
    "cxvpy": solveTRP_cvxpy_wrapper,  # backward compatibility with typo
    "mosek": solveTRP_mosek_wrapper,
    "coordinate_descent": solveTRP_coordinate_descent_wrapper,
    "gradient": solveTRP_gradient_wrapper,
    "WLK21": solveTRP_WLK21_wrapper,
    "HK16": solveTRP_HK16_wrapper,
    "LKP17": solveTRP_LKP17_wrapper,
    "AINT2017": solveTRP_AINT2017_wrapper,
    "BV18_PG": solveTRP_BV18_PG_wrapper,
    "BV18_CG": solveTRP_BV18_CG_wrapper,
}


def solveTRP(A, b, c, r, method='cvxpy', meta_in=None):
    """
    Unified TRS entry point. If meta_in is None, solver runs cold-start and
    any meta_out is still returned (caller may ignore).
    """
    solver = SOLVER_REGISTRY.get(method)
    if solver is None:
        raise ValueError(f"Invalid method: {method}")
    return solver(A, b, c, r, meta_in=meta_in)


def _slice_z_to_child(z_parent, w_parent, w_child):
    """
    Map parent eigenvector z (length 2*|w_parent|) to child ordering w_child.
    Returns None if dimensions mismatch or indices are inconsistent.
    """
    if z_parent is None or w_parent is None:
        return None
    try:
        w_to_pos = {idx: pos for pos, idx in enumerate(w_parent)}
        positions = [w_to_pos[i] for i in w_child]
    except Exception:
        return None

    z_parent = np.asarray(z_parent).reshape(-1)
    n_parent = len(w_parent)
    if z_parent.shape[0] != 2 * n_parent:
        return None

    y2_parent = z_parent[:n_parent]
    y1_parent = z_parent[n_parent:]
    y2_child = y2_parent[positions]
    y1_child = y1_parent[positions]
    return np.concatenate([y2_child, y1_child], axis=0)


def _slice_x_to_child(x_parent, w_parent, w_child):
    """
    Map parent primal x (length |w_parent|) to child ordering w_child.
    Returns None if dimensions mismatch or indices are inconsistent.
    """
    if x_parent is None or w_parent is None:
        return None
    w_to_pos = {idx: pos for pos, idx in enumerate(w_parent)}

    x_parent = np.asarray(x_parent).reshape(-1)
    n_parent = len(w_parent)
    if x_parent.shape[0] != n_parent:
        return None

    x_child = np.zeros((len(w_child), 1))
    for j, idx in enumerate(w_child):
        pos = w_to_pos.get(idx)
        if pos is not None:
            x_child[j, 0] = x_parent[pos]
        else:
            # New index not in parent support: warm-start at 0
            x_child[j, 0] = 0.0
    return x_child


def _build_meta_for_child(meta_parent, w_child, transmit_payload):
    """
    Build meta_in for a child node; returns None if parent meta is missing.
    transmit_payload: string or None, supports "eigenvector" or "primal".
    """
    if meta_parent is None:
        return None
    meta_in = {}
    w_parent = meta_parent.get("w_child", None)
    meta_in["w_parent"] = w_parent

    if transmit_payload == "eigenvector":
        z_parent = meta_parent.get("z", None)
        z_child = _slice_z_to_child(z_parent, w_parent, w_child)
        if z_child is not None:
            meta_in["z0"] = z_child
    elif transmit_payload == "primal":
        x_parent = meta_parent.get("x", None)
        x_child = _slice_x_to_child(x_parent, w_parent, w_child)
        if x_child is not None:
            meta_in["x0"] = x_child

    meta_in["w_child"] = w_child
    return meta_in


"""
Deprecated (not used): branchVariable_SB
Kept for reference; re-enable if needed.

def branchVariable_SB(x, A, b, lambda_val, Ssupp, Psupp, lb, ub, global_ub, n=10, beta=0.1, r=1, method='mosek', meta=None, transmit_meta=True):
    ...
"""



def branchVariable(x, A, b, lambda_val, Ssupp, Psupp, branch_rule='max_lagrangian_grad'):
    if branch_rule == 'max_lagrangian_grad':
        grad = lagrangian_gradient(x, b, A, lambda_val, Ssupp, Psupp)
        bb_ind = np.argmax(abs(grad[:,0]))
        ind = Psupp[bb_ind]
    else:
        raise ValueError(f"Invalid branch rule: {branch_rule}")
    
    return ind, bb_ind



def mainBnB(
    A,
    b,
    c,
    beta,
    r,
    method='mosek',
    branch_rule='max_lagrangian_grad',
    traverse_rule='bfs',
    time_limit=None,
    transmit_meta_left=None,
    transmit_meta_right=None,
):
    relErr = 1e-8

    # When transmit_meta_left/right is None, no warm-start info is sent.

    if traverse_rule == 'bfs':
        q = PriorityQueue()
    elif traverse_rule == 'dfs':
        q = LifoQueue()
    else:
        raise ValueError(f"Invalid traverse rule: {traverse_rule}")

    ub = 10e10
    global_ub = ub + 1e-4
    #x_init, lb, meta_init = solveTRP(A, b, c, r, method)

    # --- Root solve (meta_init gets the root's w_child = full indices)
    x_init, lb, meta_init = solveTRP(A, b, c, r, method, meta_in=None)
    if transmit_meta_left or transmit_meta_right:
        if meta_init is None:
            meta_init = {}
        meta_init["w_child"] = list(range(A.shape[0]))
    else:
        meta_init = None

    global_supp = []
    Ssupp = []
    Psupp = list(range(A.shape[1]))
    lambda_init = 0
    q.put([lb, ub, 0, Ssupp, Psupp, x_init, lambda_init, meta_init])

    count = 0
    while q.qsize() >= 1:
        [lb, ub, _, Ssupp, Psupp, x1, lambda_val, meta_parent] = q.get()
        count += 1

        # incumbent update
        if ub - global_ub < relErr:
            global_ub = ub
            global_supp = Ssupp
            global_x = x1

        # termination / pruning
        if global_ub - lb < relErr:
            break
        if abs(ub - lb) < relErr:
            print("ub - lb < relErr")
            continue
        if len(Psupp) == 0:
            continue

        # choose branching variable
        if len(Ssupp) >= 1:
            ind, bb_ind = branchVariable(x1, A, b, lambda_val, Ssupp, Psupp, branch_rule)
        else:
            diag = A[Psupp, :][:, Psupp].diagonal()
            bb_ind = np.argmin(diag)
            ind = Psupp[bb_ind]

        left_supp = sorted(Ssupp + [ind])
        Psupp = list(np.delete(np.array(Psupp), bb_ind))

        # -------------------------
        # RIGHT CHILD: w = S + P
        # -------------------------
        if len(Ssupp) + len(Psupp) >= 1:
            w = Ssupp + Psupp
            Aw = A[w, :][:, w]
            bw = b[w, 0:1]

            # build meta_in for child
            meta_in = _build_meta_for_child(meta_parent, w, transmit_meta_right) if transmit_meta_right else None

            x_w_opt, right_lb, meta_right = solveTRP(Aw, bw, c, r, method, meta_in=meta_in)
            right_lb = right_lb + beta * len(Ssupp)

            if transmit_meta_right:
                if meta_right is None:
                    meta_right = {}
                meta_right["w_child"] = w  # always store this node's w
            else:
                meta_right = None

            # (paper-style) inherit x1,lambda_val on right
            q.put([right_lb, ub, np.random.rand(), Ssupp, Psupp, x1, lambda_val, meta_right])

        # -------------------------
        # LEFT CHILD: w = S U {ind}
        # -------------------------
        if len(left_supp) >= 1:
            w = left_supp
            Aw = A[w, :][:, w]
            bw = b[w, 0:1]

            meta_in = _build_meta_for_child(meta_parent, w, transmit_meta_left) if transmit_meta_left else None

            x_w_opt, left_ub, meta_left = solveTRP(Aw, bw, c, r, method, meta_in=meta_in)
            left_ub = left_ub + beta * len(left_supp)

            if transmit_meta_left:
                if meta_left is None:
                    meta_left = {}
                meta_left["w_child"] = w  # always store this node's w
            else:
                meta_left = None

            q.put([lb + beta, left_ub, np.random.rand(), left_supp, Psupp, x_w_opt, lambda_val, meta_left])
        else:
            # degenerate left case
            q.put([lb + beta, ub, np.random.rand(), left_supp, Psupp, x1, lambda_val, meta_parent])

    global_supp = sorted(global_supp)
    return global_x, global_ub, global_supp, count


    

def brute_force_TRP_beta_hidden(A, b, c, beta, r):
    """
    Global optimum by brute force:
        min  xᵀ A x + 2 bᵀ x + c + beta·|supp(x)|   s.t. ‖x‖₂² ≤ radius
    """
    n        = A.shape[0]
    best_val = np.inf
    best_x   = None

    # enumerate all 2^n supports
    for mask in range(1 << n):
        S = [i for i in range(n) if mask & (1 << i)]
        if not S:                # empty support ⇒ x = 0
            val = float(c[0, 0])
            if val < best_val:
                best_val, best_x = val, np.zeros((n, 1))
            continue

        # sub‑problem on support S
        #A_S = A[np.ix_(S, S)]
        A_S = A[S,:][:,S]
        b_S = b[S].reshape(-1, 1)


        x_sub, val_sub, _ = solveTRP(A_S, b_S, c, r, "cvxpy")
        #x_full = np.zeros((n, 1))
        #x_full[S, 0] = x_sub.flatten()
        x_full = zeropadding(x_sub, S, A.shape[1]) 

        total = val_sub + beta * len(S)

        if total < best_val:
            best_val, best_x = total, x_full

    return best_x, best_val


def TRP_mip(A, b, c, r, beta):
    n = A.shape[0]
    M = 2 * r
    
    model = gp.Model("sparse_trust_region")
    
    # Variables as vectors
    x = model.addVars(n, lb=-M, ub=M, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # Convert to Gurobi expressions
    x_vec = [x[i] for i in range(n)]
    
    # Quadratic term: x^T A x
    quad_term = gp.quicksum(A[i,j] * x[i] * x[j] for i in range(n) for j in range(n))
    
    # Linear term: 2 b^T x
    linear_term = 2 * gp.quicksum(float(b[i,0]) * x[i] for i in range(n))
    
    # Constant term
    const_term = float(c[0,0])
    
    # Sparsity penalty
    sparsity_term = beta * gp.quicksum(z[i] for i in range(n))
    
    model.setObjective(quad_term + linear_term + const_term + sparsity_term, GRB.MINIMIZE)
    
    # Trust region constraint
    model.addConstr(gp.quicksum(x[i] * x[i] for i in range(n)) <= r**2)
    
    # Big-M constraints
    for i in range(n):
        model.addConstr(x[i] <= M * z[i])
        model.addConstr(x[i] >= -M * z[i])
    
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        x_opt = np.array([[x[i].x] for i in range(n)])
        z_opt = np.array([z[i].x for i in range(n)])
        return x_opt, model.objVal
    else:
        return np.zeros((n, 1)), np.inf

if __name__ == "__main__":

    np.random.seed(42) #30'da hata veriyo, n=30
    """
    #print("A: ", A[[2,0],:])
    for i in range(10):
        n = 40
        A = np.random.rand(n, n)*9+1
        A = (A.T + A) / 2
        #A = A.T @ A

        # Linear term
        #b = np.array([[1.0], [-1.0], [0.5]])
        b = np.random.rand(n, 1)

        # Constant term (can be a vector or scalar, depending on your code)
        #c = np.array([[0.0], [0.0], [0.0]])
        c = np.random.rand(1, 1)


        def normalize(A,b,c):
            s = max(np.linalg.norm(A,2), np.linalg.norm(b), abs(c), 1.0)
            return A/s, b/s, c/s
        # apply to q0 and q1 (track only if you compare objectives afterwards)
        A, b, c = normalize(A, b, c)



        # Regularization/penalty parameter
        beta = 0.1

        # Trust region radius
        r = 1

        A1 = np.eye(A.shape[0])
        b1 = np.zeros_like(b)   
        c1 = -r


        theta = 0.5
        xi = 0.1
        gamma_hat = max(0, -np.linalg.eigvalsh(A)[0]) + theta
        zeta = max(1, gamma_hat+1)
        eps = 1e-2
        p = 1e-2

        start_time = time.time()
        stat, x_opt = solveTRP_LKP17(A, b, c)
        #x_opt, f_opt = solveTRP_HK16(A, b, c)
        end_time = time.time()
        #print("x_opt: ", x_opt)
        #print("f_opt: ", f_opt)
        print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c)
        #print("norm: ", np.linalg.norm(x_opt))
        print("Time taken lkp17: ", end_time - start_time)

        start_time = time.time()
        x_opt, f_opt = solveTRP_mosek(A, b, c, r)
        end_time = time.time()
        #print("x_opt: ", x_opt)
        #print("f_opt: ", f_opt)
        print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c)
        #print("norm: ", np.linalg.norm(x_opt))
        print("Time taken mosek: ", end_time - start_time)





    """

    np.random.seed(13)
    for i in range(5):

        n = 50
        A = np.random.rand(n, n)*10+1
        A = (A.T + A) / 2
        #A = A.T @ A

        # Linear term
        #b = np.array([[1.0], [-1.0], [0.5]])
        b = np.random.rand(n, 1) * 10

        # Constant term (can be a vector or scalar, depending on your code)
        #c = np.array([[0.0], [0.0], [0.0]])
        c = np.random.rand(1, 1)

        beta = 0.1
        r = 1

        #x_opt, f_opt = TRP_mip(A, b, c, r, beta)
        #print("x_opt: ", x_opt)
        #print("f_opt: ", f_opt)
        #print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c)


        print("Running: ", i+1)
        start_time = time.time()
        global_x, global_ub, global_supp, node_count = mainBnB(A, b, c, beta, r, method='AINT2017')
        end_time = time.time()
        print("Time taken AINT2017: ", end_time - start_time)
        print("Node count: ", node_count)

        col = A.shape[1]
        x_opt = zeropadding(global_x, global_supp, col) 
        #print("x_opt: ", x_opt)
        #print("global_ub: ", global_ub)
        print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c + beta * np.count_nonzero(x_opt))
        print("norm: ", np.linalg.norm(x_opt))

        print("="*100)



        start_time = time.time()
        global_x, global_ub, global_supp, node_count = mainBnB(A, b, c, beta, r, method='mosek')
        end_time = time.time()
        print("Time taken mosek: ", end_time - start_time)
        print("Node count: ", node_count)


        col = A.shape[1]
        x_opt = zeropadding(global_x, global_supp, col) 
        #print("x_opt: ", x_opt)
        #print("global_ub: ", global_ub)
        print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c + beta * np.count_nonzero(x_opt))
        print("norm: ", np.linalg.norm(x_opt))


    """
    x_opt, obj_opt = TRP_mip(A, b, c, r, beta)
    print("x_opt: ", x_opt)
    print("obj_opt: ", obj_opt)
    print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c + beta * np.count_nonzero(x_opt))
    print("norm: ", np.linalg.norm(x_opt))



    start_time = time.time()
    global_x, global_ub, global_supp = mainBnB(A, b, c, beta, r, method='gradient')
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    #print("global_x: ", global_x)


    col = A.shape[1]
    x_opt = zeropadding(global_x, global_supp, col) 
    print("x_opt: ", x_opt)
    print("global_ub: ", global_ub)
    print("result: ", x_opt.T @ A @ x_opt + 2 * x_opt.T @ b + c + beta * np.count_nonzero(x_opt))
    print("norm: ", np.linalg.norm(global_x))


    #global_x, global_ub = brute_force_TRP_beta_hidden(A, b, c, beta, r)
    #print("global_x: ", global_x)
    #print("global_ub: ", global_ub)
    #print("result: ", global_x.T @ A @ global_x + 2 * global_x.T @ b + c + beta * np.count_nonzero(global_x))
    #print("norm: ", np.linalg.norm(global_x))





    def solveTRP_cvxpy_debug(A, b, c, r):
        d, eigenvecs = np.linalg.eigh(A)
        n = len(d)
        
        # Decision variables
        z = cp.Variable(n)
        
        f = eigenvecs.T @ b
        abs_f = np.abs(f)
        
        print(f"Debug info:")
        print(f"d (eigenvalues): {d}")
        print(f"f: {f.flatten()}")
        print(f"abs_f: {abs_f.flatten()}")
        
        # Objective function
        objective = cp.sum(cp.multiply(d, z)) - 2 * cp.sum(cp.multiply(abs_f, cp.sqrt(z))) + c
        
        # Constraint: ||y||^2 <= r
        constraint = [cp.sum(z) <= r, z >= 0]
        
        # Problem
        problem = cp.Problem(cp.Minimize(objective), constraint)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            z_opt = z.value.reshape(-1,1)
            print(f"z_opt: {z_opt.flatten()}")
            
            sign_f = np.sign(f)
            y_opt = -sign_f * np.sqrt(z_opt)
            print(f"y_opt: {y_opt.flatten()}")
            
            x_opt = eigenvecs @ y_opt
            x_opt = x_opt.reshape(-1, 1)
            print(f"x_opt: {x_opt.flatten()}")
            
            # Check constraint satisfaction
            norm_x = np.linalg.norm(x_opt)
            print(f"||x_opt|| = {norm_x:.6f} <= {r}")
            
            # Check objective
            obj_cvxpy = objective.value
            print("y shape: ", y_opt.shape)
            print("f shape: ", sign_f.shape)
            obj_manual = compute_objective(x_opt, A, b, c)
            print(f"CVXPY objective: {obj_cvxpy}")
            print(f"Manual objective: {obj_manual}")
            print(f"Difference: {abs(obj_cvxpy - obj_manual)}")
            
            return x_opt, objective.value
        else:
            print(f"Warning: Problem not optimal, status: {problem.status}")
            return np.zeros((A.shape[0], 1)), np.inf
    # Test with a simple 2x2 problem
    A_test = np.array([[2.0, 1.0], [1.0, 3.0]])
    b_test = np.array([[1.0], [-1.0]])
    c_test = np.array([[0.0]])
    r_test = 1.0

    print("Testing simple 2x2 problem:")
    x_cvxpy, obj_cvxpy = solveTRP_cvxpy_debug(A_test, b_test, c_test, r_test)
    x_grad, obj_grad = solveTRP_gradient(A_test, b_test, c_test, r_test)

    print(f"\nComparison:")
    print(f"CVXPY objective: {obj_cvxpy}")
    print(f"Gradient objective: {obj_grad}")
    print(f"CVXPY better: {obj_cvxpy < obj_grad}")




    def test_bfs_dfs_comparison(min_size=10, max_size=20, num_trials=10, method='mosek'):


        results = {}
        
        # Parameters for the optimization problem
        beta = 0.5
        r = 1.0
        
        for size in range(min_size, max_size + 1):
            print(f"Testing size {size}x{size}...")
            
            bfs_times = []
            dfs_times = []
            bfs_objectives = []
            dfs_objectives = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Generate random symmetric positive definite matrix A
                np.random.seed(42 + trial + size * 100)  # Different seed for each trial/size
                A = np.random.rand(size, size) * 9 + 1
                A = (A.T + A) / 2  # Make symmetric
                
                # Generate random vectors b and c
                b = np.random.rand(size, 1)
                c = np.random.rand(1, 1)
                
                # Test BFS
                start_time = time.time()
                try:
                    global_x_bfs, global_ub_bfs, global_supp_bfs = mainBnB(
                        A, b, c, beta, r, method=method, traverse_rule='bfs'
                    )
                    bfs_time = time.time() - start_time
                    bfs_times.append(bfs_time)
                    bfs_objectives.append(global_ub_bfs)
                except Exception as e:
                    print(f"    BFS failed for size {size}, trial {trial}: {e}")
                    bfs_times.append(np.inf)
                    bfs_objectives.append(np.inf)
                
                # Test DFS
                start_time = time.time()
                try:
                    global_x_dfs, global_ub_dfs, global_supp_dfs = mainBnB(
                        A, b, c, beta, r, method=method, traverse_rule='dfs'
                    )
                    dfs_time = time.time() - start_time
                    dfs_times.append(dfs_time)
                    dfs_objectives.append(global_ub_dfs)
                except Exception as e:
                    print(f"    DFS failed for size {size}, trial {trial}: {e}")
                    dfs_times.append(np.inf)
                    dfs_objectives.append(np.inf)
            
            # Calculate averages (excluding failed runs)
            valid_bfs_times = [t for t in bfs_times if t != np.inf]
            valid_dfs_times = [t for t in dfs_times if t != np.inf]
            valid_bfs_objectives = [obj for obj in bfs_objectives if obj != np.inf]
            valid_dfs_objectives = [obj for obj in dfs_objectives if obj != np.inf]
            
            if valid_bfs_times and valid_dfs_times:
                avg_bfs_time = np.mean(valid_bfs_times)
                avg_dfs_time = np.mean(valid_dfs_times)
                avg_bfs_objective = np.mean(valid_bfs_objectives)
                avg_dfs_objective = np.mean(valid_dfs_objectives)
                objective_diff = avg_bfs_objective - avg_dfs_objective
                
                results[size] = {
                    'avg_bfs_time': avg_bfs_time,
                    'avg_dfs_time': avg_dfs_time,
                    'avg_bfs_objective': avg_bfs_objective,
                    'avg_dfs_objective': avg_dfs_objective,
                    'objective_difference': objective_diff,
                    'bfs_faster_ratio': avg_dfs_time / avg_bfs_time if avg_bfs_time > 0 else np.inf,
                    'successful_trials': len(valid_bfs_times)
                }
            else:
                results[size] = {
                    'avg_bfs_time': np.inf,
                    'avg_dfs_time': np.inf,
                    'avg_bfs_objective': np.inf,
                    'avg_dfs_objective': np.inf,
                    'objective_difference': np.inf,
                    'bfs_faster_ratio': np.inf,
                    'successful_trials': 0
                }
        
        return results

    def print_test_results(results):

        print("\n" + "="*120)
        print("BFS vs DFS COMPARISON RESULTS")
        print("="*120)
        print(f"{'Size':<6} {'BFS Time':<12} {'DFS Time':<12} {'BFS/DFS':<10} {'BFS Obj':<12} {'DFS Obj':<12} {'Obj Diff':<12} {'Success':<8}")
        print("-"*120)
        
        for size in sorted(results.keys()):
            r = results[size]
            if r['successful_trials'] > 0:
                bfs_time = f"{r['avg_bfs_time']:.4f}"
                dfs_time = f"{r['avg_dfs_time']:.4f}"
                ratio = f"{r['bfs_faster_ratio']:.2f}"
                bfs_obj = f"{r['avg_bfs_objective']:.6f}"
                dfs_obj = f"{r['avg_dfs_objective']:.6f}"
                obj_diff = f"{r['objective_difference']:.6f}"
                success = f"{r['successful_trials']}"
            else:
                bfs_time = "FAIL"
                dfs_time = "FAIL"
                ratio = "N/A"
                bfs_obj = "N/A"
                dfs_obj = "N/A"
                obj_diff = "N/A"
                success = "0"
            
            print(f"{size:<6} {bfs_time:<12} {dfs_time:<12} {ratio:<10} {bfs_obj:<12} {dfs_obj:<12} {obj_diff:<12} {success:<8}")
        
        print("-"*120)
        
        # Summary statistics
        successful_sizes = [size for size in results.keys() if results[size]['successful_trials'] > 0]
        if successful_sizes:
            avg_bfs_times = [results[size]['avg_bfs_time'] for size in successful_sizes]
            avg_dfs_times = [results[size]['avg_dfs_time'] for size in successful_sizes]
            ratios = [results[size]['bfs_faster_ratio'] for size in successful_sizes]
            avg_bfs_objectives = [results[size]['avg_bfs_objective'] for size in successful_sizes]
            avg_dfs_objectives = [results[size]['avg_dfs_objective'] for size in successful_sizes]
            
            print(f"\nSUMMARY:")
            print(f"Average BFS time: {np.mean(avg_bfs_times):.4f}s")
            print(f"Average DFS time: {np.mean(avg_dfs_times):.4f}s")
            print(f"Average BFS/DFS ratio: {np.mean(ratios):.2f}")
            print(f"Average BFS objective: {np.mean(avg_bfs_objectives):.6f}")
            print(f"Average DFS objective: {np.mean(avg_dfs_objectives):.6f}")
            print(f"Successful sizes: {len(successful_sizes)}/{len(results)}")

    print_test_results(test_bfs_dfs_comparison(min_size=10, max_size=17, num_trials=5, method='mosek'))

    """