"""
Improved Branch-and-Bound implementation for Robust Sparse Portfolio Optimization (RMVP1).

This implementation adds Sherman-Morrison updates for left branch only, maintaining
the same structure as mainRMVP1BnB.
"""

import numpy as np
from queue import PriorityQueue, LifoQueue
from main import solveRMVP1, rmvp1_lagrangian_gradient, branchVariable_rmvp1, zeropadding


def sherman_morrison_remove(A_inv, idx):
    """
    Remove row/column at index idx from the inverse matrix using Sherman-Morrison.
    """
    m = A_inv.shape[0]
    if idx < 0 or idx >= m:
        raise ValueError(f"Index {idx} out of range [0, {m-1}]")
    
    if m == 1:
        raise ValueError("Cannot remove from 1x1 matrix")
    
    mask = np.ones(m, dtype=bool)
    mask[idx] = False
    
    A_ii = A_inv[idx, idx]
    if abs(A_ii) < 1e-12:
        A_rest = A_inv[mask][:, mask]
        A_rest_col = A_inv[mask, idx:idx+1]
        A_rest_row = A_inv[idx:idx+1, mask]
        A_new_inv = A_rest - (A_rest_col @ A_rest_row) / (A_ii + 1e-12)
    else:
        A_rest = A_inv[mask][:, mask]
        A_rest_col = A_inv[mask, idx:idx+1]
        A_rest_row = A_inv[idx:idx+1, mask]
        A_new_inv = A_rest - (A_rest_col @ A_rest_row) / A_ii
    
    return A_new_inv


def sherman_morrison_add(A_inv, D_full, mapping, new_idx):
    """
    Add a new row/column to the inverse matrix using Sherman-Morrison formula.
    """
    m = A_inv.shape[0]
    mapping = np.array(mapping)
    
    v = D_full[mapping, new_idx].reshape(-1, 1)
    d = D_full[new_idx, new_idx]
    
    vT_Ainv = v.T @ A_inv
    Ainv_v = A_inv @ v
    denominator = d - (vT_Ainv @ v)[0, 0]
    
    if abs(denominator) < 1e-12:
        new_mapping = list(mapping) + [new_idx]
        new_mapping = np.array(new_mapping)
        D_new = D_full[new_mapping][:, new_mapping]
        return np.linalg.inv(D_new)
    
    rank_one = (Ainv_v @ vT_Ainv) / denominator
    
    B_inv = np.zeros((m + 1, m + 1))
    B_inv[:m, :m] = A_inv + rank_one
    B_inv[:m, m:m+1] = -Ainv_v / denominator
    B_inv[m:m+1, :m] = -vT_Ainv / denominator
    B_inv[m, m] = 1.0 / denominator
    
    return B_inv


def solveRMVP1_with_inv(D_inv, tau_bar, tau, gamma, dual=True):
    """
    Same as solveRMVP1 but uses precomputed inverse instead of computing it.
    """
    tau = tau.reshape(-1) if tau.ndim == 2 else tau
    
    D = np.linalg.inv(D_inv)
    
    H = np.sqrt(tau.T @ D_inv @ tau)
    
    scale = tau_bar / (H * (H - gamma))
    u_hat = scale * (D_inv @ tau)
    obj_val = float(u_hat.T @ D @ u_hat)
    
    if dual:
        lambda_val = 2 * tau_bar / ((H - gamma)**2)
    else:
        lambda_val = 0
    return u_hat, obj_val, lambda_val


def _build_mapping_from_parent(parent_mapping, target_set):
    """
    Preserve parent order for common indices, append missing ones at the end.
    This guarantees target_set coverage while keeping a stable order.
    """
    in_parent = [idx for idx in parent_mapping if idx in target_set]
    missing = [idx for idx in target_set if idx not in set(parent_mapping)]
    return in_parent + missing


def mainRMVP1BnB_improved(D, tau, tau_bar, gamma, beta, method='mosek', branch_rule='max_lagrangian_grad', traverse_rule='bfs', time_limit=None):
    """
    Branch-and-bound algorithm for CP-RMVP problem:
        min_{x in R^n} x^T D x  s.t.  tau^T x - gamma * ||x||_D >= tau_bar
    
    Parameters
    ----------
    D : (n, n) array_like
        Positive semidefinite symmetric matrix D.
    tau : (n,) array_like
        The excess mean return vector estimate tau (ccr in the problem).
    tau_bar : float
        Scalar parameter tau_bar (bar{ccr} in the problem).
    gamma : float
        Ellipsoidal uncertainty radius γ.
    beta : float
        Sparsity penalty parameter.
    method : str, optional
        Solver method (not used currently).
    branch_rule : str, optional
        Branching rule ('max_lagrangian_grad').
    traverse_rule : str, optional
        Traversal rule ('bfs' or 'dfs').
    time_limit : float, optional
        Time limit for the algorithm.
    
    Returns
    -------
    global_x : (n,) ndarray
        Optimal solution.
    global_ub : float
        Global upper bound (optimal objective value).
    global_supp : list
        Support of the optimal solution.
    count : int
        Number of nodes explored.
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
    
    # Solve initial relaxed problem
    x_init, lb, lambda_init = solveRMVP1(D, tau_bar, tau, gamma)


    global_supp = []
    Ssupp = []
    Psupp = list(range(D.shape[1]))
    DD = np.diag(D)
    
    # Initialize inverse matrix and mapping for root node
    mapping = list(range(D.shape[1]))  # Full mapping: Ssupp + Psupp
    inv_matrix = np.linalg.inv(D)
    
    q.put([lb,ub,0,Ssupp,Psupp,x_init,lambda_init,inv_matrix,mapping])

    count = 0
    while q.qsize() >= 1:
        [lb,ub,_,Ssupp,Psupp,x1,lambda_val,inv_matrix,mapping] = q.get()
        
        #print("count: ", count)
        #print("lb: ", lb)
        #print("global_ub: ", global_ub)
        #print("Ssupp: ", Ssupp)
        #print("Psupp: ", Psupp)
        #print("x1: ", x1)
        #print("lambda_val: ", lambda_val)

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
        if len(Psupp) == 0:
            continue
        else:
            if len(Ssupp) >= 1:
                ind, bb_ind = branchVariable_rmvp1(x1, D, tau, tau_bar, gamma, Ssupp, Psupp, lambda_val, branch_rule)
            else:
                tau_Psupp = tau[Psupp]
                var = DD[Psupp]
                bb_dec = var/(tau_Psupp+1e-8)
                bb_ind = np.argmin(bb_dec)
                ind = Psupp[bb_ind]

            left_supp_set = set(Ssupp + [ind])
            left_supp = _build_mapping_from_parent(mapping, left_supp_set)
            Psupp = list(np.delete(np.array(Psupp), bb_ind))



            if len(Ssupp) + len(Psupp) >= 1: # Since right_sup = Ssupp + Psupp
                right_set = set(Ssupp + Psupp)
                mapping_right = _build_mapping_from_parent(mapping, right_set)
                D_w = D[mapping_right][:, mapping_right]
                tau_w = tau[mapping_right]

                x_w_opt, right_lb, lambda_val = solveRMVP1(D_w, tau_bar, tau_w, gamma) #return inv D_w too
                right_lb = right_lb + beta * len(Ssupp)

                inv_right = None
                removed = [idx for idx in mapping if idx not in right_set]
                added = [idx for idx in mapping_right if idx not in set(mapping)]
                if len(removed) == 1 and len(added) == 0:
                    pos = mapping.index(removed[0])
                    inv_right = sherman_morrison_remove(inv_matrix, pos)
                elif len(removed) == 0 and len(added) == 1:
                    inv_right = sherman_morrison_add(inv_matrix, D, mapping, added[0])
                else:
                    inv_right = np.linalg.inv(D_w)

                q.put([right_lb,ub,np.random.rand(),Ssupp,Psupp,x1,lambda_val,inv_right,mapping_right])
                

            
            if len(left_supp) >= 1:
                # Left child: use Sherman-Morrison only if change is exactly one index.
                inv_left = None
                removed = [idx for idx in mapping if idx not in left_supp_set]
                added = [idx for idx in left_supp if idx not in set(mapping)]
                if len(removed) == 1 and len(added) == 0:
                    pos = mapping.index(removed[0])
                    inv_left = sherman_morrison_remove(inv_matrix, pos)
                elif len(removed) == 0 and len(added) == 1:
                    inv_left = sherman_morrison_add(inv_matrix, D, mapping, added[0])
                else:
                    D_left = D[left_supp][:, left_supp]
                    inv_left = np.linalg.inv(D_left)

                tau_left = tau[left_supp]
                x_w_opt, left_ub, lambda_val = solveRMVP1_with_inv(inv_left, tau_bar, tau_left, gamma)
                left_ub = left_ub + beta * len(left_supp)

                q.put([lb+beta,left_ub,np.random.rand(),left_supp,Psupp,x_w_opt,lambda_val,inv_left,left_supp])
            
            else:
                q.put([lb+beta,ub,np.random.rand(),left_supp,Psupp,x1,lambda_val,inv_matrix,mapping])

    global_supp = sorted(global_supp)
    #print("count: ", count)
    return global_x, global_ub, global_supp, count
