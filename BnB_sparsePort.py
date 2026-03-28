"""
Created on Tue Nov 14 14:41:26 2023

@author: sen
"""

''' BB only revisited '''


#%% Imports

import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt 
import time
import os
import scipy
import math
from scipy import io
from queue import PriorityQueue, LifoQueue
from numpy import linalg
from numpy import inf
from scipy.optimize import minimize
from time import process_time
import cvxpy as cp
# import gurobipy
# import mosek
from numpy import random,reshape,mean,ones,array,sign,arange,delete,log,tensordot,unique,fill_diagonal,shape,std,\
    zeros,where,linspace,load,diag,argsort,sqrt,eye,nonzero
from scipy.linalg import eigvalsh,ldl,det,eigh,inv,pinv,cholesky,norm
# from scipy.sparse.linalg import eigsh

    

def zeropadding(u,w,n):
    
    U =  np.zeros((n,1))  #Makes u with 4 significant level!!
    cnt = 0
    for i in range(n):
        if np.in1d(i,w):
            U[i] = u[cnt]
            cnt = cnt+1
        else:
            U[i] = 0
    return U

# maximum value of
# |arr[i] - arr[j]| + |i - j|
 
# Return maximum value of
# |arr[i] - arr[j]| + |i - j|
def findRho(arr, i):
    ans = 0;

    for j in range(len(arr)):
        if j!=i:
            for k in range(len(arr)):
                if k!=j and k!=i:
                    ans = ans if ans > abs( (r[i] - r[k])/(r[k]-r[j]) 
                                          )  else abs( (r[i] - r[k])/(r[k]-r[j]) )
    return ans
     
def findVarrho(arr,rBar,i):
    ans = inf;
     
    for j in range(len(arr)):
        if j!=i:
            for k in range(len(arr)):
                ans = ans if ans < abs( (rBar - r[j])/(r[i]-r[j]) 
                                      )  else abs( (rBar - r[j])/(r[i]-r[j]) )
    return ans

def twoToInfNorm(D):
    
    def objective(x):     
        return -np.linalg.norm(D@x,ord=inf)
    
    def constraint(x):
        return 1 - np.linalg.norm(x,ord=2)
    
   
   # make constraints into tuple of dicts as required by scipy 
    cons = {'type':'eq','fun':constraint}
    
    x0 = np.zeros((D.shape[0]))
    
    # perform the minimization with sequential least squares programming
    opt = minimize(objective,x0, constraints=cons,method='SLSQP',options={'disp': False})
    
    return -opt.fun
    
#%% Main
n = 500
# #take data 
# path = r'/home/busesen'
#path = '/Users/sen/Desktop/Bilkent/tez kod/Results txt files/datasets'
#os.chdir(path)
# mat = scipy.io.loadmat('DowJones2005Ret.mat') #21 companies
# mat = scipy.io.loadmat('ETFsRet.mat') #24 companies
# mat = scipy.io.loadmat('EuroBondsRet.mat') #62 companies
# mat = scipy.io.loadmat('WorldMixBondsRet.mat') #104 companies
# mat = scipy.io.loadmat('P_NIKKEI225_Weekly_199_realized.mat') #199 companies
#mat = scipy.io.loadmat('SP500.mat') #442 companies
# profitArr = []
#X = np.array(mat['RR'])
np.random.seed(29)
X = np.random.rand(600, 600)
dayNo = X.shape[0]


iteration = 1

rBar = 0.001
level = 0.2 #This should be between0 and 1. If it is set to 1, BnB gives the optimal solution.
beta = 1*10**(-1)
relErr = 10**(-10)
timeout = 3600*24

col = X[0].shape
print("Currently running: ", '  sparsity level ', level*100, '% and with beta ', beta, 'and with dimension', col)

       
for k in  tqdm( range(iteration) ):
# for k in  tqdm( range(dayNo-n-1) ):
    # Xreduced = X[k:n+k]
    Xreduced = X[k:n+k,0:20]  #If you want to work on a reduced dimension, uncomment.
    # Xtest = X[n+k]
    Xtest = X[n+k,0:20]  #If you want to work on a reduced dimension, uncomment.
    row,col = Xreduced.shape
    r = np.zeros((col,1))
    
    #r vectors
    for i in range(col):
        r[i] = sum(Xreduced[:,i])/col
       
    
    D = Xreduced.T@Xreduced/row*100 + (1e-4)*np.identity(col)
    # D = Xreduced.T@Xreduced/row*100 + (1e-2)*np.identity(col)  
    # 1e-3 or 1e-4 was not enough for 199 weekly data. 1e-2 will do it! For the rest, use 1e-4

    
    DD = np.diag(D)
   
    
    #Algorithm
    supp = []
    psupp = list(np.arange(col,dtype=int))
    
    w = supp + psupp
    
    Dw = D[w,:][:,w]
    Lw = np.linalg.cholesky(Dw)
    Lw_inv = np.linalg.inv(Lw)
    Dw_inv = Lw_inv.T @ Lw_inv
    
    # D_inv = np.linalg.inv(D)
    rw = r[w,0:1]
    
    onew = np.ones((len(w),1))
    a = onew.T @ (Dw_inv @ onew)
    b = onew.T @ (Dw_inv @ rw)
    c = rw.T @ (Dw_inv @ rw)
    
    d1 = (c-b*rBar)/(a*c-b**2)
    d2 = -(a*rBar-b)/(a*c-b**2)
    uw = d1*Dw_inv @ onew - d2* Dw_inv @rw
    
    
    lb = uw.T@Dw@uw + beta*len(supp) 
    ub = 10**20
    
    bnd = np.zeros((len(w),1))
    rho = np.zeros((len(w),1))
    varrho = np.zeros((len(w),1))
   
    
    CPU_start1 = process_time()
    start1 = time.time()
    
    '''Deleting the farest supports from bound'''
    #ww, vv = np.linalg.eig(D)
    norm_val = twoToInfNorm(D)
    for i in range(len(w)):     
        if len(w) == 2:
            bnd[i] = findVarrho(rw, rBar, i)           
            
        else:
            rho[i] = findRho(rw,i)
            bnd[i] = np.sqrt(beta)/( np.linalg.norm(Lw[:,i]) + 2*rho[i]*norm_val )
            
            
    dist_from_bnd_unsorted = abs(uw) - bnd
    num_of_zeros = col - math.floor(col*level)
    dist_from_bnd = np.array( sorted(dist_from_bnd_unsorted)[0:num_of_zeros] )
    supp_tobe_deleted = np.nonzero(np.in1d(dist_from_bnd_unsorted, dist_from_bnd))[0]
    
    # delete those indices from possible support
    psupp = np.delete(psupp,np.nonzero(np.in1d(psupp, supp_tobe_deleted))[0])
    print("psupp: ", psupp)
    """ Bound Calculation Ended Here"""
    
    
    
    ub = 10**20
    q = PriorityQueue()
    #q = LifoQueue()
    q.put([lb,ub,0,supp,psupp,uw,d1,d2])
    
    global_ub = ub + 1e-4
    global_supp = []
    count = 0
    
    """Branch and Bound Algorithm """
    while q.qsize() >= 1:
        [lb,ub,_,supp,psupp,x1,v1,v2] = q.get()
        count += 1
        if ub - global_ub < relErr:
            global_ub = ub
            global_supp = supp
            global_u = x1
        if global_ub <= lb:
            break
        if len(psupp) == 0:
            continue
        else:
            # rv = r[psupp,0]
            # var = DD[psupp]
            # bb_dec = var/rv
            # bb_ind = np.argmin(bb_dec)
            # ind = psupp[bb_ind]
           
            if len(supp) >= 2:
                grad =  D[psupp,:][:,supp] @ x1 + v1 * np.ones([len(psupp),1]) + v2 * r[psupp,0:1]
                bb_ind = np.argmax(abs(grad[:,0]))
                ind = psupp[bb_ind]
                
            else:
                rv = r[psupp,0]
                var = DD[psupp]
                bb_dec = var/(rv+0.00000001)
                bb_ind = np.argmin(bb_dec)
                ind = psupp[bb_ind]
              
            left_supp = supp + [ind]
            psupp = list(np.delete(psupp,bb_ind))
            
            if len(supp) + len(psupp) >= 2:
                w = supp + psupp
                Dw = D[w,:][:,w]
                Lw = np.linalg.cholesky(Dw)
                Lw_inv = np.linalg.inv(Lw)
                Dw_inv = Lw_inv.T @ Lw_inv
               
                rw = r[w,0:1] 
                onew = np.ones((len(w),1))
                
                a = onew.T @ (Dw_inv @ onew)
                b = onew.T @ (Dw_inv @ rw)
                c = rw.T @ (Dw_inv @ rw)
                
                d1 = (c-b*rBar)/(a*c-b**2)
                d2 = -(a*rBar-b)/(a*c-b**2)
                uw = d1*Dw_inv @ onew - d2* Dw_inv @rw
                
                right_lb = uw.T@Dw@uw + beta*len(supp)
                q.put([right_lb,ub,np.random.rand(),supp,psupp,x1,v1,v2])
               
                
            if len(left_supp) >= 2:
                w = left_supp
                Dw = D[w,:][:,w]
                Lw = np.linalg.cholesky(Dw)
                Lw_inv = np.linalg.inv(Lw)
                Dw_inv = Lw_inv.T @ Lw_inv
                
                rw = r[w,0:1]
                onew = np.ones((len(w),1))
                
                a = onew.T @ (Dw_inv @ onew)
                b = onew.T @ (Dw_inv @ rw)
                c = rw.T @ (Dw_inv @ rw)
                
                d1 = (c-b*rBar)/(a*c-b**2)
                d2 = -(a*rBar-b)/(a*c-b**2)
                uw = d1*Dw_inv @ onew - d2* Dw_inv @rw
                    
                left_ub = uw.T@Dw@uw + beta*len(left_supp)
                q.put([lb+beta,left_ub,np.random.rand(),left_supp,psupp,uw,d1,d2])
                
                
            else:
                q.put([lb+beta,ub,np.random.rand(),left_supp,psupp,x1,v1,v2])
                
            
        if time.time() - start1 > timeout:
            break
        
      
        
    u_opt = zeropadding(global_u, global_supp, col) 
    optVal = u_opt.T@D@u_opt + beta*np.count_nonzero(u_opt)
    print("optVal: ", optVal)
    # profitVec = Xtest@u_opt
    # nextDayProfit = sum(profitVec)
    # profitArr.append(nextDayProfit)
     
    print('\n opt value', global_ub) 
    # print('opt u is ' , u_opt)
    print('opt u is ' ,global_u)
    print('opt u shape ' ,global_u.shape)
    print("count", count)
    
    end1 = time.time()
    CPU_end1 = process_time()
    
    
    CPU_duration_BB = CPU_end1 - CPU_start1
    Wall_duration_BB = end1 - start1
    
 
    
 #%% 
#     """Mixed Integer Quadratic Formulation"""
    
#     # Construct the problem.
#     M=2
#     x = cp.Variable(col)
#     z = cp.Variable(col,  boolean=True)
#     # D = D + np.identity(col)
#     objective = cp.Minimize( cp.quad_form(x, D) + beta*np.ones((col,1)).T @ z ) 
#     constraints = [-M*z <= x, x <= M*z, cp.sum(x) == 1, r.T@x == rBar]
#     prob = cp.Problem(objective, constraints)
    
    
#     mosek_params = {
#     #mosek.dparam.basis_tol_x: 1e-8,
#     # "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
#     # "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
#     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": relErr,
#     "MSK_DPAR_MIO_MAX_TIME": timeout
#     # "MSK_IPAR_INTPNT_MAX_ITERATIONS": 400
# }
#     result = prob.solve(solver=cp.MOSEK, mosek_params=mosek_params)
#     print('Opt val of MIQP is ', result)
#     print('\n Opt u from MIQP is ', x.value[np.nonzero(x.value)])
    
    
    
#     opt_val_diff = global_ub-prob.value
    
#     errorPercent = (opt_val_diff)/prob.value*100
#     print('Error percent of instance:',k, ' is ', errorPercent)



 