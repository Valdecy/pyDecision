###############################################################################

# Required Libraries
import numpy as np

from scipy.optimize import minimize, Bounds

###############################################################################

# Function: IDOCRIW (Integrated Determination of Objective CRIteria Weights)
def idocriw_method(dataset, criterion_type, verbose = True):
    X    = np.copy(dataset)/1.0
    X    = X/X.sum(axis = 0)
    X_ln = np.zeros(X.shape[1])
    X_r  = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        adj_col = np.where(X[:, j] == 0, 1e-9, X[:, j])
        X_ln[j] = np.sum(adj_col * np.log(adj_col))
        X_ln[j] = X_ln[j]*(-1/np.log(X.shape[1]))
    d = 1 - X_ln
    w = d/np.sum(d)
    for i in range(0, len(criterion_type)):
        if (criterion_type[i] == 'min'):
           X_r[:,i] = dataset[:,i].min() / X_r[:,i]
    X_r   = X_r/X_r.sum(axis = 0)   
    a_max = X_r.max(axis = 0) 
    A     = np.zeros((dataset.shape[1], dataset.shape[1]))
    np.fill_diagonal(A, a_max)
    for k in range(0, a_max.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i    = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]     
    a_max_ = A.max(axis = 0) 
    P      = np.copy(A)    
    for j in range(0, P.shape[1]):
        P[:,j] = (-P[:,j] + a_max_[j])/a_max[j]
    WP     = np.copy(P)
    np.fill_diagonal(WP, -P.sum(axis = 0))
    
    ################################################
    def target_function(variables):
        WP_s = np.copy(WP)
        for i in range(0, WP.shape[0]):
            for j in range(0, WP.shape[1]):
                if (WP_s[i, j] != 0):
                    WP_s[i, j] = WP_s[i, j]*variables[j]
        total = np.sum((WP_s.sum(axis = 1)))
        return total
    ################################################
    
    variables = np.ones(WP.shape[1])
    bounds    = Bounds([0.0000001]*len(variables), [1]*len(variables))
    results   = minimize(target_function, variables, method = 'L-BFGS-B', bounds = bounds)
    weights   = results.x
    weights   = weights * w
    weights   = weights / np.sum(weights)
    return weights

###############################################################################
