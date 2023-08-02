###############################################################################

# Required Libraries
import numpy as np

from scipy.optimize import minimize, Bounds

###############################################################################

# Function: BWM
def bw_method(dataset, mic, lic, verbose = True):
    X         = np.copy(dataset)/1.0
    best      = np.where(mic == 1)[0][0]
    worst     = np.where(lic == 1)[0][0]
    pairs_b   = [(best, i)  for i in range(0, mic.shape[0])]
    pairs_w   = [(i, worst) for i in range(0, mic.shape[0]) if (i, worst) not in pairs_b]
    
    ################################################
    def target_function(variables):
        eps = [float('+inf')]
        for pair in pairs_b:
            i, j = pair
            diff = abs(variables[i] - variables[j]*mic[j])
            if ( i != j):
                eps.append(diff)
        for pair in pairs_w:
            i, j = pair
            diff = abs(variables[i] - variables[j]*lic[j])
            if ( i != j):
                eps.append(diff)
        if ( np.sum(variables) == 0):
            eps = float('+inf')
        else:
            eps = max(eps[1:])
        return eps
    ################################################
    
    variables = np.ones(X.shape[1])
    bounds    = Bounds([0.0000001]*len(variables), [1]*len(variables))
    results   = minimize(target_function, variables, method = 'L-BFGS-B', bounds = bounds)
    weights   = results.x
    weights   = weights/np.sum(weights)
    return weights

###############################################################################
