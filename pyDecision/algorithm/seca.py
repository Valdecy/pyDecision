###############################################################################

# Required Libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')
warnings.filterwarnings('ignore', message = 'Values in x were outside bounds during a minimize step, clipping to bounds')

from scipy.optimize import minimize, Bounds, LinearConstraint

###############################################################################


# Function: SECA (Simultaneous Evaluation of Criteria and Alternatives)
def seca_method(dataset, criterion_type, beta = 3):
    X = np.copy(dataset)/1.0
    N = X.shape[1]
    for j in range(0, N):
        if (criterion_type[j] == 'max'):
            X[:,j] = np.min(X[:,j]) / X[:,j]
        else:
            X[:,j] = X[:,j] / np.max(X[:,j])
    std      = (np.sum((X - X.mean())**2, axis = 0)/(X.shape[0] - 1))**(1/2)
    std      = std/np.sum(std)
    sim_mat  = np.corrcoef(X.T)
    sim_mat  = np.sum(1 - sim_mat, axis = 1)
    sim_mat  = sim_mat/np.sum(sim_mat)
    
    ################################################
    
    def target_function(variables):
        Lmb_a  = np.min(np.sum(X * variables, axis = 0))
        Lmb_b  = np.sum((variables - std)**2, axis = 0)
        Lmb_c  = np.sum((variables - sim_mat)**2, axis = 0)
        Lmb    = Lmb_a - beta*(Lmb_b + Lmb_c)
        return -Lmb
    
    ################################################
    
    np.random.seed(42)
    variables   = np.random.uniform(low = 0.001, high = 1.0, size = N)
    variables   = variables / np.sum(variables)
    bounds      = Bounds(0.0001, 1.0)
    constraints = LinearConstraint(np.ones(N), 1, 1)
    results     = minimize(target_function, variables, method = 'SLSQP', constraints = constraints, bounds = bounds)
    weights     = results.x
    return weights 

###############################################################################