###############################################################################

# Required Libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')

from scipy.optimize import minimize, Bounds, LinearConstraint

###############################################################################

# Function: BWM
def bw_method(mic, lic, eps_penalty = 1, verbose = True):
    cr = []
    mx = np.max(mic) 
    if (mx == 1):
        cr = 1
    else:
        for i in range(0, mic.shape[0]):
            cr.append((mic[i] * lic[i] - mx)/(mx**2 - mx))
    cr        = np.max(cr)
    threshold = [0, 0, 0, 0.1667, 0.1898, 0.2306, 0.2643, 0.2819, 0.2958, 0.3062]
    if (verbose == True):
        if (cr <= threshold[mx]):
            print('CR:', np.round(cr, 4), '(The Consistency Level is Acceptable)')
        else:
            print('CR:', np.round(cr, 4), '(The Consistency Level is Not Acceptable)')
    
    ################################################
    def target_function(variables):
        eps     = variables[-1]
        wx      = variables[np.argmin(mic)]
        wy      = variables[np.argmin(lic)]
        cons_1  = []
        cons_2  = []
        penalty = 0
        for i in range(0, mic.shape[0]):
            cons_1.append(wx - mic[i] * variables[i])
        cons_1.extend([-item for item in cons_1])
        for i in range(0, lic.shape[0]):
            cons_2.append(variables[i] - lic[i] * wy)
        cons_2.extend([-item for item in cons_2])
        cons = cons_1 + cons_2
        for item in cons:
            if (item > eps):
                penalty = penalty + (item - eps) * 1
        penalty = penalty + eps * eps_penalty
        return penalty
    ################################################
    
    np.random.seed(42)
    variables = np.random.uniform(low = 0.001, high = 1.0, size = mic.shape[0])
    variables = variables / np.sum(variables)
    variables = np.append(variables, [0])
    bounds    = Bounds([0]*mic.shape[0] + [0], [1]*mic.shape[0] + [1])
    w_cons    = LinearConstraint(np.append(np.ones(mic.shape[0]), [0]), [1], [1])
    results   = minimize(target_function, variables, method = 'trust-constr', bounds = bounds, constraints = [w_cons])
    weights   = results.x[:-1]
    if (verbose == True):
        print('Epsilon Value:', np.round(results.x[-1], 4))
    return weights

###############################################################################
