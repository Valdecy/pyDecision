###############################################################################

# Required Libraries
import numpy as np
from ..util import extract_number

import re
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')
warnings.filterwarnings('ignore', message = 'Values in x were outside bounds during a minimize step, clipping to bounds')

from scipy.optimize import minimize, Bounds, LinearConstraint

###############################################################################

# Function: FUCOM (Full Consistency Method)
def fucom_method(criteria_rank, criteria_priority, sort_criteria = True, verbose = True):
    
    ################################################
    
    
    np.random.seed(42)
    variables   = np.random.uniform(low = 0.001, high = 1.0, size = len(criteria_priority))
    variables   = variables / np.sum(variables)
    bounds      = Bounds(0.0001, 1.0)
    constraints = LinearConstraint(np.ones(len(criteria_priority)), 1, 1)
    results     = minimize(target_function, variables, method = 'SLSQP', constraints = constraints, bounds = bounds)
    weights     = results.x
    if (sort_criteria == True):
        idx     = sorted(range(0, len(criteria_rank)), key = lambda x: extract_number(criteria_rank[x]))
        weights = results.x[idx]
    if (verbose == True):
        print('Chi:', np.round(results.fun, 4))
    return weights

###############################################################################
