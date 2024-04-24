###############################################################################

# Required Libraries
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')
warnings.filterwarnings('ignore', message = 'Values in x were outside bounds during a minimize step, clipping to bounds')

from scipy.optimize import minimize, Bounds, LinearConstraint

###############################################################################

# Function: FUCOM (Full Consistency Method)
def fucom_method(criteria_rank, criteria_priority, sort_criteria = True, verbose = True):
    
    ################################################
    
    def extract_number(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None
    
    def target_function(variables):
        variables       = np.array(variables)
        ratios_1        = variables[:-1] / variables[1:]
        target_ratios_1 = np.array(criteria_priority[1:]) / np.array(criteria_priority[:-1])
        chi_1           = np.abs(ratios_1 - target_ratios_1)
        ratios_2        = variables[:-2] / variables[2:]
        target_ratios_2 = np.array(criteria_priority[2:]) / np.array(criteria_priority[:-2])
        chi_2           = np.abs(ratios_2 - target_ratios_2)
        chi             = np.hstack((chi_1, chi_2))
        return np.max(chi)
    
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
