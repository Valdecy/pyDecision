###############################################################################

# Required Libraries
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')
warnings.filterwarnings('ignore', message = 'Values in x were outside bounds during a minimize step, clipping to bounds')

from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

###############################################################################

# Function: Fuzzy FUCOM (Full Consistency Method)
def fuzzy_fucom_method(criteria_rank, criteria_priority, n_starts = 250, sort_criteria = True, verbose = True):
    
    ################################################
    
    def extract_number(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None
    
    def generate_ordered_triplets(num_criteria):
        variables = np.zeros(3 * num_criteria)
        for i in range(0, num_criteria):
            x1                   = np.random.uniform(low = 0.0001, high = 1.0)
            x2                   = np.random.uniform(low = x1,    high = 1.0)
            x3                   = np.random.uniform(low = x2,    high = 1.0)
            variables[3 * i]     = x1
            variables[3 * i + 1] = x2
            variables[3 * i + 2] = x3
        fuzzy_set = [(variables[i], variables[i + 1], variables[i + 2]) for i in range(0, len(variables), 3)]
        weights   = np.array([(a + 4 * b + c) / 6 for a, b, c in fuzzy_set])
        total     = np.sum(weights)
        scale_fac = 1 / total
        variables = variables * scale_fac
        return variables
    
    ################################################
    
    def target_function(variables):
        fuzzy_set       = [(variables[i], variables[i + 1], variables[i + 2]) for i in range(0, len(variables), 3)]
        skip            = 1
        pairs_1         = [(i, i + skip) for i in range(len(fuzzy_set) - skip)]
        comparative_imp = []
        for i, j in pairs_1:
            a, b, c = criteria_priority[j]
            d, e, f = criteria_priority[i]
            comparative_imp.append((a/f, b/e, c/d))
        pairs_2         = [(i, i + skip) for i in range(len(comparative_imp) - skip)]
        for i, j in pairs_2:
            a, b, c = comparative_imp[i]
            d, e, f = comparative_imp[j]
            comparative_imp.append((a*d, b*e, c*f))
        chi             = []
        skip            = 2
        pairs_3         = [(i, i + skip) for i in range(len(fuzzy_set) - skip)]
        pairs           = pairs_1 + pairs_3
        count           = 0
        for i, j in pairs:
            a, b, c = fuzzy_set[i]
            d, e, f = fuzzy_set[j]
            x, y, z = comparative_imp[count]
            count   = count + 1
            chi.append( abs((a / f) - x) ) 
            chi.append( abs((b / e) - y) )
            chi.append( abs((c / d) - z) ) 
        return np.max(chi)
    
    def weights_constraint(variables):
        fuzzy_set = [(variables[i], variables[i + 1], variables[i + 2]) for i in range(0, len(variables), 3)]
        w         = [(a + 4 * b + c) / 6 for a, b, c in fuzzy_set]
        return np.sum(w)

    ################################################

    lower_bounds  = []
    upper_bounds  = []
    for i in range(0, len(criteria_priority) * 3, 3):
        lower_bounds = lower_bounds + [0.0001, 0.0001, 0.0001]
        upper_bounds = upper_bounds + [1, 1, 1]
    bounds        = Bounds(lower_bounds, upper_bounds)
    constraints_1 = NonlinearConstraint(weights_constraint, 1, 1)
    const_matrix  = []
    constraint_lb = []
    constraint_ub = []
    for i in range(0, len(criteria_priority) * 3, 3):
        row        = np.zeros(len(criteria_priority) * 3)
        row[i]     = -1
        row[i + 1] =  1
        const_matrix.append(row)
        constraint_lb.append(0)
        constraint_ub.append(np.inf)
        row        = np.zeros(len(criteria_priority)*3)
        row[i + 1] = -1
        row[i + 2] =  1
        const_matrix.append(row)
        constraint_lb.append(0)
        constraint_ub.append(np.inf)
    constraints_2 = LinearConstraint(const_matrix, constraint_lb, constraint_ub)
    chi           = np.inf
    solution      = None
    for i in range(0, n_starts):  
        if (i == 0):
            guess = generate_ordered_triplets(len(criteria_priority))
        else:
            guess = solution
        result = minimize(target_function, guess, method = 'SLSQP', constraints = [constraints_1, constraints_2], bounds = bounds)
        if (result.fun < chi):
            chi      = result.fun
            solution = result.x
    f_weights     = [(solution[i], solution[i + 1], solution[i + 2]) for i in range(0, len(solution), 3)]
    weights       = [(a + 4 * b + c) / 6 for a, b, c in f_weights]
    if (sort_criteria == True):
        idx       = sorted(range(0, len(criteria_rank)), key = lambda x: extract_number(criteria_rank[x]))
        f_weights = [f_weights[i] for i in idx]
        weights   = [  weights[i] for i in idx]
    if (verbose == True):
        print('Chi:', np.round(chi, 4))
    return f_weights, weights

###############################################################################
