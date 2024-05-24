###############################################################################

# Required Libraries
import numpy as np
import warnings
warnings.filterwarnings('ignore', message = 'delta_grad == 0.0. Check if the approximated')
warnings.filterwarnings('ignore', message = 'Values in x were outside bounds during a minimize step, clipping to bounds')

from scipy.optimize import minimize, Bounds, NonlinearConstraint

###############################################################################

# Function: Fuzzy BWM
def fuzzy_bw_method(mic, lic, eps_penalty = 1, verbose = True): 
    priority_tuples = [(7/2, 4, 9/2), (5/2, 3, 7/2), (3/2, 2, 5/2), (2/3, 1, 3/2), (1, 1, 1)]
    ci              = [8.04, 6.69, 5.29, 3.8, 3.0 ]
    tuple_to_ci     = dict(zip(priority_tuples, ci))
    
    ###############################################
    
    def find_index(criteria_list, priority_tuples):
        for tup in priority_tuples:
            if tup in criteria_list:
                return criteria_list.index(tup)
        return None
    
    def generate_ordered_triplets(num_criteria):
        variables = np.zeros(3 * num_criteria)
        for i in range(0, num_criteria):
            x1                   = np.random.uniform(low = 0.0001, high = 1.0)
            x2                   = np.random.uniform(low = x1,     high = 1.0)
            x3                   = np.random.uniform(low = x2,     high = 1.0)
            variables[3 * i]     = x1
            variables[3 * i + 1] = x2
            variables[3 * i + 2] = x3
        fuzzy_set = [(variables[i], variables[i + 1], variables[i + 2]) for i in range(0, len(variables), 3)]
        weights   = np.array([(a + 4 * b + c) / 6 for a, b, c in fuzzy_set])
        total     = np.sum(weights)
        scale_fac = 1 / total
        variables = variables * scale_fac
        return variables
    
    ###############################################
    
    ib       = find_index(mic, priority_tuples)
    iw       = find_index(lic, priority_tuples)
    ci_value = tuple_to_ci.get(mic[ib])
    pairs_w  = [(iw, i) for i in range(0, len(lic)) if i != iw]
    pairs_b  = [(i, ib) for i in range(0, len(mic)) if i != ib and i != iw]
   
    ################################################
   
    def operation(wv, eps, vector, idx_a = 2, idx_b = 0, idx_m = 0):
        a, b, c = wv[idx_a]
        d, e, f = wv[idx_b]
        fn      = (a - vector[idx_m][0]*f - eps*f, b - vector[idx_m][1]*e - eps*e, c - vector[idx_m][2]*d - eps*d)
        return fn
    
    def target_function(variables):
        eps     = variables[-1] 
        cn1     = []
        wv      = [(1, 1, 1) for item in mic]
        penalty = 0
        j       = 0
        for i in range(0, len(wv)):
            wv[i] = (variables[j], variables[j+1], variables[j+2])
            j     = j + 3
        for i in range(0, len(pairs_w)):
            a, b, c = operation(wv = wv, eps = eps, vector = mic, idx_a = pairs_w[i][0], idx_b = pairs_w[i][1], idx_m = i)
            cn1.append( a)
            cn1.append( b)
            cn1.append( c)
            cn1.append(-a)
            cn1.append(-b)
            cn1.append(-c)
        for i in range(0, len(pairs_b)):
            a, b, c = operation(wv = wv, eps = eps, vector = lic, idx_a = pairs_b[i][0], idx_b = pairs_b[i][1], idx_m = i)
            cn1.append( a)
            cn1.append( b)
            cn1.append( c)
            cn1.append(-a)
            cn1.append(-b)
            cn1.append(-c)
        for item in cn1:
            if (item > eps):
                penalty = penalty + (item - eps) * 1
        penalty = penalty + eps * eps_penalty
        return penalty
    
    def LMU_constraint(variables):
        constraints = []
        for i in range(0, len(variables) - 1, 3):
            L, m, u = variables[i], variables[i+1], variables[i+2]
            constraints.append(L)  
            constraints.append(m - L)  
            constraints.append(u - m)
        return constraints

    def weights_constraint(variables):
        fuzzy_set = [(variables[i], variables[i + 1], variables[i + 2]) for i in range(0, len(variables)-1, 3)]
        w         = [(a + 4 * b + c) / 6 for a, b, c in fuzzy_set]
        return np.sum(w)
   
    constraint_1 = NonlinearConstraint(weights_constraint, 1, 1)
    constraint_2 = {'type': 'ineq', 'fun': LMU_constraint}
    constraints  = [constraint_1, constraint_2]
    
    ################################################
    
    np.random.seed(42)
    bounds   = Bounds([0]*len(mic)*3 + [0], [1]*len(mic)*3 + [1])
    n_starts = 50
    solution = None
    obj_fun  = np.inf
    for i in range(0, n_starts):  
        if (i == 0):
            guess = generate_ordered_triplets(len(mic))
            guess = np.append(guess, [0])
        else:
            guess = solution
        
        results = minimize(target_function, guess, method = 'trust-constr', bounds = bounds, constraints = constraints)
        if (results.fun < obj_fun):
            obj_fun  = results.fun
            solution = results.x
    f_weights     = [(solution[i], solution[i + 1], solution[i + 2]) for i in range(0, len(solution)-1, 3)]
    weights       = [(a + 4 * b + c) / 6 for a, b, c in f_weights]
    if (verbose == True):
        print('Epsilon Value:', np.round(results.x[-1], 4)) 
        print('CR: ', np.round(results.x[-1]/ci_value , 4))
    return  results.x[-1], results.x[-1]/ci_value, f_weights, weights

###############################################################################
