###############################################################################

# Required Libraries
import numpy as np

from scipy.stats import norm

###############################################################################

# Function: Profiles to Quantiles
def profiles_to_quantiles(profiles = None, decision_matrix = None, num_cat = None):
    if profiles is not None and isinstance(profiles, dict) and len(profiles) > 0:
        return np.array([np.mean(np.array(profiles[c]), axis = 0) for c in profiles])
    if decision_matrix is not None:
        if num_cat is not None:
            num_categories = min(num_cat, decision_matrix.shape[0])
        else:
            num_categories = min(5, decision_matrix.shape[0])
    quantile_levels = np.linspace(0, 1, num_categories + 2)[1:-1]
    return np.array([np.quantile(decision_matrix, q, axis = 0) for q in quantile_levels])

# Fucntion: CPP Tri Method
def cpp_tri_method(decision_matrix, weights = None, profiles = None, num_cat = None, indep_criteria = True, rule = 'strict', verbose = True):
    num_alternatives, num_criteria = decision_matrix.shape
    class_profiles                 = profiles_to_quantiles(profiles, decision_matrix, num_cat)
    num_profiles                   = class_profiles.shape[0]
    if weights is None:
        weights = np.ones(num_criteria) / num_criteria
    else:
        weights = np.array(weights)
    weights             = weights / np.sum(weights)
    std_devs            = np.std(decision_matrix, axis = 0, ddof = 1)
    diff                = (decision_matrix[:, np.newaxis, :] - class_profiles[np.newaxis, :, :]) / (std_devs * np.sqrt(2))
    prob_above_matrix   = norm.cdf(diff)
    prob_below_matrix   = 1 - prob_above_matrix
    weighted_prob_above = prob_above_matrix * weights
    weighted_prob_below = prob_below_matrix * weights
    if (indep_criteria == True):
        overall_prob_above = np.prod(weighted_prob_above, axis = 2)
        overall_prob_below = np.prod(weighted_prob_below, axis = 2)
    else:
        overall_prob_above = np.min( weighted_prob_above, axis = 2)
        overall_prob_below = np.min( weighted_prob_below, axis = 2)
    results = {}
    for i in range(0, num_alternatives):
        best_class = None
        for j in range(0, num_profiles):
            if (overall_prob_above[i, j] - overall_prob_below[i, j] < 0):
                break
            if (j == 0):
                best_class = 1
            else:
                if (np.abs(overall_prob_above[i, j] - overall_prob_below[i, j]) < np.abs(overall_prob_above[i, j-1] - overall_prob_below[i, j-1])):
                    best_class = j + 1
                else:
                    best_class = j
                if (rule == 'central'):
                    best_class = (j + (j + 1)) // 2 
                elif (rule == 'benevolent'):
                    best_class = max(j + 1, 1)
        if (best_class is None):
            best_class = 1  
        results[f'a{i+1}'] = best_class
    if (verbose == True):
        for key, value in results.items():
            print(key, ':', value)
    classification = [results[k] for k in sorted(results, key=lambda x: int(x[1:]))]
    return classification

###############################################################################
