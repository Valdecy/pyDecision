###############################################################################

# Required Libraries
import numpy as np

from scipy.stats import beta, logistic, norm

###############################################################################

# Function: Profiles to Quantiles
def profiles_to_quantiles(profiles = None, decision_matrix = None, num_cat = None):
    if profiles is not None and isinstance(profiles, dict) and len(profiles) > 0:
        prof_arrays = []
        for c in profiles:
            arr = np.atleast_2d(np.array(profiles[c], dtype = float))
            prof_arrays.append(arr.mean(axis = 0))
        return np.vstack(prof_arrays)
    
    decision_matrix = np.asarray(decision_matrix, dtype = float)
    
    if num_cat is not None:
        num_categories = int(min(num_cat, decision_matrix.shape[0]))
    else:
        num_categories = int(min(5,       decision_matrix.shape[0]))

    quantile_levels = np.linspace(0, 1, num_categories + 2)[1:-1]
    class_profiles  = []
    for q in quantile_levels:
        class_profiles.append(np.quantile(decision_matrix, q, axis = 0))
    return np.vstack(class_profiles)

# Funtion: Distributions
def _compute_prob_above(diff, dist = "normal"):
    diff = np.asarray(diff, float)

    if dist == "normal":
        return norm.cdf(diff)

    elif dist == "logit":
        return logistic.cdf(diff)

    elif dist == "beta-pert":
        z_min      = -4.0
        z_max      = 4.0
        u          = (np.clip(diff, z_min, z_max) - z_min) / (z_max - z_min)
        alpha      = 3.0
        beta_param = 3.0
        return beta.cdf(u, alpha, beta_param)

    elif dist == "empirical":
        n_alt, n_prof, n_crit = diff.shape
        prob                  = np.empty_like(diff)
        for k in range(0, n_crit):
            z             = diff[:, :, k].ravel()
            z_sorted      = np.sort(z)
            n             = z_sorted.size
            ranks         = np.arange(1, n + 1, dtype = float) / n
            z_query       = diff[:, :, k].ravel()
            idx           = np.searchsorted(z_sorted, z_query, side = "right")
            idx           = np.clip(idx, 1, n)
            Fz            = ranks[idx - 1]
            prob[:, :, k] = Fz.reshape((n_alt, n_prof))
        return prob
    else:
        raise ValueError("Unknown dist '{}'. Use 'normal', 'logit', 'beta-pert', or 'empirical'.".format(dist))

###############################################################################

# Function: CPP-TRI
def cpp_tri_method(decision_matrix, weights = None, profiles = None, num_cat = None, indep_criteria = True, rule = 'central', verbose = True, dist = "normal"):
    X                              = np.asarray(decision_matrix, dtype = float)
    num_alternatives, num_criteria = X.shape

    if rule == 'central':
        c = 1.0
    else:
        c = 0.5
    
    if weights is not None:
        w = np.asarray(weights, dtype = float)
        if w.shape[0] != num_criteria:
            raise ValueError("Length of weights must match number of criteria.")
        w = w / np.sum(w)
    else:
        w = None  

    class_profiles                    = profiles_to_quantiles(profiles, X, num_cat)
    class_profiles                    = np.asarray(class_profiles, dtype = float)
    num_profiles                      = class_profiles.shape[0]
    std_devs                          = np.std(X, axis = 0, ddof = 1)
    std_devs_safe                     = std_devs.copy()
    std_devs_safe[std_devs_safe == 0] = 1e-12   
    diff                              = (X[:, np.newaxis, :] - class_profiles[np.newaxis, :, :]) / (std_devs_safe * np.sqrt(2.0))
    prob_above_matrix                 = _compute_prob_above(diff, dist = dist)
    prob_below_matrix                 = 1.0 - prob_above_matrix
    
    if indep_criteria:
        if w is None:
            overall_prob_above = np.prod(prob_above_matrix, axis = 2)
            overall_prob_below = np.prod(prob_below_matrix, axis = 2)
        else:
            overall_prob_above = np.prod(prob_above_matrix ** w, axis = 2)
            overall_prob_below = np.prod(prob_below_matrix ** w, axis = 2)
    else:
        corr      = np.corrcoef(X, rowvar = False)  
        visited   = np.zeros(num_criteria, dtype = bool)
        groups    = []
        threshold = 0.7
        for i in range(num_criteria):
            if visited[i]:
                continue
            stack      = [i]
            comp       = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(num_criteria):
                    if (not visited[v]) and (v != u) and (corr[u, v] >= threshold):
                        visited[v] = True
                        stack.append(v)
            groups.append(comp)
        group_above_list = []
        group_below_list = []
        for comp in groups:
            comp_idx = np.array(comp, dtype = int)
            if w is None:
                group_above = prob_above_matrix[:, :, comp_idx].mean(axis = 2)
                group_below = prob_below_matrix[:, :, comp_idx].mean(axis = 2)
            else:
                w_group     = w[comp_idx]
                w_group     = w_group / np.sum(w_group)
                group_above = np.tensordot(prob_above_matrix[:, :, comp_idx], w_group, axes = (2, 0))
                group_below = np.tensordot(prob_below_matrix[:, :, comp_idx], w_group, axes = (2, 0))
            group_above_list.append(group_above)
            group_below_list.append(group_below)
        group_above_stack  = np.stack(group_above_list, axis = 2)
        group_below_stack  = np.stack(group_below_list, axis = 2)
        overall_prob_above = np.min(group_above_stack,  axis = 2)
        overall_prob_below = np.min(group_below_stack,  axis = 2)

    A_plus  = overall_prob_above
    A_minus = overall_prob_below

    if   rule in ('central', 'c'):
        delta = A_plus - A_minus
    elif rule in ('benevolent', 'b'):
        delta = A_plus - c * A_minus
    elif rule in ('exacting', 'strict', 's', 'e'):
        delta = c * A_plus - A_minus
    else:
        raise ValueError("rule must be 'central', 'benevolent', or 'strict'")
    
    classification = []
    for i in range(num_alternatives):
        abs_delta    = np.abs(delta[i, :])
        chosen_class = int(np.argmin(abs_delta)) + 1
        classification.append(chosen_class)
    
    if verbose:
        for idx, cls in enumerate(classification, start = 1):
            abs_delta_i = np.abs(delta[idx - 1, :])
            details     = ", ".join([f"c{j+1} = {abs_delta_i[j]:.2f}" for j in range(0, num_profiles)])
            print(f"a{idx} : {cls} ({details})")
    return classification

###############################################################################
