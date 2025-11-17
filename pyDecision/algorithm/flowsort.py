###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Distance
def distance_matrix(dataset, criteria = 0):
    dataset        = np.asarray(dataset, dtype = float)
    n              = dataset.shape[0]
    distance_array = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance_array[i, j] = dataset[i, criteria] - dataset[j, criteria]
    return distance_array
 
###############################################################################

# Function: Preference Degrees
def preference_degree(dataset, W, Q, S, P, F):
    dataset  = np.asarray(dataset, dtype = float)
    n, m     = dataset.shape
    pd_array = np.zeros((n, n))

    for k in range(0, m):
        distance_array = distance_matrix(dataset, criteria=k)
        for i in range(0, n):
            for j in range(0, n):
                if i == j:
                    distance_array[i, j] = 0.0
                    continue
                d = distance_array[i, j]
                if   F[k] == 't1': # Usual criterion
                    distance_array[i, j] = 0.0 if d <= 0 else 1.0
                elif F[k] == 't2': # U-shape
                    if d <= Q[k]:
                        distance_array[i, j] = 0.0
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == 't3': # V-shape
                    if d <= 0:
                        distance_array[i, j] = 0.0
                    elif 0 < d <= P[k]:
                        distance_array[i, j] = d / P[k]
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == 't4': # Level
                    if d <= Q[k]:
                        distance_array[i, j] = 0.0
                    elif Q[k] < d <= P[k]:
                        distance_array[i, j] = 0.5
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == 't5': # V-shape with indifference
                    if d <= Q[k]:
                        distance_array[i, j] = 0.0
                    elif Q[k] < d <= P[k]:
                        distance_array[i, j] = (d - Q[k]) / (P[k] - Q[k])
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == 't6': # Gaussian
                    if d <= 0:
                        distance_array[i, j] = 0.0
                    else:
                        distance_array[i, j] = 1 - np.exp(-(d ** 2) / (2 * S[k] ** 2))
                elif F[k] == 't7': # "U-shape with linear preference zone"
                    if d == 0:
                        distance_array[i, j] = 0.0
                    elif 0 < d <= S[k]:
                        distance_array[i, j] = (d / S[k]) ** 0.5
                    else:
                        distance_array[i, j] = 1.0
        pd_array = pd_array + W[k] * distance_array
    pd_array = pd_array / sum(W)
    return pd_array

###############################################################################

# Function: Flows
def promethee_flows_raw(dataset, W, Q, S, P, F):
    pd_matrix  = preference_degree(dataset, W, Q, S, P, F)
    n          = pd_matrix.shape[0]
    flow_plus  = np.sum(pd_matrix, axis = 1) / (n - 1)
    flow_minus = np.sum(pd_matrix, axis = 0) / (n - 1)
    flow_net   = flow_plus - flow_minus
    return pd_matrix, flow_plus, flow_minus, flow_net

###############################################################################

# Function: Limiting
def flowsort_limiting(dataset, profiles, W, Q, S, P, F):
    dataset  = np.asarray(dataset,  dtype = float)
    profiles = np.asarray(profiles, dtype = float)
    n_alt, m = dataset.shape
    K1       = profiles.shape[0]    
    K        = K1 - 1               
    cat_pos  = np.zeros(n_alt, dtype = int)
    cat_neg  = np.zeros(n_alt, dtype = int)
    cat_net  = np.zeros(n_alt, dtype = int)

    for i, a in enumerate(dataset):
        R_i           = np.vstack([profiles, a])
        _, fp, fm, fn = promethee_flows_raw(R_i, W, Q, S, P, F)
        fp_r, fp_a    = fp[:K1], fp[-1]   
        fm_r, fm_a    = fm[:K1], fm[-1]  
        fn_r, fn_a    = fn[:K1], fn[-1]  
        
        h_pos = None
        for h in range(0, K):
            if fp_r[h] >= fp_a > fp_r[h + 1]:
                h_pos = h + 1
                break
        if h_pos is None:
            if fp_a > fp_r[0]:
                h_pos = 1
            elif fp_a <= fp_r[-1]:
                h_pos = K
            else:
                for h in range(0, K):
                    if fp_r[h] >= fp_a >= fp_r[h + 1]:
                        h_pos = h + 1
                        break
                if h_pos is None:
                    h_pos = 1
        h_neg = None
        for h in range(0, K):
            if fm_r[h] < fm_a <= fm_r[h + 1]:
                h_neg = h + 1
                break
        if h_neg is None:
            if fm_a <= fm_r[0]:
                h_neg = 1
            elif fm_a > fm_r[-1]:
                h_neg = K
            else:
                for h in range(0, K):
                    if fm_r[h] <= fm_a <= fm_r[h + 1]:
                        h_neg = h + 1
                        break
                if h_neg is None:
                    h_neg = 1

        h_net = None
        for h in range(0, K):
            if fn_r[h] >= fn_a > fn_r[h + 1]:
                h_net = h + 1
                break
        if h_net is None:
            if fn_a > fn_r[0]:
                h_net = 1
            elif fn_a <= fn_r[-1]:
                h_net = K
            else:
                for h in range(0, K):
                    if fn_r[h] >= fn_a >= fn_r[h + 1]:
                        h_net = h + 1
                        break
                if h_net is None:
                    h_net = 1
        cat_pos[i] = h_pos
        cat_neg[i] = h_neg
        cat_net[i] = h_net
    return cat_pos, cat_neg, cat_net

# Function: Central
def flowsort_central(dataset, profiles, W, Q, S, P, F):
    dataset  = np.asarray(dataset,  dtype = float)
    profiles = np.asarray(profiles, dtype = float)
    n_alt, m = dataset.shape
    K        = profiles.shape[0]
    cat_pos  = np.zeros(n_alt, dtype = int)
    cat_neg  = np.zeros(n_alt, dtype = int)
    cat_net  = np.zeros(n_alt, dtype = int)
   
    for i, a in enumerate(dataset):
        R_i           = np.vstack([profiles, a])
        _, fp, fm, fn = promethee_flows_raw(R_i, W, Q, S, P, F)
        fp_r, fp_a    = fp[:K], fp[-1]   
        fm_r, fm_a    = fm[:K], fm[-1]   
        fn_r, fn_a    = fn[:K], fn[-1]   
        h_pos         = int(np.argmin(np.abs(fp_r - fp_a))) + 1
        h_neg         = int(np.argmin(np.abs(fm_r - fm_a))) + 1
        h_net         = int(np.argmin(np.abs(fn_r - fn_a))) + 1
        cat_pos[i]    = h_pos
        cat_neg[i]    = h_neg
        cat_net[i]    = h_net
    return cat_pos, cat_neg, cat_net

###############################################################################

# Function: Flowsort
def flowsort_method(dataset, profiles, W, Q, S, P, F, mode = "limiting", rule = "net", verbose = True):
    mode = mode.lower()
    rule = rule.lower()

    if mode == "limiting":
        cat_pos, cat_neg, cat_net = flowsort_limiting(dataset, profiles, W, Q, S, P, F)
    else: 
        cat_pos, cat_neg, cat_net = flowsort_central(dataset, profiles, W, Q, S, P, F)
        
    if rule == "net":
        cat_idx = cat_net
    elif rule == "positive":
        cat_idx = cat_pos
    else: 
        cat_idx = cat_neg
    if verbose:
        for i in range(0, len(cat_idx)):
            print('a'+str(i+1)+': '+ 'C' + str(cat_idx[i]))
    return cat_idx

###############################################################################