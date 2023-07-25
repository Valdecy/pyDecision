###############################################################################

# Required Libraries
import numpy as np

from pyDecision.util.gwo import grey_wolf_optimizer

###############################################################################

# Function: BWM
def bw_method(dataset, mic, lic, size = 50, iterations = 150, verbose = True):
    X         = np.copy(dataset)/1.0
    best      = np.where(mic == 1)[0][0]
    worst     = np.where(lic == 1)[0][0]
    pairs_b   = [(best, i)  for i in range(0, mic.shape[0])]
    pairs_w   = [(i, worst) for i in range(0, mic.shape[0]) if (i, worst) not in pairs_b]
    def target_function(variables):
        eps       = [float('+inf')]
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
    weights = grey_wolf_optimizer(pack_size = size, min_values = [0.01]*X.shape[1], max_values = [1]*X.shape[1], iterations = iterations, target_function = target_function, verbose = verbose)
    weights = weights[0][:-1]/sum(weights[0][:-1])
    return weights

###############################################################################