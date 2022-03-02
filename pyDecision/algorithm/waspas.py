###############################################################################

# Required Libraries

import numpy as np

###############################################################################

# Function: WASPAS
def waspas_method(dataset, criterion_type, weights, lambda_value):
    x = np.zeros((dataset.shape[0], dataset.shape[1]), dtype = float)
    for j in range(0, dataset.shape[1]):
        if (criterion_type[j] == 'max'):
            x[:,j] = 1 + ( dataset[:,j] - np.min(dataset[:,j]) ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) )
        else:
            x[:,j] = 1 + ( np.max(dataset[:,j]) - dataset[:,j] ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) )
    wsm    = np.sum(x*weights, axis = 1)
    wpm    = np.prod(x**weights, axis = 1)
    waspas = (lambda_value)*wsm + (1 - lambda_value)*wpm
    return wsm, wpm, waspas

###############################################################################

