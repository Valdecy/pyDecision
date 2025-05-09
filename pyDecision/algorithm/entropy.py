###############################################################################

# Required Libraries
import itertools
import numpy as np

###############################################################################

# Function: Entropy
def entropy_method(dataset, criterion_type):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] =  X[:,j] / np.sum(X[:,j])
        else:
            X[:,j] = (1 / (X[:,j] + 1e-9)) / np.sum(1 / (X[:,j] + 1e-9))
    X = np.abs(X)
    H = np.zeros((X.shape))
    for j, i in itertools.product(range(H.shape[1]), range(H.shape[0])):
        if (X[i, j]):
            H[i, j] = X[i, j] * np.log(X[i, j] + 1e-9)
    h = np.sum(H, axis = 0) * (-1 * ((np.log(H.shape[0] + 1e-9)) ** (-1)))
    d = 1 - h
    d = d + 1e-9
    w = d / (np.sum(d))
    return w

###############################################################################
