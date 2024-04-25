###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: MPSI (Modified Preference Selection Index)
def mpsi_method(dataset, criterion_type):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] = X[:,j] / np.max(X[:,j])
        else:
            X[:,j] = np.min(X[:,j]) / X[:,j]
    R = np.mean(X, axis = 0)
    Z = np.sum((X - R)**2, axis = 0)
    I = Z/np.sum(Z)
    return I

###############################################################################