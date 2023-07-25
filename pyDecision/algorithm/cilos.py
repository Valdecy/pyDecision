###############################################################################

# Required Libraries
import numpy as np

from scipy.linalg import null_space

###############################################################################

# Function: CILOS (Criterion Impact LOSs)
def cilos_method(dataset, criterion_type):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] = np.min(X[:,j]) / X[:,j]
        else:
            X[:,j] = X[:,j] / np.max(X[:,j])
        X[:,j] = X[:,j] / np.sum(X[:,j])
    A = X[np.argmax(X, axis = 0)]
    P = (np.diag(A) - A) / np.diag(A)
    F = P - np.diag(np.sum(P, axis = 0))
    q = null_space(F)
    q = (q / np.sum(q)).flatten()
    return q

###############################################################################