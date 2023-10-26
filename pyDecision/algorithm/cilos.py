###############################################################################

# Required Libraries
import numpy as np

from scipy.linalg import null_space

###############################################################################

# Function: CILOS (Criterion Impact LOSs)
def cilos_method(dataset, criterion_type):
    X = np.copy(dataset)/1.0
    for j in range(X.shape[1]):
        divisor = np.min(X[:, j]) if criterion_type[j] == 'max' else np.max(X[:, j])
        if (divisor == 0):  
            X[:, j] = 1e-9
        else:
            X[:, j] = X[:, j] / divisor
        sum_X_j = np.sum(X[:, j])
        if (sum_X_j == 0):  
            X[:, j] = 1e-9
        else:
            X[:, j] = X[:, j] / sum_X_j
    for j in range(X.shape[1]):
        unique_vals = np.unique(X[:, j])
        if (len(unique_vals) == 1):
            X[:, j] = X[:, j] + np.random.uniform(low=1e-2, high=1e-1, size=X[:, j].shape)
    A = X[np.argmax(X, axis = 0)]
    P = (np.diag(A) - A) / np.diag(A)
    F = P - np.diag(np.sum(P, axis = 0))
    q = null_space(F)
    q = (q / np.sum(q)).flatten()
    return q

###############################################################################
