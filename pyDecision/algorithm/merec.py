###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: MEREC (MEthod based on the Removal Effects of Criteria)
def merec_method(dataset, criterion_type):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            divisor = np.min(X[:,j])
            if (divisor == 0):
                divisor =  1e-9
            X[:,j] = divisor / (X[:,j] + 1e-9)
        else:
            divisor = np.max(X[:,j])
            if (divisor == 0):
                divisor =  1e-9
            X[:,j] = X[:,j] / (divisor + 1e-9)
    S = np.log(1 + (1/X.shape[1] * np.sum(np.abs(np.log(X)), axis = 1)))
    R = np.zeros(X.shape)
    for j in range(0, X.shape[1]):
        Z       = np.delete(X, j, axis = 1)
        R[:, j] = np.log(1 + (1/X.shape[1] * np.sum(np.abs(np.log(Z)), axis = 1)))
    E = np.sum(np.abs(R.T - S), axis = 1)
    E = E / np.sum(E)
    return E

###############################################################################
