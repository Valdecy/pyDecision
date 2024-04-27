###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Fuzzy MEREC (Fuzzy MEthod based on the Removal Effects of Criteria)
def fuzzy_merec_method(dataset, criterion_type):
    X_a = np.zeros((len(dataset), len(dataset[0])))
    X_b = np.zeros((len(dataset), len(dataset[0])))
    X_c = np.zeros((len(dataset), len(dataset[0])))
    X   = np.zeros((len(dataset), len(dataset[0])))
    S_  = np.zeros((len(dataset), len(dataset[0])))
    for j in range(0, X_a.shape[1]):
        for i in range(0, X_a.shape[0]):
            a, b, c  = dataset[i][j]
            X_a[i,j] = a
            X_b[i,j] = b
            X_c[i,j] = c
    for j in range(0, X_a.shape[1]):
        if (criterion_type[j] == 'max'):
            X_a[:, j]  = X_a[:, j] / np.max(X_c[:, j])
            X_b[:, j]  = X_b[:, j] / np.max(X_c[:, j])
            X_c[:, j]  = X_c[:, j] / np.max(X_c[:, j])
        else:
            X_a[:, j]  = np.min(X_a[:, j]) / X_a[:, j]
            X_b[:, j]  = np.min(X_a[:, j]) / X_b[:, j]
            X_c[:, j]  = np.min(X_a[:, j]) / X_c[:, j]
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X[i,j] = (X_a[i,j] + 4*X_b[i,j] + X_c[i,j])/6
    S = (1/X.shape[1]) * np.sum(np.log(1 - X), axis = 1)
    for j in range(0, X.shape[1]):
        idx      = [item for item in list(range(0, X.shape[1])) if item != j]
        S_[:, j] = (1/X.shape[1]) * np.sum(np.log(1 - X[:,idx]), axis = 1)
    D = np.sum(abs(S_ - S.reshape(-1,1)), axis = 0)
    D = D/np.sum(D)
    return D

###############################################################################