###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Fuzzy CRITIC (Fuzzy CRiteria Importance Through Intercriteria Correlation)
def fuzzy_critic_method(dataset, criterion_type):
    X_a   = np.array([[triplet[0] for triplet in row] for row in dataset])
    X_b   = np.array([[triplet[1] for triplet in row] for row in dataset])
    X_c   = np.array([[triplet[2] for triplet in row] for row in dataset])
    min_a = X_a.min(axis = 0)
    min_b = X_b.min(axis = 0)
    min_c = X_c.min(axis = 0)
    max_a = X_a.max(axis = 0)
    max_b = X_b.max(axis = 0)
    max_c = X_c.max(axis = 0)
    R     = np.zeros((X_a.shape))
    for j in range(0, X_a.shape[1]):
        for i in range(0, X_a.shape[0]):
            if (criterion_type[j] == 'max'):
                p1      = (X_a[i, j]**2)*(min_a[j]**2) + (X_b[i, j]**2)*(min_b[j]**2) + (X_c[i, j]**2)*(min_c[j]**2)
                p2      = np.max((X_a[i, j]**4, min_a[j]**4)) + np.max((X_b[i, j]**4, min_b[j]**4)) + np.max((X_c[i, j]**4, min_c[j]**4))
                p       = p1/p2
                q1      = (max_a[j]**2)*(min_a[j]**2) + (max_b[j]**2)*(min_b[j]**2) + (max_c[j]**2)*(min_c[j]**2)
                q2      = np.max((max_a[j]**4, min_a[j]**4)) + np.max((max_b[j]**4, min_b[j]**4)) + np.max((max_c[j]**4, min_c[j]**4))
                q       = q1/q2
                R[i, j] = (1 - p)/(1 - q)
            else:
                p1      = (X_a[i, j]**2)*(max_a[j]**2) + (X_b[i, j]**2)*(max_b[j]**2) + (X_c[i, j]**2)*(max_c[j]**2)
                p2      = np.max((X_a[i, j]**4, max_a[j]**4)) + np.max((X_b[i, j]**4, max_b[j]**4)) + np.max((X_c[i, j]**4, max_c[j]**4))
                p       = p1/p2
                q1      = (max_a[j]**2)*(min_a[j]**2) + (max_b[j]**2)*(min_b[j]**2) + (max_c[j]**2)*(min_c[j]**2)
                q2      = np.max((max_a[j]**4, min_a[j]**4)) + np.max((max_b[j]**4, min_b[j]**4)) + np.max((max_c[j]**4, min_c[j]**4))
                q       = q1/q2
                R[i, j] = (1 - p)/(1 - q)
    std      = (np.sum((R - R.mean())**2, axis = 0)/(R.shape[0] - 1))**(1/2)
    sim_mat  = np.corrcoef(R.T)
    conflict = np.sum(1 - sim_mat, axis = 1)
    infor    = std*conflict
    weights  = infor/np.sum(infor)
    return weights

###############################################################################