###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: CRITIC (CRiteria Importance Through Intercriteria Correlation)
def critic_method(dataset, criterion_type):
    X     = np.copy(dataset)/1.0
    best  = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])
    for i in range(0, dataset.shape[1]):
        if (criterion_type[i] == 'max'):
            best[i]  = np.max(X[:, i])
            worst[i] = np.min(X[:, i])
        else:
            best[i]  = np.min(X[:, i])
            worst[i] = np.max(X[:, i])        
    for j in range(0, X.shape[1]):
        X[:,j] = ( X[:,j] - worst[j] ) / ( best[j] - worst[j] + 1e-10) 
    std      = (np.sum((X - X.mean())**2, axis = 0)/(X.shape[0] - 1))**(1/2)
    sim_mat  = np.corrcoef(X.T)
    sim_mat  = np.nan_to_num(sim_mat) 
    conflict = np.sum(1 - sim_mat, axis = 1)
    infor    = std*conflict
    weights  = infor/np.sum(infor)
    return weights

###############################################################################
