###############################################################################

# Required Libraries
import numpy as np

from pyDecision.util.ga import genetic_algorithm

###############################################################################

# Function: IDOCRIW (Integrated Determination of Objective CRIteria Weights)
def idocriw_method(dataset, criterion_type, size = 20, gen = 12000, verbose = False):
    X    = np.copy(dataset)
    X    = X/X.sum(axis = 0)
    X_ln = np.copy(dataset)
    X_r  = np.copy(dataset)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_ln[i,j] = X[i,j]*np.log(X[i,j])
    d = np.zeros((1, X.shape[1]))
    w = np.zeros((1, X.shape[1]))
    for i in range(0, d.shape[1]):
        d[0,i] = 1-( -1/(np.log(d.shape[1]))*sum(X_ln[:,i])) 
    for i in range(0, w.shape[1]):
        w[0,i] = d[0,i]/d.sum(axis = 1)
    for i in range(0, len(criterion_type)):
        if (criterion_type[i] == 'min'):
           X_r[:,i] = dataset[:,i].min() / X_r[:,i]
    X_r   = X_r/X_r.sum(axis = 0)   
    a_max = X_r.max(axis = 0) 
    A     = np.zeros(dataset.shape)
    np.fill_diagonal(A, a_max)
    for k in range(0, a_max.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i    = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]     
    a_max_ = A.max(axis = 0) 
    P      = np.copy(A)    
    for i in range(0, P.shape[1]):
        P[:,i] = (-P[:,i] + a_max_[i])/a_max[i]
    WP     = np.copy(P)
    np.fill_diagonal(WP, -P.sum(axis = 0))
    
    ################################################
    def target_function(variable = [0]*WP.shape[1]):
        variable = [variable[i]/sum(variable) for i in range(0, len(variable))]
        WP_s     = np.copy(WP)
        for i in range(0, WP.shape[0]):
            for j in range(0, WP.shape[1]):
                WP_s[i, j] = WP_s[i, j]*variable[j]
        total = abs(WP_s.sum(axis = 1)) 
        total = sum(total) 
        return total
    ################################################
    
    solution = genetic_algorithm(population_size = size, mutation_rate = 0.1, elite = 1, min_values = [0]*WP.shape[1], max_values = [1]*WP.shape[1], eta = 1, mu = 1, generations = gen, target_function = target_function, verbose = verbose)
    solution = solution[:-1]
    solution = solution/sum(solution)
    w_       = np.copy(w)
    w_       = w_*solution
    w_       = w_/w_.sum()
    w_       = w_.T
    return w_

###############################################################################
