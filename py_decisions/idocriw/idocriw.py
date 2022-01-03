###############################################################################

# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

from py_decisions.idocriw.util import genetic_algorithm

###############################################################################

# Function: Rank 
def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: IDOCRIW
def idocriw_method(dataset, criterion_type, size = 20, gen = 12000, graph = True):
    X    = np.copy(dataset)
    X    = X/X.sum(axis = 0)
    X_ln = np.copy(dataset)
    X_r  = np.copy(dataset)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_ln[i,j] = X[i,j]*math.log(X[i,j])
    d    = np.zeros((1, X.shape[1]))
    w    = np.zeros((1, X.shape[1]))
    for i in range(0, d.shape[1]):
        d[0,i] = 1-( -1/(math.log(d.shape[1]))*sum(X_ln[:,i])) 
    for i in range(0, w.shape[1]):
        w[0,i] = d[0,i]/d.sum(axis = 1)
    for i in range(0, len(criterion_type)):
        if (criterion_type[i] == 'min'):
           X_r[:,i] = dataset[:,i].min() / X_r[:,i]
    X_r   = X_r/X_r.sum(axis = 0)
    #a_min = X_r.min(axis = 0)       
    a_max = X_r.max(axis = 0) 
    A     = np.zeros(dataset.shape)
    np.fill_diagonal(A, a_max)
    for k in range(0, A.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i    = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]
    #a_min_ = A.min(axis = 0)       
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
    
    solution = genetic_algorithm(population_size = size, mutation_rate = 0.1, elite = 1, min_values = [0]*WP.shape[1], max_values = [1]*WP.shape[1], eta = 1, mu = 1, generations = gen, target_function = target_function)
    solution = solution[:-1]
    solution = solution/sum(solution)
    w_       = np.copy(w)
    w_       = w_*solution
    w_       = w_/w_.sum()
    w_       = w_.T
    for i in range(0, w_.shape[0]):
        print('a' + str(i+1) + ': ' + str(round(w_[i][0], 4)))
    if ( graph == True):
        flow = np.copy(w_)
        flow = np.reshape(flow, (w_.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, w_.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return w_

###############################################################################
