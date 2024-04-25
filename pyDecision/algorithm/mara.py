###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

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

# Function: MARA (Magnitude of the Area for the Ranking of Alternatives)
def mara_method(dataset, weights, criterion_type, graph = True, verbose = True):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] = X[:,j] / np.max(X[:,j])
        else:
            X[:,j] = np.min(X[:,j]) / X[:,j]
    X   = X * weights    
    opt = np.max(X, axis = 0) 
    S_k = 0
    S_L = 0
    T_k = np.zeros(X.shape[0])
    T_L = np.zeros(X.shape[0])
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            S_k = S_k + opt[j]
            for i in range(0, X.shape[0]):
                T_k[i] = T_k[i] + X[i, j]
        else:
            S_L = S_L + opt[j]
            for i in range(0, X.shape[0]):
                T_L[i] = T_L[i] + X[i, j]
    f_opt = (S_L - S_k)/2 + S_k
    f_i   = (T_L - T_k)/2 + T_k
    M     = f_opt - f_i    
    if (verbose == True):
        for i in range(0, M.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(M[i], 2)))
    if ( graph == True):
        flow = np.copy(M)
        flow = np.reshape(flow, (M.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, M.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        ranking(flow)
    return M

###############################################################################