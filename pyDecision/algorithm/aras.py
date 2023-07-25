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

# Function: ARAS (Additive Ratio Assessment)
def aras_method(dataset, weights, criterion_type, graph = True, verbose = True):
    X   = np.copy(dataset)/1.0
    bst = np.zeros(X.shape[1])
    bsm = np.zeros(X.shape[1])
    for j in range(0, X.shape[1]):
        if ( criterion_type[j] == 'max'):
            bst[j] = np.max(X[:, j])
            bsm[j] = bst[j] + np.sum(X[:, j])
        elif ( criterion_type[j] == 'min'):
            bst[j]  = 1/np.min(X[:, j])
            X[:,j]  = 1/X[:, j]
            bsm[j]  = bst[j] + np.sum(X[:, j])
    for j in range(0, X.shape[1]):
        bst[j] = bst[j]/ bsm[j]
        for i in range(0, X.shape[0]):
            X[i, j] = X[i, j]/ bsm[j]
    X    = X * weights
    bst  = bst * weights
    n_0  = np.sum(bst)
    n_i  = np.sum(X, axis = 1)
    k_i  = n_i/n_0
    flow = np.copy(k_i)
    flow = np.reshape(flow, (k_i.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, k_i.shape[0]+1)), axis = 1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow

###############################################################################