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

# Function: CODAS (Combinative Distance-based Assessment)
def codas_method(dataset, weights, criterion_type, lmbd = 0.02, graph = True, verbose = True):
    X   = np.copy(dataset)/1.0
    bst = np.zeros(X.shape[1])
    r_m = np.zeros((X.shape[0], X.shape[0]))
    for j in range(0, X.shape[1]):
        if ( criterion_type[j] == 'max'):
            bst[j] = np.max(X[:, j])
        elif ( criterion_type[j] == 'min'):
            bst[j] = np.min(X[:, j])
    for j in range(0, X.shape[1]):
        for i in range(0, X.shape[0]):
            if ( criterion_type[j] == 'max'):
                X[i, j] = X[i, j]/bst[j]
            elif ( criterion_type[j] == 'min'):
                X[i, j] = bst[j]/X[i, j]
    X   = X * weights
    n_i = np.min(X, axis = 0)
    e_i = np.sum( (X - n_i)**2, axis = 1)**(1/2)
    t_i = np.sum( abs(X - n_i), axis = 1)
    for j in range(0, r_m.shape[1]):
        for i in range(0, r_m.shape[0]):
            r_m[i, j] = (e_i[i] - e_i[j]) + lmbd*( (e_i[i] - e_i[j]) *  (t_i[i] - t_i[j]) )
    h_i  = np.sum(r_m, axis = 1)
    flow = np.copy(h_i)
    flow = np.reshape(flow, (h_i.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, h_i.shape[0]+1)), axis = 1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow