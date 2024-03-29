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

# Function: CRADIS (Compromise Ranking of Alternatives from Distance to Ideal Solution)
def cradis_method(dataset, criterion_type, weights, graph = True, verbose = True):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] = np.min(X[:,j]) / X[:,j]
        else:
            X[:,j] = X[:,j] / np.max(X[:,j])
    X   = X * weights
    Sp  = np.sum(np.max(X) - X, axis = 1) + 0.0000000000000001
    Sm  = np.sum(X - np.min(X), axis = 1)
    Sop = np.sum(np.max(X) - np.max(X, axis = 0))
    Som = np.sum(np.max(X, axis = 0) - np.min(X))
    Kp  = Sop / Sp
    Km  = Sm / Som
    Q   = (Kp + Km) / 2
    if (verbose == True):
        for i in range(0, Q.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(Q[i], 2)))
    if ( graph == True):
        flow = np.copy(Q)
        flow = np.reshape(flow, (Q.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, Q.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        ranking(flow)
    return Q

###############################################################################