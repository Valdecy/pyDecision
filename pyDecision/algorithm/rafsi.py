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

# Function: RAFSI (Ranking of Alternatives through Functional mapping of criterion sub-intervals into a Single Interval)
def rafsi_method(dataset, weights, criterion_type, ideal = [], anti_ideal = [], n_i = 1, n_k = 6, graph = True, verbose = True):
    X     = np.copy(dataset)/1.0
    coef  = np.zeros((2, X.shape[1]))
    best  = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            if (len(ideal) == 0):
                best[j]  = np.max(X[:,j])*2
            else:    
                best[j]  = ideal[j]
            if (len(anti_ideal) == 0):
                worst[j]  = np.min(X[:,j])*0.5
            else:
                worst[j] = anti_ideal[j]
        else:
            if (len(anti_ideal) == 0):
                best[j]  = np.min(X[:,j])*0.5
            else:    
                best[j]  = anti_ideal[j]
            if (len(ideal) == 0):
                worst[j]  = np.max(X[:,j])*2
            else:
                worst[j] = ideal[j]
        coef[0, j] = (n_k - n_i)/(best[j] - worst[j])
        coef[1, j] = (best[j]*n_i - worst[j]*n_k)/(best[j] - worst[j])
    S = X * coef[0, :] + coef[1,:]
    A = np.mean((n_i, n_k))
    H = 2/(n_i**-1 + n_k**-1)
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            S[:,j] = S[:,j]/(2*A)
        else:
            S[:,j] = H/(2*S[:,j])
    V    = np.sum(S*weights, axis = 1)
    flow = np.copy(V)
    flow = np.reshape(flow, (V.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, V.shape[0]+1)), axis = 1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return V

###############################################################################