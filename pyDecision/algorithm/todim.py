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

# Function: TODIM (TOmada de Decisao Interativa e Multicriterio - Interactive and Multicriteria Decision Making)
def todim_method(dataset, criterion_type, weights, teta = 1, graph = True, verbose = True):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] = X[:,j]  / np.sum(X[:,j]) 
        else:
            X[:,j] = (1/X[:,j])  / np.sum(1/X[:,j])       
    weights = weights/np.max(weights)
    D       = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, D.shape[0]):
        for j in range(0, D.shape[1]):
            if (i != j):
                for k in range(0, X.shape[1]):
                    p_i = X[i, k]
                    p_j = X[j, k]
                    if (p_i - p_j > 0):
                        D[i, j] = D[i, j] + (weights[k]*(p_i - p_j) / np.sum(weights))**(1/2)
                    elif (p_i - p_j == 0):
                        D[i, j] = D[i, j] + 0
                    else:
                        D[i, j] = D[i, j] + (-1/teta)*(np.sum(weights)*(p_j - p_i)/weights[k])**(1/2)
    r  = np.sum(D, axis = 1) 
    r  = (r - np.min(r)) / (np.max(r) - np.min(r))
    if (verbose == True):
        for i in range(0, r.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(r[i], 2)))
    if ( graph == True):
        flow = np.copy(r)
        flow = np.reshape(flow, (r.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, r.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return r

###############################################################################