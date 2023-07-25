###############################################################################

# Required Libraries

import numpy as np
import matplotlib.pyplot as plt

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

# Function: SMART
def smart_method(dataset, grades, lower, upper, criterion_type, graph = True, verbose = True):
    w = np.copy(grades)/1.0
    for i in range(0, w.shape[0]):
        w[i] = (2**(1/2))**w[i]
    w = w/np.sum(w)
    X = np.copy(dataset)/1.0
    for i in range(0, X.shape[1]):
        if ( criterion_type[i] == 'max'):
            X[:,i] =  4 + np.log2( ((dataset[:,i] - lower[i] ) / (upper[i] - lower[i]))*(64) ) 
        else:
            X[:,i] = 10 - np.log2( ((dataset[:,i] - lower[i] ) / (upper[i] - lower[i]))*(64) ) 
    Y    = np.sum(X*w, axis = 1)
    flow = np.copy(Y)
    flow = np.reshape(flow, (Y.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, Y.shape[0]+1)), axis = 1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow

###############################################################################