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

# Function: Borda
def borda_method(dataset, criterion_type, graph = True, verbose = True):
    X = np.zeros((dataset.shape[0], dataset.shape[1]))
    for j in range(0, dataset.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] = np.argsort(np.argsort(-dataset[:,j]))+1
        else:
            X[:,j] = np.argsort(np.argsort(dataset[:,j]))+1
    total = np.sum(X, axis = 1)
    if (verbose == True):
        for i in range(0, total.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(total[i], 2)))
    if ( graph == True):
        flow = np.copy(total)
        flow = np.reshape(flow, (total.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, total.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        ranking(flow)
    return total

###############################################################################