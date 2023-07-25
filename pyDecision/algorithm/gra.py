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

# Function: GRA (Grey Relational Analysis)
def gra_method(dataset, criterion_type, weights, epsilon = 0.5, graph = True, verbose = True):
    x = np.zeros((dataset.shape[0], dataset.shape[1]), dtype = float)
    for j in range(0, dataset.shape[1]):
        if (criterion_type[j] == 'max'):
            x[:,j] = ( dataset[:,j] - np.min(dataset[:,j]) ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) + 0.0000000000000001)
        else:
            x[:,j] = ( np.max(dataset[:,j]) - dataset[:,j] ) / ( np.max(dataset[:,j]) - np.min(dataset[:,j]) + 0.0000000000000001)
    deviation_sequence = 1 - x
    gra_coefficient    = epsilon/(deviation_sequence + epsilon)
    gra_grade          = np.sum(gra_coefficient*weights, axis = 1) / dataset.shape[0]
    if (verbose == True):
        for i in range(0, gra_grade.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(gra_grade[i], 4)))
    if ( graph == True):
        flow = np.copy(gra_grade)
        flow = np.reshape(flow, (gra_grade.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, gra_grade.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return gra_grade 

###############################################################################