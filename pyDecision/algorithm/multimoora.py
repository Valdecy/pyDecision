###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank 
def ranking_m(flow_1, flow_2, flow_3):    
    rank_xy = np.zeros((flow_1.shape[0] + 1, 6)) 
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = -1
        rank_xy[i, 1] = flow_1.shape[0]-i+1     
        rank_xy[i, 2] = 0
        rank_xy[i, 3] = flow_2.shape[0]-i+1  
        rank_xy[i, 4] = 1
        rank_xy[i, 5] = flow_3.shape[0]-i+1    
    plt.text(rank_xy[0, 0],  rank_xy[0, 1], 'MOORA', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 2],  rank_xy[0, 3], 'MOORA RP', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 4],  rank_xy[0, 5], 'MULTIMOORA', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    for i in range(1, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow_1[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = "round", ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
        plt.text(rank_xy[i, 2],  rank_xy[i, 3], 'a' + str(int(flow_2[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
        plt.text(rank_xy[i, 4],  rank_xy[i, 5], 'a' + str(int(flow_3[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),)) 
    for i in range(1, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        plt.arrow(rank_xy[i, 2], rank_xy[i, 3], rank_xy[i+1, 2] - rank_xy[i, 2], rank_xy[i+1, 3] - rank_xy[i, 3], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        plt.arrow(rank_xy[i, 4], rank_xy[i, 5], rank_xy[i+1, 4] - rank_xy[i, 4], rank_xy[i+1, 5] - rank_xy[i, 5], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-2, +2])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: MULTIMOORA (Multi-objective Optimization on the basis of Ratio Analisys Multiplicative Form)
def multimoora_method(dataset, criterion_type, graph = True):
    X    = np.copy(dataset)/1.0
    root = (np.sum(X**2, axis = 0))**(1/2)
    X    = X/root
    best = np.zeros(X.shape[1])
    Y1   = np.zeros(X.shape[0]) # MOORA
    Y2   = np.zeros(X.shape[0]) # MOORA Reference Point
    Y3   = np.zeros(X.shape[0]) # MULTIMOORA
    id1  = [i for i, j in enumerate(criterion_type) if j == 'max']
    id2  = [i for i, j in enumerate(criterion_type) if j == 'min']
    s_p  = np.zeros(X.shape[0])
    s_m  = np.zeros(X.shape[0])
    if (len(id1) > 0):
        s_p = np.sum(X[:,id1], axis = 1)
    if (len(id2) > 0):
        s_m = np.sum(X[:,id2], axis = 1)
    Y1 = s_p - s_m
    for i in range(0, X.shape[1]):
        if ( criterion_type[i] == 'max'):
            best[i] = np.max(X[:,i])
        else:
            best[i] = np.min(X[:,i])
    Y2 = np.max(np.absolute(X - best), axis = 1)
    if ( criterion_type[0] == 'max'):
        Y3 = np.copy(dataset[:,0])
    else:
        Y3 = 1/np.copy(dataset[:,0])
    for i in range(0, dataset.shape[0]):
        for j in range(1, dataset.shape[1]):
            if ( criterion_type[j] == 'max'):
                Y3[i] = Y3[i]*dataset[i,j] 
            else:
                Y3[i] = Y3[i]/dataset[i,j] 
    Y1     = Y1/np.max(Y1)
    Y2     = Y2/np.max(Y2)
    Y3     = Y3/np.max(Y3)
    flow_1 = np.copy(Y1)
    flow_1 = np.reshape(flow_1, (Y1.shape[0], 1))
    flow_1 = np.insert(flow_1, 0, list(range(1, Y1.shape[0]+1)), axis = 1)
    flow_2 = np.copy(Y2)
    flow_2 = np.reshape(flow_2, (Y2.shape[0], 1))
    flow_2 = np.insert(flow_2, 0, list(range(1, Y2.shape[0]+1)), axis = 1)
    flow_3 = np.copy(Y3)
    flow_3 = np.reshape(flow_3, (Y3.shape[0], 1))
    flow_3 = np.insert(flow_3, 0, list(range(1, Y3.shape[0]+1)), axis = 1)
    if (graph == True):
        ranking_m(flow_1[np.argsort(flow_1[:, 1])[::-1]], flow_2[np.argsort(flow_2[:, 1])], flow_3[np.argsort(flow_3[:, 1])[::-1]])
    return flow_1, flow_2, flow_3