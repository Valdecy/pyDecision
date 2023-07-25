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

# Function: COPRAS (Complex Proportional Assessment)
def copras_method(dataset, weights, criterion_type, graph = True, verbose = True):
    X   = np.copy(dataset)/1.0
    X   = X/np.sum(X, axis = 0)
    X   = X*weights
    s_p = np.zeros(X.shape[0])
    s_m = np.zeros(X.shape[0])
    s_d = np.zeros(X.shape[0])
    q_i = np.zeros(X.shape[0])
    u_i = np.zeros(X.shape[0])
    id1 = [i for i, j in enumerate(criterion_type) if j == 'max']
    id2 = [i for i, j in enumerate(criterion_type) if j == 'min']
    if (len(id1) > 0):
        s_p = np.sum(X[:,id1], axis = 1)
        q_i = s_p
    if (len(id2) > 0):
        s_m = np.sum(X[:,id2], axis = 1)
        s_d = np.min(s_m)/s_m
        q_i = s_p + (np.min(s_m)*np.sum(s_m))/(s_m*np.sum(s_d))
    u_i = q_i/np.max(q_i)
    flow = np.copy(u_i)
    flow = np.reshape(flow, (u_i.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, u_i.shape[0]+1)), axis = 1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow