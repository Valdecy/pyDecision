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

# Function: FARAS (Fuzzy Additive Ratio Assessment)
def fuzzy_aras_method(dataset, weights, criterion_type, graph = True, verbose = True):
    X_a        = np.zeros((len(dataset), len(dataset[0])))
    X_b        = np.zeros((len(dataset), len(dataset[0])))
    X_c        = np.zeros((len(dataset), len(dataset[0])))
    weights_a  = np.zeros(len(weights[0]))
    weights_b  = np.zeros(len(weights[0]))
    weights_c  = np.zeros(len(weights[0]))
    for j in range(0, X_a.shape[1]):
        weights_a[j] = weights[0][j][0]
        weights_b[j] = weights[0][j][1]
        weights_c[j] = weights[0][j][2]
        for i in range(0, X_a.shape[0]):
            a, b, c  = dataset[i][j]
            X_a[i,j] = a
            X_b[i,j] = b
            X_c[i,j] = c
    for j in range(0, X_a.shape[1]):
        if (criterion_type[j] == 'max'):
            X_a[:, j]  = X_a[:, j] / np.sum(X_a[:, j])
            X_b[:, j]  = X_b[:, j] / np.sum(X_b[:, j])
            X_c[:, j]  = X_c[:, j] / np.sum(X_c[:, j])
        else:
            X_a[:, j]  = (1 / X_a[:, j]) / np.sum((1 / X_a[:, j]))
            X_b[:, j]  = (1 / X_b[:, j]) / np.sum((1 / X_b[:, j]))
            X_c[:, j]  = (1 / X_c[:, j]) / np.sum((1 / X_c[:, j]))
    for j in range(0, X_a.shape[1]):
        X_a[:, j]  = X_a[:, j] * weights_a[j]
        X_b[:, j]  = X_b[:, j] * weights_b[j]
        X_c[:, j]  = X_c[:, j] * weights_c[j]
    S_a = np.sum(X_a, axis = 1)
    S_b = np.sum(X_b, axis = 1)
    S_c = np.sum(X_c, axis = 1)
    S   = (1/3)*(S_a + S_b + S_c)
    Q   = S/np.max(S)
    if (verbose == True):
        for i in range(0, Q.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(Q[i], 2)))
    if ( graph == True):
        flow = np.copy(Q)
        flow = np.reshape(flow, (Q.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, Q.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return Q

###############################################################################