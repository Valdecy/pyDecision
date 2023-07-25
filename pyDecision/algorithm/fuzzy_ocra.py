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

# Function: Fuzzy OCRA (Operational Competitiveness RAting)
def fuzzy_ocra_method(dataset, weights, criterion_type, graph = True, verbose = True):
    X_a        = np.zeros((len(dataset), len(dataset[0])))
    X_b        = np.zeros((len(dataset), len(dataset[0])))
    X_c        = np.zeros((len(dataset), len(dataset[0])))
    I_a        = np.zeros((len(dataset)))
    I_b        = np.zeros((len(dataset)))
    I_c        = np.zeros((len(dataset)))
    O_a        = np.zeros((len(dataset)))
    O_b        = np.zeros((len(dataset)))
    O_c        = np.zeros((len(dataset)))
    P_a        = np.zeros((len(dataset)))
    P_b        = np.zeros((len(dataset)))
    P_c        = np.zeros((len(dataset)))
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
    for i in range(0, X_a.shape[0]):
        for j in range(0, X_a.shape[1]):
            if (criterion_type[j] == 'max'):
                O_a[i] = O_a[i] + weights_a[j] * (X_a[i, j] -  np.min(X_a[:, j])) / np.min(X_a[:, j])
                O_b[i] = O_b[i] + weights_b[j] * (X_b[i, j] -  np.min(X_b[:, j])) / np.min(X_b[:, j])
                O_c[i] = O_c[i] + weights_c[j] * (X_c[i, j] -  np.min(X_c[:, j])) / np.min(X_c[:, j])
            else:
                I_a[i] = I_a[i] + weights_a[j] * (np.max(X_a[:, j]) - X_a[i, j]) / np.min(X_a[:, j])
                I_b[i] = I_b[i] + weights_b[j] * (np.max(X_b[:, j]) - X_b[i, j]) / np.min(X_b[:, j])
                I_c[i] = I_c[i] + weights_c[j] * (np.max(X_c[:, j]) - X_c[i, j]) / np.min(X_c[:, j])
    min_i_a = np.min(I_a)
    min_i_b = np.min(I_b)
    min_i_c = np.min(I_c)
    min_o_a = np.min(O_a)
    min_o_b = np.min(O_b)
    min_o_c = np.min(O_c)
    I_a     = I_a - min_i_a
    I_b     = I_b - min_i_b
    I_c     = I_c - min_i_c
    O_a     = O_a - min_o_a
    O_b     = O_b - min_o_b
    O_c     = O_c - min_o_c
    P_a     = I_a + O_a - np.min(I_a + O_a)
    P_b     = I_b + O_b - np.min(I_b + O_b)
    P_c     = I_c + O_c - np.min(I_c + O_c)
    P       = (P_a + P_b + P_c) / 3
    if (verbose == True):
        for i in range(0, P.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(P[i], 3))) 
    if (graph == True):
        flow = np.copy(P)
        flow = np.reshape(flow, (P.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, P.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return P