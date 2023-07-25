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

# Function: COCOSO (COmbined COmpromise SOlution)
def cocoso_method(dataset, criterion_type, weights, L = 0.5, graph = True, verbose = True):
    X     = np.copy(dataset)/1.0
    best  = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])
    for i in range(0, dataset.shape[1]):
        if (criterion_type[i] == 'max'):
            best[i]  = np.max(X[:, i])
            worst[i] = np.min(X[:, i])
        else:
            best[i]  = np.min(X[:, i])
            worst[i] = np.max(X[:, i])        
    for j in range(0, X.shape[1]):
        X[:,j] = ( X[:,j] - worst[j] ) / ( best[j] - worst[j] + 0.0000000000000001) 
    S      = np.sum(X  * weights, axis = 1)
    P      = np.sum(X ** weights, axis = 1)
    if (np.min(S) == 0):
        S = S + 1
    if (np.min(P) == 0):
        P = P + 1
    ksi_a  = (P + S) / np.sum(P + S, axis = 0)
    ksi_b  = (S / np.min(S)) + (P / np.min(P))
    ksi_c  = (L * S + (1 - L) * P) / (L * np.max(S) + (1 - L) * np.max(P))
    ksi    = np.power(ksi_a * ksi_b * ksi_c, 1/3) + 1/3 * (ksi_a + ksi_b + ksi_c)
    if (verbose == True):
        for i in range(0, ksi.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(ksi[i], 2)))
    if ( graph == True):
        flow = np.copy(ksi)
        flow = np.reshape(flow, (ksi.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, ksi.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return ksi

###############################################################################