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

# Function: WISP (Integrated Simple Weighted Sum Product)
def wisp_method(dataset, criterion_type, weights, simplified = False, graph = True, verbose = True):
    weights = [item/sum(weights) for item in weights]
    X       = np.copy(dataset)/1.0
    best    = np.max(X, axis = 0)      
    X       = X  / ( best + 0.0000000000000001) 
    X       = X * weights
    v_p     = np.zeros((X.shape[0])) 
    v_m     = np.zeros((X.shape[0])) 
    w_p     = np.ones( (X.shape[0])) 
    w_m     = np.ones( (X.shape[0])) 
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            if (criterion_type[j] == 'max'):
                v_p[i] = v_p[i] + X[i, j]
                w_p[i] = w_p[i] * X[i, j]
            else:
                v_m[i] = v_m[i] + X[i, j]
                w_m[i] = w_m[i] * X[i, j]
    u_wsd = v_p - v_m
    u_wpd = w_p - w_m 
    if ('min' not in criterion_type):
        u_wsr = v_p
        u_wpr = w_p
    elif ('max' not in criterion_type):
        u_wsr = 1/v_m
        u_wpr = 1/w_m
    else:
        u_wsr = v_p / (v_m ) 
        u_wpr = w_p / (w_m ) 
    n_wsd = (1 + u_wsd) / (1 + np.max(u_wsd))
    n_wpd = (1 + u_wpd) / (1 + np.max(u_wpd)) 
    n_wsr = (1 + u_wsr) / (1 + np.max(u_wsr)) 
    n_wpr = (1 + u_wpr) / (1 + np.max(u_wpr)) 
    if (simplified == False):
        u = (n_wsd + n_wpd + n_wsr + n_wpr) / 4
    else:
        u = (n_wsd + n_wpr) / 2
    if (verbose == True):
        for i in range(0, u.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(u[i], 2)))
    if ( graph == True):
        flow = np.copy(u)
        flow = np.reshape(flow, (u.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, u.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return u

###############################################################################