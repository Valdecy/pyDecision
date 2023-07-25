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

# Function: Fuzzy EDAS
def fuzzy_edas_method(dataset, criterion_type, weights, graph = True, verbose = True):
    pda_A      = np.zeros((len(dataset), len(dataset[0]) ))
    nda_A      = np.zeros((len(dataset), len(dataset[0]) ))
    pda_B      = np.zeros((len(dataset), len(dataset[0]) ))
    nda_B      = np.zeros((len(dataset), len(dataset[0]) ))
    pda_C      = np.zeros((len(dataset), len(dataset[0]) ))
    nda_C      = np.zeros((len(dataset), len(dataset[0]) ))
    dataset_A  = np.zeros((len(dataset), len(dataset[0]) ))
    dataset_B  = np.zeros((len(dataset), len(dataset[0]) ))
    dataset_C  = np.zeros((len(dataset), len(dataset[0]) ))    
    weights_A  = np.zeros(len(weights[0]))
    weights_B  = np.zeros(len(weights[0]))
    weights_C  = np.zeros(len(weights[0]))
    for j in range(0, dataset_A.shape[1]):
        weights_A[j] = weights[0][j][0]
        weights_B[j] = weights[0][j][1]
        weights_C[j] = weights[0][j][2]
        for i in range(0, dataset_A.shape[0]):
            a, b, c = dataset[i][j]
            dataset_A[i, j] = a
            dataset_B[i, j] = b
            dataset_C[i, j] = c
    col_mean_A = np.mean(dataset_A, axis = 0)
    col_mean_B = np.mean(dataset_B, axis = 0)
    col_mean_C = np.mean(dataset_C, axis = 0)
    for i in range(0, dataset_A.shape[0]):
        for j in range(0, dataset_A.shape[1]):
            if (criterion_type[j] == 'max'):
                pda_A[i,j] = max(0,  dataset_A[i,j] - col_mean_A[j]) / col_mean_A[j]
                nda_A[i,j] = max(0, -dataset_A[i,j] + col_mean_A[j]) / col_mean_A[j]
                pda_B[i,j] = max(0,  dataset_B[i,j] - col_mean_B[j]) / col_mean_B[j]
                nda_B[i,j] = max(0, -dataset_B[i,j] + col_mean_B[j]) / col_mean_B[j]
                pda_C[i,j] = max(0,  dataset_C[i,j] - col_mean_C[j]) / col_mean_C[j]
                nda_C[i,j] = max(0, -dataset_C[i,j] + col_mean_C[j]) / col_mean_C[j]
            else:
                pda_A[i,j] = max(0, -dataset_A[i,j] + col_mean_A[j]) / col_mean_A[j]
                nda_A[i,j] = max(0,  dataset_A[i,j] - col_mean_A[j]) / col_mean_A[j]
                pda_B[i,j] = max(0, -dataset_B[i,j] + col_mean_B[j]) / col_mean_B[j]
                nda_B[i,j] = max(0,  dataset_B[i,j] - col_mean_B[j]) / col_mean_B[j]
                pda_C[i,j] = max(0, -dataset_C[i,j] + col_mean_C[j]) / col_mean_C[j]
                nda_C[i,j] = max(0,  dataset_C[i,j] - col_mean_C[j]) / col_mean_C[j]
    w_pda_A = pda_A*weights_A
    w_pda_B = pda_B*weights_B
    w_pda_C = pda_C*weights_C
    w_nda_A = nda_A*weights_A
    w_nda_B = nda_B*weights_B
    w_nda_C = nda_C*weights_C
    s_p_A   = np.sum(w_pda_A, axis = 1)
    s_p_B   = np.sum(w_pda_B, axis = 1)
    s_p_C   = np.sum(w_pda_C, axis = 1)
    s_n_A   = np.sum(w_nda_A, axis = 1)
    s_n_B   = np.sum(w_nda_B, axis = 1)
    s_n_C   = np.sum(w_nda_C, axis = 1)
    n_s_p_A = s_p_A/max(s_p_A)
    n_s_p_B = s_p_B/max(s_p_C)
    n_s_p_C = s_p_C/max(s_p_B)
    n_s_n_A = 1 - s_n_A/max(s_n_A)
    n_s_n_B = 1 - s_n_B/max(s_n_B)
    n_s_n_C = 1 - s_n_C/max(s_n_C)
    a_s_A   = (1/2)*(n_s_p_A + n_s_n_A)
    a_s_B   = (1/2)*(n_s_p_B + n_s_n_B)
    a_s_C   = (1/2)*(n_s_p_C + n_s_n_C)
    a_s     = (1/3)*(a_s_A + a_s_B + a_s_C)
    if (verbose == True):
        for i in range(0, a_s.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(a_s[i], 4)))
    if ( graph == True):
        flow = np.copy(a_s)
        flow = np.reshape(flow, (a_s.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, a_s.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return a_s

###############################################################################