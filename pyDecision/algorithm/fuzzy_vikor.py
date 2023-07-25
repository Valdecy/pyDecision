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

# Function: VIKOR
def fuzzy_vikor_method(dataset, weights, criterion_type, strategy_coefficient = 0.5, graph = True, verbose = True):
    dataset_A  = np.zeros((len(dataset), len(dataset[0]) ))
    dataset_B  = np.zeros((len(dataset), len(dataset[0]) ))
    dataset_C  = np.zeros((len(dataset), len(dataset[0]) ))  
    weights_A  = np.zeros(len(weights[0]))
    weights_B  = np.zeros(len(weights[0]))
    weights_C  = np.zeros(len(weights[0]))
    best_A  = np.zeros(dataset_A.shape[1])
    best_B  = np.zeros(dataset_A.shape[1])
    best_C  = np.zeros(dataset_A.shape[1])
    worst_A = np.zeros(dataset_A.shape[1])
    worst_B = np.zeros(dataset_A.shape[1])
    worst_C = np.zeros(dataset_A.shape[1])
    for j in range(0, dataset_A.shape[1]):
        weights_A[j] = weights[0][j][0]
        weights_B[j] = weights[0][j][1]
        weights_C[j] = weights[0][j][2]
        for i in range(0, dataset_A.shape[0]):
            a, b, c = dataset[i][j]
            dataset_A[i, j] = a
            dataset_B[i, j] = b
            dataset_C[i, j] = c
    for i in range(0, dataset_A.shape[1]):
        if (criterion_type[i] == 'max'):
            best_A[i]  = np.max(dataset_A[:, i])
            best_B[i]  = np.max(dataset_B[:, i])
            best_C[i]  = np.max(dataset_C[:, i])
            worst_A[i] = np.min(dataset_A[:, i])
            worst_B[i] = np.min(dataset_B[:, i])
            worst_C[i] = np.min(dataset_C[:, i])
        else:
            best_A[i]  = np.min(dataset_A[:, i])
            best_B[i]  = np.min(dataset_B[:, i])
            best_C[i]  = np.min(dataset_C[:, i])
            worst_A[i] = np.max(dataset_A[:, i])
            worst_B[i] = np.max(dataset_B[:, i])
            worst_C[i] = np.max(dataset_C[:, i]) 
    s_i_A = weights_A*( abs(best_A - dataset_A) / abs(best_A - worst_A ) )
    s_i_B = weights_B*( abs(best_B - dataset_B) / abs(best_B - worst_B) )
    s_i_C = weights_C*( abs(best_C - dataset_C) / abs(best_C - worst_C) )
    r_i_A = np.max(s_i_A, axis = 1)
    r_i_B = np.max(s_i_B, axis = 1)
    r_i_C = np.max(s_i_C, axis = 1)
    s_i_A = np.sum(s_i_A, axis = 1)
    s_i_B = np.sum(s_i_B, axis = 1)
    s_i_C = np.sum(s_i_C, axis = 1)
    s_best_A  = np.min(s_i_A)
    s_best_B  = np.min(s_i_B)
    s_best_C  = np.min(s_i_C)
    s_worst_A = np.max(s_i_A)
    s_worst_B = np.max(s_i_B)
    s_worst_C = np.max(s_i_C)
    r_best_A  = np.min(r_i_A)
    r_best_B  = np.min(r_i_B)
    r_best_C  = np.min(r_i_C)
    r_worst_A = np.max(r_i_A)
    r_worst_B = np.max(r_i_B)
    r_worst_C = np.max(r_i_C)
    q_i_A = strategy_coefficient*( (s_i_A - s_best_A) / (s_worst_A - s_best_A) ) + (1 - strategy_coefficient)*( (r_i_A - r_best_A) / (r_worst_A - r_best_A) )
    q_i_B = strategy_coefficient*( (s_i_B - s_best_B) / (s_worst_B - s_best_B) ) + (1 - strategy_coefficient)*( (r_i_B - r_best_B) / (r_worst_B - r_best_B) )
    q_i_C = strategy_coefficient*( (s_i_C - s_best_C) / (s_worst_C - s_best_C) ) + (1 - strategy_coefficient)*( (r_i_C - r_best_C) / (r_worst_C - r_best_C) )
    s_i = (1/3)*(s_i_A + s_i_B + s_i_C)
    r_i = (1/3)*(r_i_A + r_i_B + r_i_C)
    q_i = (1/3)*(q_i_A + q_i_B + q_i_C)
    dq = 1 /(dataset_A.shape[0] - 1)
    flow_s = np.copy(s_i)
    flow_s = np.reshape(flow_s, (s_i.shape[0], 1))
    flow_s = np.insert(flow_s, 0, list(range(1, s_i.shape[0]+1)), axis = 1)
    flow_s = flow_s[np.argsort(flow_s[:, 1])]
    flow_r = np.copy(r_i)
    flow_r = np.reshape(flow_r, (r_i.shape[0], 1))
    flow_r = np.insert(flow_r, 0, list(range(1, r_i.shape[0]+1)), axis = 1)
    flow_r = flow_r[np.argsort(flow_r[:, 1])]
    flow_q = np.copy(q_i)
    flow_q = np.reshape(flow_q, (q_i.shape[0], 1))
    flow_q = np.insert(flow_q, 0, list(range(1, q_i.shape[0]+1)), axis = 1)
    flow_q = flow_q[np.argsort(flow_q[:, 1])]
    condition_1 = False
    condition_2 = False
    if (flow_q[1, 1] - flow_q[0, 1] >= dq):
        condition_1 = True
    if (flow_q[0,0] == flow_s[0,0] or flow_q[0,0] == flow_r[0,0]):
        condition_2 = True
    solution = np.copy(flow_q)
    if (condition_1 == True and condition_2 == False):
        solution = np.copy(flow_q[0:2,:])
    elif (condition_1 == False and condition_2 == True):
        for i in range(solution.shape[0] -1, -1, -1):
            if(solution[i, 1] - solution[0, 1] >= dq):
              solution = np.delete(solution, i, axis = 0)  
    if (verbose == True):
        for i in range(0, solution.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(solution[i, 0], 2)))
    if ( graph == True):
        ranking(solution) 
    return flow_s, flow_r, flow_q, solution

###############################################################################
