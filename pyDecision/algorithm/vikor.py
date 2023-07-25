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
def vikor_method(dataset, weights, criterion_type, strategy_coefficient = 0.5, graph = True, verbose = True):
    X     = np.copy(dataset)
    w     = np.copy(weights)
    best  = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])
    for i in range(0, dataset.shape[1]):
        if (criterion_type[i] == 'max'):
            best[i]  = np.max(X[:, i])
            worst[i] = np.min(X[:, i])
        else:
            best[i]  = np.min(X[:, i])
            worst[i] = np.max(X[:, i]) 
    s_i = w * ( abs(best - X) / (abs(best - worst) + 0.0000000000000001) )
    r_i = np.max(s_i, axis = 1)
    s_i = np.sum(s_i, axis = 1)
    s_best = np.min(s_i)
    s_worst = np.max(s_i)
    r_best = np.min(r_i)
    r_worst = np.max(r_i)
    q_i = strategy_coefficient*( (s_i - s_best) / (s_worst - s_best) ) + (1 - strategy_coefficient)*( (r_i - r_best) / (r_worst - r_best) )
    dq = 1 /(X.shape[0] - 1)
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
