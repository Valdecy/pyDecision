###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import block_diag
from scipy.optimize import linprog

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

# Function: OPA (Ordinal Priority Approach)
def opa_method(experts_rank = [], experts_rank_criteria = [], experts_rank_alternatives = [], graph = True, verbose = True):
    experts      = len(experts_rank)
    criteria     = len(experts_rank_criteria[0])
    alternatives = len(experts_rank_alternatives[0])
    experts_rc   = np.vstack(experts_rank_criteria)
    experts_ra   = np.vstack(experts_rank_alternatives)
    C            = []
    index_temp   = 0
    
    for e in range(experts):
        experts_ra_temp = experts_ra[e * alternatives:(e + 1) * alternatives, :]
        for i in range(criteria):
            index_temp = index_temp + 1
            mini_A     = np.zeros((alternatives, alternatives))
            for j in range(alternatives):
                if (j == alternatives - 1):
                    index            = np.where(experts_ra_temp[:, i] == alternatives)[0]
                    mini_A[j, index] = experts_rank[e] * experts_rc[e, i] * (j + 1)
                    break
                for k in range(j, j + 2):
                    index = np.where(experts_ra_temp[:, i] == (k + 1))[0]
                    if (k == j):
                        mini_A[j, index] =   experts_rank[e] * experts_rc[e, i] * (j + 1)
                    elif (k == j + 1):
                        mini_A[j, index] = -(experts_rank[e] * experts_rc[e, i] * (j + 1))
            C.append(mini_A)

    A      = block_diag(*C)
    A      = np.hstack([np.ones((A.shape[0], 1)), A])
    f      = -np.hstack([1, np.zeros(experts * criteria * alternatives)])
    A_ineq = A
    b_ineq = np.zeros(A.shape[0])
    A_eq   = np.hstack([0, np.ones(A.shape[1] - 1)]).reshape(1, -1)
    b_eq   = np.array([1])
    lb     = [-np.inf] + [0] * (A.shape[1] - 1)
    result = linprog(f, A_ub = A_ineq, b_ub = b_ineq, A_eq = A_eq, b_eq = b_eq, bounds = [(lb[i], None) for i in range(len(lb))], method = 'highs')
    if (result.success):
        x         = result.x
        x_weight  = x[1:]
        w_experts = np.split(x_weight, experts)
        w_e       = []
        if (verbose == True):
            print('Weights - Experts')
        for i, w in enumerate(w_experts, 1):
            if (verbose == True):
                print(f'Expert {i}: {np.sum(w):.4f}')
            w_e.append(np.sum(w))
        w_criteria_mat = np.zeros((experts, criteria))
        for i in range(0, experts):
            w_criteria = np.split(w_experts[i], criteria)
            for j in range(0, criteria):
                w_criteria_mat[i, j] = np.sum(w_criteria[j])
        w_c = []
        for i in range(0, criteria):
            w_c.append(np.sum(w_criteria_mat[:, i], axis = 0))
        if (verbose == True):
            print('')
            print('Weights - Criteria')
        if (verbose == True):
            for i in range(0, len(w_c)):
                print(f'g{i+1}: {w_c[i]:.4f}')
        w_a = []
        if (verbose == True):
            print('')
            print('Weights - Alternatives')
        for i in range(0, alternatives):
            index = np.arange(i, experts * criteria * alternatives, alternatives)
            if (verbose == True):
                print(f'a{i + 1}: {np.sum(x_weight[index]):.4f}')
            w_a.append(np.sum(x_weight[index]))
    else:
        print('Optimization failed.')
    if ( graph == True):
        a_s  = np.array(w_a)
        flow = np.copy(a_s)
        flow = np.reshape(flow, (a_s.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, a_s.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return w_e, w_c, w_a

###############################################################################