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

###############################################################################

# Function: LMAW (Logarithm Methodology of Additive Weights) 
def lmaw_method(performance_matrix, weights, criterion_type, graph = True, verbose = True):
    performance_matrix  = performance_matrix.astype(np.float64)
    m, n                = performance_matrix.shape
    weights             = np.asarray(weights, dtype = np.float64).reshape(1, n)
    weights             = weights/np.sum(weights) 
    eps                 = np.finfo(np.float64).eps
    col_max             = np.max(performance_matrix, axis = 0)
    col_min             = np.min(performance_matrix, axis = 0)
    standardized_matrix = np.empty_like(performance_matrix)
    for j in range(n):
        if (criterion_type[j] == 'max'):
            denom                     = col_max[j] if np.abs(col_max[j]) > eps else eps
            standardized_matrix[:, j] = (performance_matrix[:, j] + col_max[j]) / denom
        else: 
            denom                     = np.where(np.abs(performance_matrix[:, j]) < eps, eps, performance_matrix[:, j])
            standardized_matrix[:, j] = (performance_matrix[:, j] + col_min[j]) / denom
    standardized_matrix = np.where(standardized_matrix <= 0, eps, standardized_matrix)
    log_std             = np.log(standardized_matrix) 
    prod_g              = np.prod(standardized_matrix, axis = 0)
    safe_prod_g         = np.where(prod_g <= 0, eps, prod_g)
    log_prod            = np.log(safe_prod_g)
    log_prod            = np.where(np.abs(log_prod) < eps, eps, log_prod)
    phi_matrix          = log_std / log_prod
    phi_w               = np.power(phi_matrix, weights)
    two_minus_phi_w     = np.power(2 - phi_matrix, weights)
    denominator         = two_minus_phi_w + phi_w
    denominator         = np.where(np.abs(denominator) < eps, eps, denominator)
    weighted_matrix     = (2 * phi_w) / denominator
    ranking_scores      = np.sum(weighted_matrix, axis = 1)
    if (verbose == True):
        for i in range(0, ranking_scores .shape[0]):
            print('a' + str(i+1) + ': ' + str(round(ranking_scores [i], 4)))
    if ( graph == True):
        flow = np.copy(ranking_scores)
        flow = np.reshape(flow, (ranking_scores .shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, ranking_scores .shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return ranking_scores

###############################################################################