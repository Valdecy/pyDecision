###############################################################################

# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import quad

###############################################################################

# Function: Distance Matrix
def distance_matrix(dataset, criteria = 0):
    distance_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance_array[i,j] = dataset[i, criteria] - dataset[j, criteria] 
    return distance_array

# Optimized Integration
def integration_preference_degree(dataset, W, Q, S, P, F):
    pd_array = np.zeros((dataset.shape[0], dataset.shape[0]))
    
    ###############################################################################
    
    def preference_function(distance, Fk, Qk, Pk, Sk):
        if (Fk == 't1'):
            return 1 if distance > 0 else 0
        elif (Fk == 't2'):
            return 1 if distance > Qk else 0
        elif (Fk == 't3'):
            return distance / Pk if 0 < distance <= Pk else (1 if distance > Pk else 0)
        elif (Fk == 't4'):
            return 0.5 if Qk < distance <= Pk else (1 if distance > Pk else 0)
        elif (Fk == 't5'):
            return (distance - Qk) / (Pk - Qk) if Qk < distance <= Pk else (1 if distance > Pk else 0)
        elif (Fk == 't6'):
            return 1 - math.exp(-(distance**2) / (2 * Sk**2)) if distance > 0 else 0
        else:
            return 0  
    
    ###############################################################################
    
    for k in range(0, dataset.shape[1]):
        distances = distance_matrix(dataset, criteria = k)
        for i in range(0, distances.shape[0]):
            for j in range(0, distances.shape[1]):
                if (i != j):
                    distance = distances[i, j]
                    if (distance > 0):
                        area, _        = quad(preference_function, 0, distance, args = (F[k], Q[k], P[k], S[k]))
                        pd_array[i, j] = pd_array[i, j] + W[k] * area
    
    pd_array = pd_array/sum(W)
    return pd_array

# Function: Rank 
def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        if (flow[i,1] >= 0):
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        else:
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12,ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))            
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    xmin = np.amin(rank_xy[:,0])
    xmax = np.amax(rank_xy[:,0])
    axes.set_xlim([xmin-1, xmax+1])
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

# Function: Promethee IV
def promethee_iv(dataset, W, Q, S, P, F, sort = True, steps = 0.001, topn = 0, graph = False, verbose = True):
    pd_matrix  = integration_preference_degree(dataset, W, Q, S, P, F)
    flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
    flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
    flow       = flow_plus - flow_minus 
    flow       = np.reshape(flow, (pd_matrix.shape[0], 1))
    flow       = np.insert(flow, 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    if (sort == True or graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
    if (topn > 0):
        if (topn > pd_matrix.shape[0]):
            topn = pd_matrix.shape[0]
        if (verbose == True):
            for i in range(0, topn):
                print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3)))  
    if (graph == True):
        ranking(flow)
    return flow

###############################################################################
