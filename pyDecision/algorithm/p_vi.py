###############################################################################

# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import os

###############################################################################

# Function: Distance Matrix
def distance_matrix(dataset, criteria = 0):
    distance_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance_array[i,j] = dataset[i, criteria] - dataset[j, criteria] 
    return distance_array

# Function: Preferences
def preference_degree(dataset, W, Q, S, P, F):
    pd_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for k in range(0, dataset.shape[1]):
        distance_array = distance_matrix(dataset, criteria = k)
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                if (i != j):
                    if (F[k] == 't1'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't2'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't3'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > 0 and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  = distance_array[i,j]/P[k]
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't4'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  = 0.5
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't5'):
                        if (distance_array[i,j] <= Q[k]):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                            distance_array[i,j]  =  (distance_array[i,j] - Q[k])/(P[k] -  Q[k])
                        else:
                            distance_array[i,j] = 1
                    if (F[k] == 't6'):
                        if (distance_array[i,j] <= 0):
                            distance_array[i,j]  = 0
                        else:
                            distance_array[i,j] = 1 - math.exp(-(distance_array[i,j]**2)/(2*S[k]**2))
                    if (F[k] == 't7'):
                        if (distance_array[i,j] == 0):
                            distance_array[i,j]  = 0
                        elif (distance_array[i,j] > 0 and distance_array[i,j] <= S[k]):
                            distance_array[i,j]  =  (distance_array[i,j]/S[k])**0.5
                        elif (distance_array[i,j] > S[k] ):
                            distance_array[i,j] = 1
        pd_array = pd_array + W[k]*distance_array
    pd_array = pd_array/sum(W)
    return pd_array

# Function: Rank 
def ranking(flow_1, flow_2, flow_3):    
    rank_xy = np.zeros((flow_1.shape[0] + 1, 6)) 
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = -1
        rank_xy[i, 1] = flow_1.shape[0]-i+1     
        rank_xy[i, 2] = 0
        rank_xy[i, 3] = flow_2.shape[0]-i+1  
        rank_xy[i, 4] = 1
        rank_xy[i, 5] = flow_3.shape[0]-i+1    
    plt.text(rank_xy[0, 0],  rank_xy[0, 1], 'Lower', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 2],  rank_xy[0, 3], 'Favorable', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 4],  rank_xy[0, 5], 'Upper', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    for i in range(1, rank_xy.shape[0]-1):
        if (flow_1[i,1] >= 0):
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow_1[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = "round", ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        elif (flow_1[i,1] < 0):
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow_1[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))  
        if (flow_2[i,1] >= 0):
            plt.text(rank_xy[i, 2],  rank_xy[i, 3], 'a' + str(int(flow_2[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        elif (flow_2[i,1] < 0):
            plt.text(rank_xy[i, 2],  rank_xy[i, 3], 'a' + str(int(flow_2[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))  
        if (flow_3[i,1] >= 0):
            plt.text(rank_xy[i, 4],  rank_xy[i, 5], 'a' + str(int(flow_3[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        elif (flow_3[i,1] < 0):
            plt.text(rank_xy[i, 4],  rank_xy[i, 5], 'a' + str(int(flow_3[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))  
    for i in range(1, rank_xy.shape[0]-2):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        plt.arrow(rank_xy[i, 2], rank_xy[i, 3], rank_xy[i+1, 2] - rank_xy[i, 2], rank_xy[i+1, 3] - rank_xy[i, 3], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
        plt.arrow(rank_xy[i, 4], rank_xy[i, 5], rank_xy[i+1, 4] - rank_xy[i, 4], rank_xy[i+1, 5] - rank_xy[i, 5], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-2, +2])
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

# Function: Promethee VI
def promethee_vi(dataset, W_lower, W_upper, Q, S, P, F, sort = True, topn = 0, iterations = 1000, graph = False, verbose = True):
    pd_matrix_1  = preference_degree(dataset, W_lower, Q, S, P, F)
    flow_plus_1  = np.sum(pd_matrix_1, axis = 1)/(pd_matrix_1.shape[0] - 1)
    flow_minus_1 = np.sum(pd_matrix_1, axis = 0)/(pd_matrix_1.shape[0] - 1)
    flow_1       = flow_plus_1 - flow_minus_1 
    pd_matrix_2  = preference_degree(dataset, W_upper, Q, S, P, F)
    flow_plus_2  = np.sum(pd_matrix_2, axis = 1)/(pd_matrix_2.shape[0] - 1)
    flow_minus_2 = np.sum(pd_matrix_2, axis = 0)/(pd_matrix_2.shape[0] - 1)
    flow_2       = flow_plus_2 - flow_minus_2 
    flow_3        = 0
    for i in range(0, iterations):
        random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        W = (W_upper - W_lower + 1) * random + W_lower
        pd_matrix  = preference_degree(dataset, W, Q, S, P, F)
        flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
        flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
        flow_3     = flow_3 + (flow_plus - flow_minus)   
    flow_3 = flow_3/iterations
    flow_1 = np.reshape(flow_1, (pd_matrix.shape[0], 1))
    flow_1 = np.insert(flow_1 , 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    flow_2 = np.reshape(flow_2, (pd_matrix.shape[0], 1))
    flow_2 = np.insert(flow_2 , 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    flow_3 = np.reshape(flow_3, (pd_matrix.shape[0], 1))
    flow_3 = np.insert(flow_3 , 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    if (sort == True or graph == True):
        flow_1 = flow_1[np.argsort(flow_1[:, 1])]
        flow_1 = flow_1[::-1]
        flow_2 = flow_2[np.argsort(flow_2[:, 1])]
        flow_2 = flow_2[::-1]
        flow_3 = flow_3[np.argsort(flow_3[:, 1])]
        flow_3 = flow_3[::-1]
    if (topn > 0):
        if (topn > pd_matrix.shape[0]):
            topn = pd_matrix.shape[0]
        if (verbose == True):
            for i in range(0, topn):
                print('a' + str(int(flow_3[i,0])) + ': ' + str(round(flow_3[i,1], 3))) 
    if (graph == True):
        ranking(flow_1, flow_2, flow_3)
    return flow_1, flow_3, flow_2

###############################################################################
