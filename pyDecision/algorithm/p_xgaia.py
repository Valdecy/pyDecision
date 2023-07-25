###############################################################################

# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import TruncatedSVD

##############################################################################

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
 
# Function: Promethee Gaia
def promethee_gaia(dataset, W, Q, S, P, F, size_x = 10, size_y = 10):
    pd_matrix   = preference_degree(dataset[:,0].reshape(-1,1), W, Q, S, P, F)
    flow_plus   = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
    flow_minus  = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
    flow        = flow_plus - flow_minus 
    flow_matrix = np.reshape(flow, (pd_matrix.shape[0], 1))
    for i in range(1, dataset.shape[1]):
        pd_matrix   = preference_degree(dataset[:,i].reshape(-1,1), W, Q, S, P, F)
        flow_plus   = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
        flow_minus  = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
        flow        = flow_plus - flow_minus 
        flow_matrix = np.hstack((flow_matrix, np.reshape(flow, (pd_matrix.shape[0], 1))))
    dataset = np.copy(flow_matrix)    
    tSVD         = TruncatedSVD(n_components = 2, n_iter = 100, random_state = 42)
    min_values   = np.min(dataset, axis = 0)
    max_values   = np.max(dataset, axis = 0)
    alternatives = tSVD.fit_transform(np.vstack((dataset, min_values, max_values)))
    criteria     = tSVD.fit_transform(dataset.T)
    variance     = sum(np.var(alternatives, axis = 0) / np.var(np.vstack((dataset, min_values, max_values)), axis = 0).sum()) # same as: sum(tSVD.explained_variance_ratio_)
    alts         = list(range(1, dataset.shape[0] + 1)) 
    alts         = ['a' + str(alt) for alt in alts]
    crits        = list(range(1, dataset.shape[1] + 1)) 
    crits        = ['g' + str(crit) for crit in crits]
    plt.figure(figsize = [size_x, size_y])
    for i in range(0, alternatives.shape[0]-2):
        plt.text(alternatives[i, 0],  alternatives[i, 1], alts[i], size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 1.0, 0.8),))
    plt.text(alternatives[-1, 0],  alternatives[-1, 1], '+', size = 8, ha = 'center', va = 'center', color = 'black', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.0, 1.0, 0.0),))
    plt.text(alternatives[-2, 0],  alternatives[-2, 1], ' - ', size = 8, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.0, 0.0),))
    for i in range(0, criteria .shape[0]):
        plt.text(criteria [i, 0],  criteria [i, 1], crits[i], size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.0, 0.0, 0.0),))
    for i in range(0, criteria.shape[0]):
        plt.arrow(0, 0, criteria[i, 0], criteria[i, 1], head_width = 0.005, head_length = 0.005, overhang = -0.2, color = 'blue', length_includes_head = True)
    axes = plt.gca()
    xmin = np.amin(alternatives[:,0])
    if (np.amin(criteria[:,0]) < xmin):
        xmin = np.amin(criteria[:,0])
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(alternatives[:,0])
    if (np.amax(criteria[:,0]) > xmax):
        xmax = np.amax(criteria[:,0])
    if (xmax < 0):
        xmax = 0
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(alternatives[:,1])
    if (np.amin(criteria[:,1]) < ymin):
        ymin = np.amin(criteria[:,1])
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(alternatives[:,1])
    if (np.amax(criteria[:,0]) > ymax):
        ymax = np.amax(criteria[:,1])
    if (ymax < 0):
        ymax = 0
    axes.set_ylim([ymin-1, ymax+1])        
    plt.axvline(x = 0, linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.axhline(y = 0, linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.xlabel('EV: ' + str(round(variance*100, 2)) + '%')
    plt.show()
    return

###############################################################################
