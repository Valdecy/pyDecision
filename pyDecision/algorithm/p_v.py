###############################################################################

# Required Libraries
import math
import numpy as np

from pyDecision.util.ga import genetic_algorithm

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

# Function: Promethee II
def promethee_ii(dataset, W, Q, S, P, F, sort = True, topn = 0):
    pd_matrix  = preference_degree(dataset, W, Q, S, P, F)
    flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
    flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
    flow       = flow_plus - flow_minus 
    flow       = np.reshape(flow, (pd_matrix.shape[0], 1))
    flow       = np.insert(flow, 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    if (sort == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
    if (topn > 0):
        if (topn > pd_matrix.shape[0]):
            topn = pd_matrix.shape[0]
        for i in range(0, topn):
            print("a" + str(int(flow[i,0])) + ": " + str(round(flow[i,1], 3)))         
    return flow

###############################################################################
    
# Function: Promethee V
def promethee_v(dataset, W, Q, S, P, F, sort = True, criteria = 1, cost = [], budget = 0, forbidden = [], iterations = 500, verbose = True):
    flow = promethee_ii(dataset, W, Q, S, P, F, sort = sort, topn = 0)
    def flow_set(variables):
        flow_sum = 0
        cost_sum = 0
        actions  = []
        obj_function_1 = 0
        obj_function_2 = 0
        obj_function_3 = 0
        for i in range(0, len(variables)):
            if (variables[i] > 0.5):
                flow_sum = flow_sum + flow[i,1]
                if (len(cost) > 0):
                    cost_sum = cost_sum + cost[i]
                actions.append('a' + str(int(flow[i,0])))
        if (flow_sum <= 0):
            flow_sum = 0.1
        if (len([num for num in variables if num > 0.5]) - criteria > 0):
            obj_function_1 = 1
        else:
            obj_function_1 = 0
        if (cost_sum > budget):
            obj_function_2 = 1
        else:
            obj_function_2 = 0
        for i in range(0, len(forbidden)):
            if ( set(forbidden[i]).issubset( set(actions) ) ):
                obj_function_3 = 1
            else:
                obj_function_3 = 0                
        return (1/flow_sum) + obj_function_1 + obj_function_2 + obj_function_3
    ga = genetic_algorithm(population_size = 100, mutation_rate = 0.1, elite = 0, min_values = [0]*flow.shape[0], max_values = [1]*flow.shape[0], eta = 1, mu = 1, generations = iterations, target_function = flow_set, verbose = verbose)
    for i in range(0, len(ga)-1):
        if (ga[i] > 0.5):
            ga[i] = 1
        else:
            ga[i] = 0
    ga      = np.reshape(ga[:-1], (flow.shape[0], 1))
    flow    = np.hstack((flow, ga))
    actions = []
    for i in range(0, flow.shape[0]):
        if (flow[i,2] == 1):
            actions.append('a' + str(int(flow[i,0])))
    print(actions)
    return flow

###############################################################################
