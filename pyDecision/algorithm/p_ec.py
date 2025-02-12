###############################################################################

# Required Libraries
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from collections import Counter
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

###############################################################################

# Function: CRITIC (CRiteria Importance Through Intercriteria Correlation)
def critic_method(dataset, criterion_type):
    X     = np.copy(dataset)/1.0
    best  = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])
    for i in range(0, dataset.shape[1]):
        if (criterion_type[i] == 'max'):
            best[i]  = np.max(X[:, i])
            worst[i] = np.min(X[:, i])
        else:
            best[i]  = np.min(X[:, i])
            worst[i] = np.max(X[:, i])
        if (best[i] == worst[i]):
            best[i]  = best[i]  
            worst[i] = worst[i] 
    for j in range(0, X.shape[1]):
        X[:,j] = ( X[:,j] - worst[j] ) / ( best[j] - worst[j]  + 1e-10)
    std      = (np.sum((X - X.mean())**2, axis = 0)/(X.shape[0] - 1))**(1/2)
    sim_mat  = np.corrcoef(X.T)
    sim_mat  = np.nan_to_num(sim_mat)
    conflict = np.sum(1 - sim_mat, axis = 1)
    infor    = std*conflict
    weights  = infor/np.sum(infor)
    return weights

###############################################################################

# Function: Entropy
def entropy_method(dataset, criterion_type):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:,j] =  X[:,j] / np.sum(X[:,j])
        else:
            X[:,j] = (1 / X[:,j]) / np.sum((1 / X[:,j]))
    X = np.abs(X)
    H = np.zeros((X.shape))
    for j, i in itertools.product(range(H.shape[1]), range(H.shape[0])):
        if (X[i, j]):
            H[i, j] = X[i, j] * np.log(X[i, j] + 1e-9)
    h = np.sum(H, axis = 0) * (-1 * ((np.log(H.shape[0] + 1e-9)) ** (-1)))
    d = 1 - h
    d = d + 1e-9
    w = d / (np.sum(d))
    return w

###############################################################################

# Function: Rank
def solution_p_ranking(p2):
    flow_0  = np.arange(1, p2.shape[1] + 1)
    flow_1  = np.sum(p2, axis = 0)
    flow    = np.column_stack((flow_0, flow_1))
    flow    = flow[np.argsort(flow[:, 1])]
    flow    = flow[::-1]
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i
    for i in range(0, rank_xy.shape[0]):
        if (flow[i,1] >= 0):
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        else:
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))
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
def promethee_ii(dataset, W, Q, S, P, F, sort = True, topn = 0, verbose = True):
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
        if (verbose == True):
            for i in range(0, topn):
                print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3)))
    return flow

###############################################################################

# Function: Generate Ranks
def generate_rank_array(arr, sorted_indices):
    rank_array = np.zeros(len(arr), dtype = int)
    for rank, index in enumerate(sorted_indices, start = 1):
        rank_array[index] = rank
    return rank_array

# Function: Find Mode
def find_column_modes(matrix):
    transposed_matrix = np.transpose(matrix)
    mode_list         = []
    for column in transposed_matrix:
        counter   = Counter(column)
        max_count = max(counter.values())
        modes     = [x for x, count in counter.items() if count == max_count]
        mode_list.append(modes)
    return mode_list

# Function: Tranpose Dictionary
def transpose_dict(rank_count_dict):
    transposed_dict = {}
    list_length     = len(next(iter(rank_count_dict.values())))
    for i in range(list_length):
        transposed_dict[i+1] = [values[i] for values in rank_count_dict.values()]
    return transposed_dict

# Function: Plot Ranks
def plot_rank_freq(rank, size_x = 8, size_y = 10):
    flag_1             = 0
    ranks              = rank.T
    alternative_labels = [f'a{i+1}' for i in range(ranks.shape[0])]
    rank_count_dict    = {i+1: [0]*ranks.shape[0] for i in range(0, ranks.shape[0])}
    for i in range(0, ranks.shape[0]):
        for j in range(0, ranks.shape[1]):
            rank = int(ranks[i, j])
            rank_count_dict[i+1][rank-1] = rank_count_dict[i+1][rank-1] + 1
    rank_count_dict = transpose_dict(rank_count_dict)
    fig, ax         = plt.subplots(figsize = (size_x, size_y))
    try:
      cmap   = colormaps.get_cmap('tab20')
      colors = [cmap(i) for i in np.linspace(0, 1, ranks.shape[0])]
    except:
      colors = plt.cm.get_cmap('tab20', ranks.shape[0])
      flag_1 = 1
    bottom = np.zeros(len(alternative_labels))
    for rank, counts in rank_count_dict.items():
        if (flag_1 == 0):
          bars = ax.barh(alternative_labels, counts, left = bottom, color = colors[rank-1])
        else:
          bars = ax.barh(alternative_labels, counts, left = bottom, color = colors(rank-1))
        bottom = bottom + counts
        for rect, c in zip(bars, counts):
            if (c > 0):
                width = rect.get_width()
                ax.text(width/2 + rect.get_x(), rect.get_y() + rect.get_height() / 2, f"r{rank} ({c})", ha = 'center', va = 'center', color = 'black')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax.tick_params(axis = 'y', which = 'both', pad = 25) 
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Alternative')
    ax.set_title('Rank Frequency per Alternative')
    plt.show()
    return

# Function: EC PROMETHEE   
def ec_promethee(dataset, criterion_type, custom_sets, Q, S, P, F, iterations = 10, verbose = True):
    X                 = np.copy(dataset).astype(float)
    min_indices       = np.where(np.array(criterion_type) == 'min')[0]
    X[:, min_indices] = 1.0 / X[:, min_indices]
    critic_weights    = critic_method( dataset, criterion_type)
    entropy_weights   = entropy_method(dataset, criterion_type)
    ranks_matrix      = []
    wnorm_matrix      = []
    p2_matrix         = []
    lower_upper_pairs = []
    sol               = []
    if (verbose == True):
        print ('Entropy Weights:')
        formatted         = ['{:.3f}'.format(val) for val in entropy_weights ]
        print('[' + ', '.join(formatted) + ']')
        print('')
        print ('CRITIC Weights:')
        formatted         = ['{:.3f}'.format(val) for val in critic_weights]
        print('[' + ', '.join(formatted) + ']')
    for i in range(len(critic_weights)):
        all_weights = [critic_weights[i], entropy_weights[i]]
        if (custom_sets):
            for custom_set in custom_sets:
                total          = sum(custom_set)
                normalized_set = [x/total for x in custom_set] if total else custom_set
                if (i < len(custom_set)):
                    all_weights.append(normalized_set[i]) 
        lower = min(all_weights)
        lower = max(1e-9, lower)
        upper = max(all_weights)
        lower_upper_pairs.append((lower, upper))
    if (verbose == True):
        if (custom_sets):
            for custom_set in custom_sets:
                count          = 1
                total          = sum(custom_set)
                normalized_set = [x/total for x in custom_set] if total else custom_set
                print('')
                print ('Custom Weights', str(count), ':')
                formatted = ['{:.3f}'.format(val) for val in normalized_set ]
                print('[' + ', '.join(formatted) + ']')
                count     = count + 1
        print('')
        print ('Lower:')
        formatted = ['{:.3f}'.format(lower) for lower, upper in lower_upper_pairs ]
        print('[' + ', '.join(formatted) + ']')
        print('')
        print ('Upper:')
        formatted = ['{:.3f}'.format(upper) for lower, upper in lower_upper_pairs ]
        print('[' + ', '.join(formatted) + ']')
    for _ in range(iterations):
        random_weights   = np.array([random.uniform(lower, upper) for lower, upper in lower_upper_pairs])
        wnorm_matrix.append(random_weights)
        promethee_result = promethee_ii(X, random_weights, Q, S, P, F, sort = False, topn = 0, verbose = False)
        p2_matrix.append(promethee_result[:,-1])
        ranks            = np.argsort(promethee_result[:, 1])[::-1]
        ranks            = generate_rank_array(promethee_result[:, 1], ranks)
        ranks_matrix.append(ranks)
    wnorm_matrix = np.array(wnorm_matrix)
    ranks_matrix = np.array(ranks_matrix)
    p2_matrix    = np.array(p2_matrix)
    p2_sum       = np.sum(p2_matrix, axis = 0)
    p2_rank      = np.argsort(p2_sum)[::-1]
    sol          = generate_rank_array(p2_sum, p2_rank)
    return wnorm_matrix, ranks_matrix, p2_matrix, sol

###############################################################################
