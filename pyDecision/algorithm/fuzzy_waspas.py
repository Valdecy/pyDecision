###############################################################################

# Required Libraries
import copy
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank 
def ranking_m(flow_1, flow_2, flow_3):    
    rank_xy = np.zeros((flow_1.shape[0] + 1, 6)) 
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = -1
        rank_xy[i, 1] = flow_1.shape[0]-i+1     
        rank_xy[i, 2] = 0
        rank_xy[i, 3] = flow_2.shape[0]-i+1  
        rank_xy[i, 4] = 1
        rank_xy[i, 5] = flow_3.shape[0]-i+1    
    plt.text(rank_xy[0, 0],  rank_xy[0, 1], 'F-WSM', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 2],  rank_xy[0, 3], 'F-WPM', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    plt.text(rank_xy[0, 4],  rank_xy[0, 5], 'F-WASPAS', size = 12, ha = 'center', va = 'center', color = 'white', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0, 0, 0),))
    for i in range(1, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow_1[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = "round", ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
        plt.text(rank_xy[i, 2],  rank_xy[i, 3], 'a' + str(int(flow_2[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
        plt.text(rank_xy[i, 4],  rank_xy[i, 5], 'a' + str(int(flow_3[i-1,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),)) 
    for i in range(1, rank_xy.shape[0]-1):
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

# Function: F-WASPAS
def fuzzy_waspas_method(dataset, criterion_type, weights, graph = True):
    x          = copy.deepcopy(dataset)
    max_values = [max(max(outer_list[i]) for outer_list in dataset) for i in range(0, len(dataset[0]))] 
    min_values = [min(min(outer_list[i]) for outer_list in dataset) for i in range(0, len(dataset[0]))]
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[0])):
            if (criterion_type[j] == 'max'):
                a, b, c = x[i][j]
                d, e, f = weights[0][j]
                a, b, c = (a/max_values[j])*d, (b/max_values[j])*e, (c/max_values[j])*f
                x[i][j] = (a, b, c)
            else:
                a, b, c = x[i][j]
                d, e, f = weights[0][j]
                a, b, c = (min_values[j]/a)*d, (min_values[j]/b)*e, (min_values[j]/c)*f
                x[i][j] = (a, b, c)
    sums_dataset = []
    prod_dataset = []
    for row in dataset:
        sums_row = []
        prod_row = []
        for inner_list in row:
            sum_value = sum(inner_list)/3
            prd_value = np.prod(inner_list)/3
            sums_row.append([sum_value])
            prod_row.append([prd_value])
        sums_dataset.append(sums_row)
        prod_dataset.append(prod_row)
    prod_dataset = np.array(prod_dataset)
    rows_sums_q  = np.sum(prod_dataset, axis = 1)
    f_wpm        = [i[0] for i in rows_sums_q]
    sums_dataset = np.array(sums_dataset)
    rows_sums_p  = np.sum(sums_dataset, axis = 1)
    f_wsm        = [i[0] for i in rows_sums_p]
    lmbd         = sum(rows_sums_p)[0]/ (sum(rows_sums_q)[0] + sum(rows_sums_p)[0])
    f_waspas     = [lmbd*f_wsm[i] + (1 - lmbd)*f_wpm[i] for i in range(0, len(f_wsm))]
    if (graph == True):
        flow_1 = np.array(f_wsm)
        flow_1 = np.reshape(flow_1, (len(f_wsm), 1))
        flow_1 = np.insert(flow_1, 0, list(range(1, len(f_wsm)+1)), axis = 1)
        flow_2 = np.array(f_wpm)
        flow_2 = np.reshape(flow_2, (len(f_wsm), 1))
        flow_2 = np.insert(flow_2, 0, list(range(1, len(f_wsm)+1)), axis = 1)
        flow_3 = np.array(f_waspas)
        flow_3 = np.reshape(flow_3, (len(f_wsm), 1))
        flow_3 = np.insert(flow_3, 0, list(range(1,len(f_wsm)+1)), axis = 1)
        ranking_m(flow_1[np.argsort(flow_1[:, 1])[::-1]], flow_2[np.argsort(flow_2[:, 1])[::-1]], flow_3[np.argsort(flow_3[:, 1])[::-1]])
    return f_wsm, f_wpm, f_waspas

###############################################################################