###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Pre-Order Rank 
def po_ranking(po_string):
    alts   = list(range(1, po_string.shape[0] + 1)) 
    alts   = ['a' + str(alt) for alt in alts]
    for i in range (po_string.shape[0] - 1, -1, -1):
        for j in range (po_string.shape[1] -1, -1, -1):
            if (po_string[i,j] == 'I'):
                po_string = np.delete(po_string, i, axis = 0)
                po_string = np.delete(po_string, i, axis = 1)
                alts[j] = str(alts[j] + "; " + alts[i])
                del alts[i]
                break    
    graph = {}
    for i in range(po_string.shape[0]):
        if (len(alts[i]) == 0):
            graph[alts[i]] = i 
        else:
            graph[alts[i][ :2]] = i   
            graph[alts[i][-2:]] = i 
    po_matrix = np.zeros((po_string.shape[0], po_string.shape[1]))
    for i in range (0, po_string.shape[0]):
        for j in range (0, po_string.shape[1]):
            if (po_string[i,j] == 'P+'):
                po_matrix[i,j] = 1
    col_sum   = np.sum(po_matrix, axis = 1)
    alts_rank = [x for _, x in sorted(zip(col_sum, alts))]
    if (np.sum(col_sum) != 0):
        alts_rank.reverse()      
    graph_rank = {}
    for i in range(po_string.shape[0]):
        if (len(alts_rank[i]) == 0):
            graph_rank[alts_rank[i]] = i 
        else:
            graph_rank[alts_rank[i][ :2]] = i   
            graph_rank[alts_rank[i][-2:]] = i
    rank = np.copy(po_matrix)
    for i in range(0, po_matrix.shape[0]):
        for j in range(0, po_matrix.shape[1]): 
            if (po_matrix[i,j] == 1):
                rank[i,:] = np.clip(rank[i,:] - rank[j,:], 0, 1)   
    rank_xy = np.zeros((len(alts_rank), 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        if (len(alts_rank) - np.sum(~rank.any(1)) != 0):
            rank_xy[i, 1] = len(alts_rank) - np.sum(~rank.any(1))
        else:
            rank_xy[i, 1] = 1
    for i in range(0, len(alts_rank) - 1):
        i1 = int(graph[alts_rank[ i ][:2]]) 
        i2 = int(graph[alts_rank[i+1][:2]])
        if (po_string[i1,i2] == 'P+'):
            rank_xy[i+1,1] = rank_xy[i+1,1] - 1
            for j in range(i+2, rank_xy.shape[0]):
                rank_xy[j,1] = rank_xy[i+1,1]
        if (po_string[i1,i2] == 'R'):
            rank_xy[i+1,0] = rank_xy[i,0] + 1            
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], alts_rank[i], size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, len(alts_rank)):
        alts_rank[i] = alts_rank[i][:2]
    for i in range(0, rank.shape[0]):
        for j in range(0, rank.shape[1]):
            k1 = int(graph_rank[list(graph.keys())[list(graph.values()).index(i)]])
            k2 = int(graph_rank[list(graph.keys())[list(graph.values()).index(j)]])
            if (rank[i, j] == 1):  
                plt.arrow(rank_xy[k1, 0], rank_xy[k1, 1], rank_xy[k2, 0] - rank_xy[k1, 0], rank_xy[k2, 1] - rank_xy[k1, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
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

# Function: Regime
def regime_method(dataset, weights, criterion_type):
    X       = np.copy(dataset)/1.0
    weights = weights/np.sum(weights)
    g_ind   = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, g_ind.shape[0]):
        for k in range(0, g_ind.shape[0]):
            for j in range(0, X.shape[1]):
                if (i!= k):
                    if (criterion_type[j] == 'max'):
                        if (X[i, j] >= X[k, j]):
                            g_ind[i, k] = g_ind[i, k] + weights[j]
                        else:
                            g_ind[i, k] = g_ind[i, k] - weights[j]
                    else:
                        if (X[i, j] < X[k, j]):
                            g_ind[i, k] = g_ind[i, k] + weights[j]
                        else:
                            g_ind[i, k] = g_ind[i, k] - weights[j]
    cp_matrix  = np.empty((X.shape[0], X.shape[0]), dtype = 'U25')
    cp_matrix.fill('-')
    for i in range(0, cp_matrix.shape[0]):
          for j in range(0, cp_matrix.shape[0]): 
              if (i != j):
                  if (g_ind[i, j] > 0):
                      cp_matrix[i,j] = 'P+'
                  if (g_ind[i, j] == 0 or g_ind[i, j] == g_ind[j, i]):
                      cp_matrix[i,j] = 'I'  
    po_ranking(cp_matrix)
    return cp_matrix

###############################################################################