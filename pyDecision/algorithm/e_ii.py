###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

###############################################################################

# Code available at: https://gist.github.com/qpwo/272df112928391b2c83a3b67732a5c25
# Author: Luke Harold Miles
# email: luke@cs.uky.edu
# Site: https://lukemiles.org

# Function: Cycle Finder
def simple_cycles(G):
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()
    G    = {v: set(nbrs) for (v,nbrs) in G.items()}
    sccs = strongly_connected_components(G)
    while sccs:
        scc       = sccs.pop()
        startnode = scc.pop()
        path      = [startnode]
        blocked   = set()
        closed    = set()
        blocked.add(startnode)
        B     = defaultdict(set)
        stack = [ (startnode,list(G[startnode])) ]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append( (nextnode, list(G[nextnode])) )
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs:
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                path.pop()
        remove_node(G, startnode)
        H = subgraph(G, set(scc))
        sccs.extend(strongly_connected_components(H))

# Function: SCC       
def strongly_connected_components(graph):
    index_counter = [0]
    stack         = []
    lowlink       = {}
    index         = {}
    result        = []   
    def _strong_connect(node):
        index[node]   = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node) 
        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node],lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node],index[successor])
        if lowlink[node] == index[node]:
            connected_component = []
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            result.append(connected_component[:])
    for node in graph:
        if node not in index:
            _strong_connect(node)
    return result

# Function: Remove Node
def remove_node(G, target):
    del G[target]
    for nbrs in G.values():
        nbrs.discard(target)

# Function: Subgraph
def subgraph(G, vertices):
    return {v: G[v] & vertices for v in vertices}

###############################################################################

# Function: Concordance Matrix
def concordance_matrix(dataset, W):
    concordance = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(0, concordance.shape[0]):
        for j in range(0, concordance.shape[1]):
            value = 0
            for k in range(0, dataset.shape[1]):
                if (dataset[i,k] >= dataset[j,k]):
                    value = value + W[k]
            concordance[i,j] = value      
    if (np.sum(W) != 0):
        concordance = concordance/np.sum(W)
    return concordance

# Function: Discordance Matrix
def discordance_matrix(dataset):
    delta       = np.max(np.amax(dataset, axis = 0) - np.amin(dataset, axis = 0))
    discordance = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(0, discordance.shape[0]):
        for j in range(0, discordance.shape[1]):
            discordance[i,j] = np.max((dataset[j,:]  - dataset[i,:]))/delta
            if (discordance[i,j] < 0):
                discordance[i,j] = 0    
    return discordance

# Function: Dominance Matrix
def dominance_matrix(concordance, discordance, c_minus = 0.65, c_zero = 0.75, c_plus = 0.85, d_minus = 0.25, d_plus = 0.50):
    dominance_s = np.zeros((concordance.shape[0], concordance.shape[0]))
    dominance_w = np.zeros((concordance.shape[0], concordance.shape[0]))
    for i in range (0, dominance_s.shape[0]):
        for j in range (0, dominance_s.shape[1]):
            if (concordance[i,j] >= concordance[j,i] and i != j):
                if ( ( (concordance[i,j] >= c_plus) and (discordance[i,j] <= d_plus ) ) or  ( (concordance[i,j] >= c_zero) and (discordance[i,j] <= d_minus) )):
                    dominance_s[i, j] = 1
                if (  (concordance[i,j] >= c_zero and discordance[i,j] <= d_plus) or (concordance[i,j] >= c_minus and discordance[i,j] <= d_minus) ):
                    dominance_w[i, j] = 1                 
    return dominance_s, dominance_w

# Function: Find Cycles and Unites it as a Single Criteria
def johnson_algorithm_cycles(dominance_s, dominance_w):
    graph_s = {}
    value_s = [[] for i in range(dominance_s.shape[0])]
    keys_s  = range(dominance_s.shape[0])
    for i in range(0, dominance_s.shape[0]):
        for j in range(0, dominance_s.shape[0]):
            if (dominance_s[i,j] == 1):
                value_s[i].append(j)
    for i in keys_s:
        graph_s[i] = value_s[i]  
    s1 = list(simple_cycles(graph_s))
    for k in range(0, len(s1)):   
        for j in range(0, len(s1[k]) -1):
            dominance_s[s1[k][j], s1[k][j+1]] = 0
            dominance_s[s1[k][j+1], s1[k][j]] = 0
    s2 = s1[:]
    for m in s1:
        for n in s1:
            if set(m).issubset(set(n)) and m != n:
                s2.remove(m)
                break
    for k in range(0, len(s2)):   
        for j in range(0, len(s2[k])):
            dominance_s[s2[k][j], :] = 0
        for i in range(0, dominance_s.shape[0]):
            count = 0
            for j in range(0, len(s2[k])):
                if (dominance_s[i, s2[k][j]] > 0):
                    count = count + 1
            if (count > 0):
                for j in range(0, len(s2[k])):
                    dominance_s[i, s2[k][j]] = 1
                
    graph_w = {}
    value_w = [[] for i in range(dominance_s.shape[0])]
    keys_w  = range(dominance_s.shape[0])
    for i in range(0, dominance_w.shape[0]):
        for j in range(0, dominance_w.shape[0]):
            if (dominance_w[i,j] == 1):
                value_w[i].append(j)
    for i in keys_w:
        graph_w[i] = value_w[i] 
    w1 = list(simple_cycles(graph_w))
    for k in range(0, len(w1)):   
        for j in range(0, len(w1[k]) -1):
            dominance_w[w1[k][j], w1[k][j+1]] = 0
            dominance_w[w1[k][j+1], w1[k][j]] = 0
    w2 = w1[:]
    for m in w1:
        for n in w1:
            if set(m).issubset(set(n)) and m != n:
                w2.remove(m)
                break
    for k in range(0, len(w2)):   
        for j in range(0, len(w2[k])):
            dominance_w[w2[k][j], :] = 0
        for i in range(0, dominance_w.shape[0]):
            count = 0
            for j in range(0, len(w2[k])):
                if (dominance_w[i, w2[k][j]] > 0):
                    count = count + 1
            if (count > 0):
                for j in range(0, len(w2[k])):
                    dominance_w[i, w2[k][j]] = 1  
    return dominance_s, dominance_w

# Function: Destilation
def ranking(dominance_s, dominance_w):
    dominance = np.clip(2*dominance_s + dominance_w, 0, 2)
    y    = list(range(1, dominance.shape[0] + 1)) 
    y    = ['a' + str(alt) for alt in y]
    rank = []
    while (len(y) > 0):
        d    = []
        u    = []
        b    = []
        a    = []
        for j in range (0, dominance.shape[1]):
            check_d = 0
            for i in range (0, dominance.shape[0]):
                if (dominance[i, j] == 2):
                    check_d = check_d + 1
            if (check_d == 0 and np.sum(dominance[:, j], axis = 0) != -dominance.shape[0]):
                d.append('a' + str(j + 1))
        idx  = []
        for k in range (0, len(d)):
            idx.append(int(d[k].replace('a','')) - 1)
        for m in range (0, len(d)):
            for n in range (0, len(d)):
                if (dominance[idx[m], idx[n]] == 1):
                    if (str('a' + str(idx[m] + 1)) not in u):
                        u.append('a' + str(idx[m] + 1))
                    if (str('a' + str(idx[n] + 1)) not in u):
                        u.append('a' + str(idx[n] + 1))
        idx = []
        for k in range (0, len(u)):
            idx.append(int(u[k].replace('a','')) - 1)
        for m in range (0, len(u)):
            check_b = 0
            for n in range (0, len(u)):
                if (dominance[idx[n], idx[m]] == 1):
                    check_b = check_b + 1
            if (check_b == 0):
                if (str('a' + str(idx[m] + 1)) not in b):
                    b.append('a' + str(idx[m] + 1))
        a = [item for item in d if item not in u]
        a = a + b
        rank.append(a)
        idx = []
        for k in range (0, len(a)):
            idx.append(int(a[k].replace('a','')) - 1)
        y = [item for item in y if item not in a]
        for j in range(0, len(idx)):
            dominance[:, idx[j]] = -1
            dominance[idx[j], :] = -1
        len(y)
    return rank

# Function: Pre-Order Matrix
def pre_order_matrix(rank_D, rank_A, number_of_alternatives = 7):
    alts   = list(range(1, number_of_alternatives + 1)) 
    alts   = ['a' + str(alt) for alt in alts]
    alts_D = [0]*number_of_alternatives
    alts_A = [0]*number_of_alternatives
    for i in range(0, number_of_alternatives):
        for j in range(0, len(rank_D)):
            if (alts[i] in rank_D[j]):
                alts_D[i] = j + 1
        for k in range(0, len(rank_A)):
            if (alts[i] in rank_A[k]):
                alts_A[i] = k + 1    
    po_string = np.empty((number_of_alternatives, number_of_alternatives), dtype = 'U25')
    po_string.fill('-')
    for i in range(0, number_of_alternatives):
        for j in range(0, number_of_alternatives): 
            if (i < j):
                if ( (alts_D[i] < alts_D[j] and alts_A[i] < alts_A[j]) or (alts_D[i] == alts_D[j] and alts_A[i] < alts_A[j]) or (alts_D[i] < alts_D[j] and alts_A[i] == alts_A[j]) ):
                    po_string[i,j] = 'P+'
                    po_string[j,i] = 'P-'
                if ( (alts_D[i] > alts_D[j] and alts_A[i] > alts_A[j]) or (alts_D[i] == alts_D[j] and alts_A[i] > alts_A[j]) or (alts_D[i] > alts_D[j] and alts_A[i] == alts_A[j]) ):
                    po_string[i,j] = 'P-'
                    po_string[j,i] = 'P+'
                if ( (alts_D[i] == alts_D[j] and alts_A[i] == alts_A[j]) ):
                    po_string[i,j] = 'I'
                    po_string[j,i] = 'I'
                if ( (alts_D[i] > alts_D[j] and alts_A[i] < alts_A[j]) or (alts_D[i] < alts_D[j] and alts_A[i] > alts_A[j])):
                    po_string[i,j] = 'R'
                    po_string[j,i] = 'R'
    return po_string  

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

# Function: ELECTRE II
def electre_ii(dataset, W, c_minus = 0.65, c_zero = 0.75, c_plus = 0.85, d_minus = 0.25, d_plus = 0.50, graph = False):
    alts   = list(range(1, dataset.shape[0] + 1)) 
    alts   = ['a' + str(alt) for alt in alts]
    alts_D = [0]*dataset.shape[0]
    alts_A = [0]*dataset.shape[0]
    concordance = concordance_matrix(dataset, W)
    discordance = discordance_matrix(dataset)
    dominance_s, dominance_w = dominance_matrix(concordance, discordance, c_minus = c_minus, c_zero = c_zero, c_plus = c_plus, d_minus = d_minus, d_plus = d_plus)
    dominance_s, dominance_w = johnson_algorithm_cycles(dominance_s, dominance_w)
    rank_A = ranking(dominance_s, dominance_w)
    rank_D = ranking(dominance_s.T, dominance_w.T)
    rank_D.reverse()
    rank_M = []
    for i in range(0, dataset.shape[0]):
        for j in range(0, len(rank_D)):
            if (alts[i] in rank_D[j]):
                alts_D[i] = j + 1
        for k in range(0, len(rank_A)):
            if (alts[i] in rank_A[k]):
                alts_A[i] = k + 1 
    for i in range(0, len(alts)):
        rank_M.append('a' + str(i+1) )
    rank_M.sort()
    rank_P = pre_order_matrix(rank_D, rank_A, number_of_alternatives = dataset.shape[0])
    if (graph == True):
        po_ranking(rank_P)
    return concordance, discordance, dominance_s, dominance_w, rank_D, rank_A, rank_M, rank_P

###############################################################################  
