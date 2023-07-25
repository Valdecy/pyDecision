###############################################################################

# Required Libraries
import math
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

# Function: Global Concordance Matrix
def global_concordance_matrix(dataset, Q, P, W):
    global_concordance = np.zeros((dataset.shape[0], dataset.shape[0]))
    for k in range(0, dataset.shape[1]):
        for i in range(0, global_concordance.shape[0]):
            for j in range(0, global_concordance.shape[1]):
                if (dataset[j,k] - dataset[i,k] >= P[k] ):
                    global_concordance[i,j] = global_concordance[i,j] + W[k]*0
                elif (dataset[j,k] - dataset[i,k] < Q[k] ):
                    global_concordance[i,j] = global_concordance[i,j] + W[k]*1
                elif (dataset[j,k] - dataset[i,k] >= Q[k] and dataset[j,k] - dataset[i,k] < P[k]):
                    global_concordance[i,j] = global_concordance[i,j] + W[k]*((P[k] - dataset[j,k] + dataset[i,k])/(P[k] - Q[k]))
    if (np.sum(W) != 0):
        global_concordance = global_concordance/np.sum(W)
    return global_concordance

# Function: Discordance Matrix
def discordance_matrix(dataset, global_concordance, Q, V, W, lambda_value = 0.5):
    discordance = np.zeros((global_concordance.shape[0], global_concordance.shape[1]))
    w_sum = 1
    for i in range(0, discordance.shape[0]):
        for j in range(0, discordance.shape[1]):
            for k in range(0, dataset.shape[1]):
                if ((W[k]/np.sum(W)) != 0):
                    w_sum = (W[k]/np.sum(W))
                if ( (dataset[j,k] - dataset[i,k]) >= ( V[k] - Q[k]*( (1 - global_concordance[i,j] - (w_sum)) / (1 - lambda_value - (w_sum) ) ) ) ):
                    discordance[i,j] = 1
    return discordance

# Function: Credibility Matrix
def credibility_matrix (global_concordance, discordance, lambda_value = 0.5):
    credibility = np.zeros((global_concordance.shape[0], global_concordance.shape[1]))
    for i in range(0, credibility.shape[0]):
        for j in range(0, credibility.shape[1]):
            if (global_concordance[i,j] >= lambda_value and discordance[i,j] == 0 and i != j):
                credibility[i,j] = 1
    return credibility

# Function: Find  and Remove Cycles
def johnson_algorithm_cycles(credibility):
    graph = {}
    value = [[] for i in range(credibility.shape[0])]
    keys   = range(credibility.shape[0])
    for i in range(0, credibility.shape[0]):
        for j in range(0, credibility.shape[0]):
            if (credibility[i,j] == 1):
                value[i].append(j)
    for i in keys:
            graph[i] = value[i]  
    s1 = list(simple_cycles(graph))
    for k in range(0, len(s1)):   
        for j in range(0, len(s1[k]) -1):
            credibility[s1[k][j], s1[k][j+1]] = 0
            credibility[s1[k][j+1], s1[k][j]] = 0
    return credibility

# Function: Electre I_s
def electre_i_s(dataset, Q, P, V, W, graph = True, lambda_value = 0.5):
    kernel    = []
    dominated = []
    global_concordance = global_concordance_matrix(dataset, Q = Q, P = P, W = W)
    discordance = discordance_matrix(dataset, global_concordance, Q = Q, V = V, W = W, lambda_value = lambda_value)
    credibility = credibility_matrix (global_concordance, discordance, lambda_value = lambda_value)
    credibility = johnson_algorithm_cycles(credibility)
    row_sum     = np.sum(credibility , axis = 0)
    kernel      = np.where(row_sum == 0)[0].tolist()            
    for j in range(0, credibility .shape[1]):
        for i in range(0, len(kernel)):
            if (credibility[kernel[i], j] == 1):
                if (j not in dominated):
                    dominated.append(j) 
    limit = len(kernel)
    for j in range(0, credibility.shape[1]):
        for i in range(0, limit):
            if (credibility[kernel[i], j] == 0 and np.sum(credibility[:,j], axis = 0) > 0):
                if (j not in dominated and j not in kernel):
                    kernel.append(j)
    kernel    = ['a' + str(alt + 1) for alt in kernel]
    dominated = ['a' + str(alt + 1) for alt in dominated]
    if (graph == True):
        for i in range(0, credibility.shape[0]):
            radius = 1
            node_x = radius*math.cos(math.pi * 2 * i / credibility.shape[0])
            node_y = radius*math.sin(math.pi * 2 * i / credibility.shape[0])
            if ('a' + str(i+1) in kernel):
                plt.text(node_x,  node_y, 'a' + str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
            else:
              plt.text(node_x,  node_y, 'a' + str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))  
        for i in range(0, credibility.shape[0]):
            for j in range(0, credibility.shape[1]):
                node_xi = radius*math.cos(math.pi * 2 * i / credibility.shape[0])
                node_yi = radius*math.sin(math.pi * 2 * i / credibility.shape[0])
                node_xj = radius*math.cos(math.pi * 2 * j / credibility.shape[0])
                node_yj = radius*math.sin(math.pi * 2 * j / credibility.shape[0])
                if (credibility[i, j] == 1):  
                    if ('a' + str(i+1) in kernel):
                        plt.arrow(node_xi, node_yi, node_xj - node_xi, node_yj - node_yi, head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
                    else:
                        plt.arrow(node_xi, node_yi, node_xj - node_xi, node_yj - node_yi, head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'red', linewidth = 0.9, length_includes_head = True)
        axes = plt.gca()
        axes.set_xlim([-radius, radius])
        axes.set_ylim([-radius, radius])
        plt.axis('off')
        plt.show() 
    return global_concordance, discordance, kernel, credibility, dominated

###############################################################################
