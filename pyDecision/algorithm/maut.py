###############################################################################

# Code Contributor: Sabir Mohammedi Taieb - Université Abdelhamid Ibn Badis Mostaganem

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt

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

# Function: Marginal Utility Exp
def u_exp(x):
    y = (np.exp(x**2)-1)/1.72
    return y

# Function: Marginal Utility Step
def u_step(x, op):
    y = np.ceil(op*x)/op
    return y

# Function: Marginal Utility Log10
def u_log(x):
    y = np.log10(9*x+1)
    return y

# Function: Marginal Utility LN
def u_ln(x):
    y = np.log((np.exp(1)-1)*x+1)
    return y

# Function: Marginal Utility Quadratic
def u_quad(x):
    y = (2*x-1)**2
    return y

###############################################################################

# Function: MAUT
def maut_method(dataset, weights, criterion_type, utility_functions, step_size = 1, graph = True, verbose = True):
    X = np.copy(dataset)/1.0
    for j in range(0, X.shape[1]):
        if (criterion_type[j] == 'max'):
            X[:, j] = (X[:,j] - np.min(X[:, j]))/(np.max(X[:, j]) - np.min(X[:, j]) + 0.0000000000000001)
        else:
            X[:, j] = 1 + (np.min(X[:, j])- X[:, j])/(np.max(X[:, j]) - np.min(X[: ,j])+ 0.0000000000000001)
    for i in range(0, X.shape[1]):
        if (utility_functions[i] == 'exp'):
            ArrExp  = np.vectorize(u_exp)
            X[:, i] = ArrExp(X[:, i])
        elif (utility_functions[i] == 'step'):
            ArrStep = np.vectorize(u_step)
            X[:, i] = ArrStep(X[:, i], step_size)
        elif (utility_functions[i] == 'quad'):
            ArrQuad = np.vectorize(u_quad)
            X[:, i] = ArrQuad(X[:, i])
        elif (utility_functions[i] == 'log'):
            ArrLog  = np.vectorize(u_log)
            X[:, i] = ArrLog(X[:, i])
        elif (utility_functions[i] == 'ln'):
            ArrLn   = np.vectorize(u_ln)
            X[:, i] = ArrLn(X[:, i])
    for i in range(0, X.shape[1]):
        X[:, i] = X[:, i]*weights[i]
    Y    = np.sum(X, axis = 1)
    flow = np.copy(Y)
    flow = np.reshape(flow, (Y.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, Y.shape[0]+1)), axis = 1)
    if (verbose == True):
        for i in range(0, flow.shape[0]):
            print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3)))
    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return flow

###############################################################################
