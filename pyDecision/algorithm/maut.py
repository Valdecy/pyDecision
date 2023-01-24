import numpy as np
import matplotlib.pyplot as plt
import math

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

#Marginal Utility Functions
def Uexp(x):
    return (math.exp(x**2)-1)/1.72

# Custom made Marginal Utility functions based on Step, Log and quadratic functions which return values in [0,1]
# op: number of evaluation options for the criterion j; 
# When using step function:
# Add a list containing number of options for each criterion using step function otherwise put 0 as a placeholder
def Ustep(x,op):
    return ceil(op*x)/op

# Logarithmic base 10
def Ulog(x):
    return math.log(9*x+1,10)

# Natural Logarithm (ln) base e
def Uln(x):
    return math.log((math.exp(1)-1)*x+1)

# Quadratic
def Uquad(x):
    return (2*x-1)**2

def maut(dataset, criterion_type, utility_functions, weights, options_number=1, graph=True):
    X = np.copy(dataset)
    # normalization
    for i in range(0,X.shape[1]):
        if(criterion_type[i]=='max'):
            X[:,i]=(X[:,i] - np.min(X[:,i]))/(np.max(X[:,i])-np.min(X[:,i]))
        else:
            X[:,i]=1+ (np.min(X[:,i])- X[:,i])/(np.max(X[:,i])-np.min(X[:,i]))

    # Apply selected Maginal Utility Function on criterion
    for i in range(0,X.shape[1]):
        if(utility_functions[i]=='exp'):
            ArrExp=np.vectorize(Uexp)
            X[:,i]=ArrExp(X[:,i])
        elif(utility_functions[i]=='step'):
            ArrStep=np.vectorize(Ustep)
            X[:,i]=ArrStep(X[:,i],options_number)
        elif(utility_functions[i]=='quad'):
            ArrQuad=np.vectorize(Uquad)
            X[:,i]=ArrQuad(X[:,i])
        elif(utility_functions[i]=='log'):
            ArrLog=np.vectorize(Ulog)
            X[:,i]=ArrLog(X[:,i])
        elif(utility_functions[i]=='ln'):
            ArrLn=np.vectorize(Uln)
            X[:,i]=ArrLn(X[:,i])


    # Multiplying by Weights
    for i in range(0,X.shape[1]):
        X[:,i] = X[:,i]*weights[i]

    
    # Final Additive Utility Score
    Y = np.sum(X,axis=1)

    # Printing Scores of alternatives
    flow = np.copy(Y)
    flow = np.reshape(flow, (Y.shape[0], 1))
    flow = np.insert(flow, 0, list(range(1, Y.shape[0]+1)), axis = 1)
    for i in range(0, flow.shape[0]):
        print('a' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3)))

    if (graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    print(flow)
    return flow