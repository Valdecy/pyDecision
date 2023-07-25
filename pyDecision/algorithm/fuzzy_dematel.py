###############################################################################

# Required Libraries

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# Function: Fuzzy DEMATEL
def fuzzy_dematel_method(dataset, size_x = 10, size_y = 10):  
    X_a = np.zeros((len(dataset), len(dataset)))
    X_b = np.zeros((len(dataset), len(dataset)))
    X_c = np.zeros((len(dataset), len(dataset)))
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset)):
            a, b, c  = dataset[i][j]
            X_a[i,j] = a
            X_b[i,j] = b
            X_c[i,j] = c
    X_a = X_a / np.max(np.sum(X_a, axis = 1))
    X_b = X_b / np.max(np.sum(X_b, axis = 1))
    X_c = X_c / np.max(np.sum(X_c, axis = 1))
    Y_a = np.linalg.inv(np.identity(len(dataset)) - X_a)
    Y_b = np.linalg.inv(np.identity(len(dataset)) - X_b)
    Y_c = np.linalg.inv(np.identity(len(dataset)) - X_c)
    T_a = np.matmul (X_a, Y_a)
    T_b = np.matmul (X_b, Y_b)
    T_c = np.matmul (X_c, Y_c)
    D_a = np.sum(T_a, axis = 1)
    D_b = np.sum(T_b, axis = 1)
    D_c = np.sum(T_c, axis = 1)
    R_a = np.sum(T_a, axis = 0)
    R_b = np.sum(T_b, axis = 0)
    R_c = np.sum(T_c, axis = 0)
    D_plus_R  = (D_a + D_b + D_c)/3 + (R_a + R_b + R_c)/3    
    D_minus_R = (D_a + D_b + D_c)/3 - (R_a + R_b + R_c)/3 
    weights   = D_plus_R/np.sum(D_plus_R)
    print('QUADRANT I has the Most Important Criteria (Prominence: High, Relation: High)') 
    print('QUADRANT II has Important Criteira that can be Improved by Other Criteria (Prominence: Low, Relation: High)') 
    print('QUADRANT III has Criteria that are not Important (Prominence: Low, Relation: Low)')
    print('QUADRANT IV has Important Criteria that cannot be Improved by Other Criteria (Prominence: High, Relation: Low)')
    print('')
    plt.figure(figsize = [size_x, size_y])
    plt.style.use('ggplot')
    for i in range(0, len(dataset)):
        if (D_minus_R[i] >= 0 and D_plus_R[i] >= np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 1.0, 0.7),)) 
            print('g'+str(i+1)+': Quadrant I')
        elif (D_minus_R[i] >= 0 and D_plus_R[i] < np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 1.0, 0.7),))
            print('g'+str(i+1)+': Quadrant II')
        elif (D_minus_R[i] < 0 and D_plus_R[i] < np.mean(D_plus_R)):
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.7, 0.7),)) 
            print('g'+str(i+1)+': Quadrant III')
        else:
            plt.text(D_plus_R[i],  D_minus_R[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 0.7, 1.0),)) 
            print('g'+str(i+1)+': Quadrant IV')
    axes = plt.gca()
    xmin = np.amin(D_plus_R)
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(D_plus_R)
    if (xmax < 0):
        xmax = 0
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(D_minus_R)
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(D_minus_R)
    if (ymax < 0):
        ymax = 0
    axes.set_ylim([ymin-1, ymax+1]) 
    plt.axvline(x = np.mean(D_plus_R), linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.axhline(y = 0, linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.xlabel('Prominence (D + R)')
    plt.ylabel('Relation (D - R)')
    plt.show()
    return D_plus_R, D_minus_R, weights

###############################################################################
