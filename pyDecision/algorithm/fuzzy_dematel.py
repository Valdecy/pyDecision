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
    m_a = np.ones ((len(dataset), len(dataset))) # min
    m_c = np.zeros((len(dataset), len(dataset))) # max
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset)):
            a, b, c  = dataset[i][j]
            X_a[i,j] = a
            X_b[i,j] = b
            X_c[i,j] = c
            if (a < m_a[i,j]):
                m_a[i,j] = a
            if (c > m_c[i,j]):
                m_c[i,j] = c
    m_a       = np.min(m_a, axis = 1)
    m_c       = np.max(m_c, axis = 1)
    X_a       = (X_a - m_a) / (m_c - m_a)
    X_b       = (X_b - m_a) / (m_c - m_a)
    X_c       = (X_c - m_a) / (m_c - m_a)
    L         = X_b / (1 + X_b - X_a)
    R         = X_c / (1 + X_c - X_b)
    W         = (L * (1 - L) + R * R) / (1 - L + R)
    Z         = m_a  + W * (m_c - m_a)
    X         = Z/np.max(np.sum(Z, axis = 1))
    Y         = np.linalg.inv(np.identity(X.shape[0]) - X) 
    T         = np.matmul(X, Y)
    D         = np.sum(T, axis = 1)
    R         = np.sum(T, axis = 0)
    D_plus_R  = D + R
    D_minus_R = D - R 
    weights   = (D_plus_R - D_minus_R)/(np.sum(D_plus_R + D_minus_R))
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
