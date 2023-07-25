###############################################################################

# Required Libraries

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# Function: WINGS (Weighted Influence Non-linear Gauge System)
def wings_method(dataset, size_x = 10, size_y = 10): 
    D = dataset
    C = D/np.sum(D)
    I = np.eye(dataset.shape[0])
    T = np.dot(C, np.linalg.inv( I - C ))
    c_i = np.sum(T, axis = 0)
    r_i = np.sum(T, axis = 1)
    r_plus_c  = r_i + c_i 
    r_minus_c = r_i - c_i 
    weights   = r_plus_c/np.sum(r_plus_c)
    plt.figure(figsize = [size_x, size_y])
    plt.style.use('ggplot')
    for i in range(0, dataset.shape[0]):
        if (r_minus_c[i] >= 0 and  r_plus_c[i] >= np.mean(r_plus_c)):
            plt.text(r_plus_c[i],  r_minus_c[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 1.0, 0.7),)) 
            print('g'+str(i+1)+': Quadrant I')
        elif (r_minus_c[i] >= 0 and r_plus_c[i] < np.mean(r_plus_c)):
            plt.text(r_plus_c[i],  r_minus_c[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 1.0, 0.7),))
            print('g'+str(i+1)+': Quadrant II')
        elif (r_minus_c[i] < 0 and r_plus_c[i] < np.mean(r_plus_c)):
            plt.text(r_plus_c[i],  r_minus_c[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.7, 0.7),)) 
            print('g'+str(i+1)+': Quadrant III')
        else:
            plt.text(r_plus_c[i], r_minus_c[i], 'g'+str(i+1), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.7, 0.7, 1.0),)) 
            print('g'+str(i+1)+': Quadrant IV')
    axes = plt.gca()
    xmin = np.amin(r_plus_c)
    if (xmin > 0):
        xmin = 0
    xmax = np.amax(r_plus_c)
    if (xmax < 0):
        xmax = 0
    axes.set_xlim([xmin-1, xmax+1])
    ymin = np.amin(r_minus_c)
    if (ymin > 0):
        ymin = 0
    ymax = np.amax(r_minus_c)
    if (ymax < 0):
        ymax = 0
    axes.set_ylim([ymin-1, ymax+1]) 
    plt.axvline(x = np.mean(r_plus_c), linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.axhline(y = 0, linewidth = 0.9, color = 'r', linestyle = 'dotted')
    plt.xlabel('(R + C)')
    plt.ylabel('(R - C)')
    plt.show()
    return r_plus_c, r_minus_c, weights

###############################################################################