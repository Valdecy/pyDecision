###############################################################################

# Required Libraries
import numpy as np
import re

###############################################################################

# Function: ROC (Rank Ordered Centroid)
def roc_method(criteria_rank):
    
    ################################################
    
    def extract_number(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None
    
    ################################################
    
    x = np.zeros(len(criteria_rank))
    for i in range(0, x.shape[0]):
        for j in range(i, x.shape[0]):
            x[i] = x[i] + 1/(j+1)
    x   = x/len(criteria_rank)
    x   = x/np.sum(x)
    idx = sorted(range(0, len(criteria_rank)), key = lambda x: extract_number(criteria_rank[x]))
    x   = x[idx]
    return x

###############################################################################
