###############################################################################

# Required Libraries
import numpy as np
import re

###############################################################################

# Function: RSW (Rank Summed Weight)
def rsw_method(criteria_rank):
    
    ################################################
    
    def extract_number(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None
    
    ################################################
    
    N = len(criteria_rank)
    x = np.zeros(N)
    for i in range(0, x.shape[0]):
        x[i] = ( 2 * (N - (i+1) + 1) ) / ( N * (N  + 1) )
    x   = x/np.sum(x)
    idx = sorted(range(0, len(criteria_rank)), key = lambda x: extract_number(criteria_rank[x]))
    x   = x[idx]
    return x

###############################################################################
