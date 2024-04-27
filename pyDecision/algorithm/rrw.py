###############################################################################

# Required Libraries
import numpy as np
import re

###############################################################################

# Function: RRW (Rank Reciprocal Weighting)
def rrw_method(criteria_rank):
    
    ################################################
    
    def extract_number(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None
    
    ################################################
    
    S = 0
    x = np.zeros(len(criteria_rank))
    for i in range(0, x.shape[0]):
        S = S + 1/(i+1) 
    for i in range(0, x.shape[0]):
        x[i] = 1 / ( (i+1) * S)
    x   = x/np.sum(x)
    idx = sorted(range(0, len(criteria_rank)), key = lambda x: extract_number(criteria_rank[x]))
    x   = x[idx]
    return x

###############################################################################