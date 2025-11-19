###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: ANP
def anp_method(W, max_iter = 100, tol = 1e-12):
    M_prev = W.copy()
    M      = W.copy()
    for k in range(0, max_iter):
        M = M @ W
        if np.allclose(M, M_prev, atol = tol):
            return M
        M_prev = M
    return M

###############################################################################

