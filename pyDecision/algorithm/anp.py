###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Normalization
def normalize_matrix(W, zero_column_policy = "keep"):
    W                       = np.array(W, dtype = float)
    col_sums                = W.sum(axis = 0)
    W_norm                  = W.copy()
    n_rows, _               = W.shape
    nonzero_mask            = col_sums != 0
    zero_mask               = ~nonzero_mask
    W_norm[:, nonzero_mask] = W_norm[:, nonzero_mask] / col_sums[nonzero_mask]
    if np.any(zero_mask):
        if zero_column_policy == "uniform":
            W_norm[:, zero_mask] = 1.0 / n_rows
    return W_norm

###############################################################################

# Function: ANP
def anp_method(W, max_iter = 100, tol = 1e-12, cesaro = True):
    W          = np.array(W, dtype = float)
    W          = normalize_matrix(W, zero_column_policy = "keep")
    rows, cols = W.shape
    if rows != cols:
        raise ValueError(f"Input matrix must be square. Got {rows}x{cols}.")
    if (cesaro == True):
        W_power        = W.copy()          
        cumulative_sum = W.copy()   
        current_mean   = W.copy()
        for k in range(2, max_iter + 1):
            W_power        = W_power @ W
            cumulative_sum = cumulative_sum + W_power
            new_mean       = cumulative_sum / k
            if np.allclose(new_mean, current_mean, atol = tol):
                L                  = new_mean.copy()
                L[np.abs(L) < tol] = 0.0
                print(f"Cesaro mean converged at iteration k = {k}")
                return L
            current_mean = new_mean
        L                  = current_mean.copy()
        L[np.abs(L) < tol] = 0.0
    else:
        M_prev = W.copy()
        L      = W.copy()
        for k in range(0, max_iter):
            L = L @ W
            if np.allclose(L, M_prev, atol = tol):
                return L
            M_prev = L
    return L

###############################################################################
