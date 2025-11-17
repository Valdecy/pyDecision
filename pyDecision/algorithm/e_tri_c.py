###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Matrices
def _tri_c_r_matrices(performance_matrix, C, W, Q, P, V = None):
    A    = np.array(performance_matrix, dtype = float)
    B    = np.array(C, dtype = float)
    m, n = A.shape
    hB   = B.shape[0]
    W    = np.array(W, dtype = float)
    W    = W / W.sum()  
    Q    = np.array(Q, dtype = float)
    P    = np.array(P, dtype = float)
    if V is not None and len(V) > 0:
        V        = np.array(V, dtype = float)
        use_veto = True
    else:
        V = None
        use_veto = False

    def c(a, b):
        d = a - b  
        if use_veto:
            if np.any(-d >= V):
                return 0.0
        C_I     = np.abs(d) <= Q               
        C_P     = d > P                        
        C_Q     = (d > Q) & (d <= P)           
        C_Q_rev = (-d > Q) & (-d <= P)         
        s       = W[C_I | C_Q | C_P].sum()
        u       = (d + P) / (P - Q)
        s       = s + (W[C_Q_rev] * u[C_Q_rev]).sum()
        return float(s)

    r_ab = np.zeros((m, hB))  
    r_ba = np.zeros((m, hB))

    for a_idx in range(0, m):
        for h in range(0, hB):
            a_vec          = A[a_idx]
            b_vec          = B[h]
            r_ab[a_idx, h] = c(a_vec, b_vec)
            r_ba[a_idx, h] = c(b_vec, a_vec)
    return r_ab, r_ba

###############################################################################

# Function: Electre Tri-C
def electre_tri_c(performance_matrix, W = [], Q = [], P = [], V = [], C = [], cut_level = 0.6, verbose = True,  tol = 1e-9):
    A          = np.array(performance_matrix, dtype = float)
    B          = np.array(C, dtype = float)
    m, n       = A.shape
    n_profiles = B.shape[0]     
    q          = n_profiles - 2          
    r_ab, r_ba = _tri_c_r_matrices(A, B, W, Q, P, V)

    def q_select(a_idx, h_idx):
        return min(r_ab[a_idx, h_idx], r_ba[a_idx, h_idx])

    lowest_classes  = []
    highest_classes = []

    for a in range(0, m):
        t = None
        for h in range(q + 1, -1, -1):  
            if r_ab[a, h] >= cut_level - tol:
                t = h
                break
        if t is None:
            t = 0
        if t == q:
            desc_cat = q
        elif 0 < t < q:
            if q_select(a, t) > q_select(a, t + 1) + tol:
                desc_cat = t
            else:
                desc_cat = t + 1
        elif t == 0:
            desc_cat = 1
        else:
            desc_cat = q

        k_idx = None
        for h in range(0, q + 2):  
            if r_ba[a, h] >= cut_level - tol:
                k_idx = h
                break
        if k_idx is None:
            k_idx = q + 1
        if k_idx == 1:
            asc_cat = 1
        elif 1 < k_idx < (q + 1):
            if q_select(a, k_idx) > q_select(a, k_idx - 1) + tol:
                asc_cat = k_idx
            else:
                asc_cat = k_idx - 1
        elif k_idx == (q + 1):
            asc_cat = q
        else:  
            asc_cat = 1

        low  = min(desc_cat, asc_cat)
        high = max(desc_cat, asc_cat)
        lowest_classes.append(low)
        highest_classes.append(high)
        if verbose:
            print(f'a{a+1}: [C{low}, C{high}]')
    return lowest_classes, highest_classes

###############################################################################