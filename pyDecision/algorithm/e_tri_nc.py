###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Matrices 
def _tri_nc_r_matrices(performance_matrix, B_list, W, Q, P, V = None):
    A    = np.array(performance_matrix, dtype = float)
    m, n = A.shape
    W    = np.array(W, dtype = float)
    W    = W / W.sum()
    Q    = np.array(Q, dtype = float)
    P    = np.array(P, dtype = float)

    if V is not None and len(V) > 0:
        V = np.array(V, dtype = float)
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

    if isinstance(B_list, np.ndarray):
        if B_list.ndim != 2:
            raise ValueError("When using numpy array for C/B, it must be 2D.")
        subsets = [B_list[h:h+1, :] for h in range(B_list.shape[0])]
    else:
        subsets = [np.atleast_2d(np.array(Bh, dtype=float)) for Bh in B_list]

    hB   = len(subsets)
    r_ab = np.zeros((m, hB))
    r_ba = np.zeros((m, hB))

    for a_idx in range(0, m):
        a_vec = A[a_idx]
        for h, Bh in enumerate(subsets):
            best_ab = 0.0
            best_ba = 0.0
            for b_vec in Bh:
                cab = c(a_vec, b_vec)
                cba = c(b_vec, a_vec)
                if cab > best_ab:
                    best_ab = cab
                if cba > best_ba:
                    best_ba = cba
            r_ab[a_idx, h] = best_ab
            r_ba[a_idx, h] = best_ba
    return r_ab, r_ba

###############################################################################

# Function: Electre Tri - nC
def electre_tri_nc(performance_matrix, W = [], Q = [], P = [], V = [],  C = None, cut_level = 0.6, verbose = True, tol = 1e-9):
    A           = np.array(performance_matrix, dtype = float)
    m, n        = A.shape
    num_subsets = len(C)  
    if num_subsets < 3:
        raise ValueError("There must be at least 3 subsets of reference actions (B0, B1..Bq, Bq+1).")
    q          = num_subsets - 2
    r_ab, r_ba = _tri_nc_r_matrices(A, C, W, Q, P, V)

    def q_select(a_idx, h_idx):
        return min(r_ab[a_idx, h_idx], r_ba[a_idx, h_idx])

    lowest_classes  = []
    highest_classes = []

    for a in range(m):
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