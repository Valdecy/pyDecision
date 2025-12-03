###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Credibility
def credibility(x, y, w, q, p, v):
    x                = np.asarray(x, dtype = float)
    y                = np.asarray(y, dtype = float)
    w                = np.asarray(w, dtype = float)
    q                = np.asarray(q, dtype = float)
    p                = np.asarray(p, dtype = float)
    v                = np.asarray(v, dtype = float)
    d                = y - x
    c                = np.zeros_like(d, dtype = float)
    mask_full        = d <= q
    c[mask_full]     = 1.0
    mask_int         = (~mask_full) & (d < p)
    c[mask_int]      = (p[mask_int] - d[mask_int]) / (p[mask_int] - q[mask_int])
    C                = (w * c).sum() / w.sum()
    D                = np.zeros_like(d, dtype = float)
    mask_full_dis    = d >= v
    D[mask_full_dis] = 1.0
    mask_int_dis     = (d > p) & (~mask_full_dis)
    D[mask_int_dis]  = (d[mask_int_dis] - p[mask_int_dis]) / (v[mask_int_dis] - p[mask_int_dis])
    sigma            = C
    if C < 1.0:
        for Dj in D:
            if Dj > C:
                sigma *= (1 - Dj) / (1 - C)
    return float(sigma)

# Function: Profiles
def relations_x_Bk(x, Bk, w, q, p, v, lam):
    Bk     = np.asarray(Bk, dtype = float)
    n_prof = Bk.shape[0]
    if n_prof == 0:
        return False, False
    
    sig_xb = np.zeros(n_prof)
    sig_bx = np.zeros(n_prof)
    for j in range(0, n_prof):
        sig_xb[j] = credibility(x, Bk[j], w, q, p, v)
        sig_bx[j] = credibility(Bk[j], x, w, q, p, v)
    
    S_xb  = sig_xb >= lam   
    S_bx  = sig_bx >= lam  
    xP_b  = S_xb & (~S_bx)
    bP_x  = S_bx & (~S_xb)
    xS_Bk = S_xb.any() and (not bP_x.any())
    BkP_x = bP_x.any() and (not xP_b.any())
    return xS_Bk, BkP_x

###############################################################################

# Function: Electre Tri - nB
def electre_tri_nb(perf_matrix, B_sets, w, q, p, v, lam = 0.7):
    X        = np.asarray(perf_matrix, dtype = float)
    n_alt, m = X.shape
    M        = len(B_sets) - 1  
    xS_Bk    = np.zeros((n_alt, M+1), dtype = bool)   
    BkP_x    = np.zeros((n_alt, M+1), dtype = bool)   
    for i in range(0, n_alt):
        x = X[i]
        for k in range(0, M+1):
            s, p_rel    = relations_x_Bk(x, B_sets[k], w, q, p, v, lam)
            xS_Bk[i, k] = s
            BkP_x[i, k] = p_rel

    C_pc = np.ones(n_alt, dtype = int) 
    
    for i in range(0, n_alt):
        for k in range(M, 0, -1):  
            k_minus = k - 1
            cond1   = xS_Bk[i, k_minus]
            cond2   = True
            if k_minus > 0:
                cond2 = not BkP_x[i, :k_minus].any()
            if cond1 and cond2:
                C_pc[i] = k
                break
    C_pd = np.ones(n_alt, dtype = int)
    for i in range(0, n_alt):
        assigned = False
        for k in range(1, M+1):  
            cond1 = BkP_x[i, k]
            cond2 = True
            if k < M:
                cond2 = not xS_Bk[i, k+1:].any()
            if cond1 and cond2:
                C_pd[i]  = k
                assigned = True
                break
        if not assigned:
            C_pd[i] = 1 
    return C_pc, C_pd

###############################################################################