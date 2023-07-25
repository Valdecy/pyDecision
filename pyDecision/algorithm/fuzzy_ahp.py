###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Fuzzy AHP
def fuzzy_ahp_method(dataset):
    row_sum = []
    s_row   = []
    f_w     = []
    d_w     = []
    inc_rat = np.array([0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59])
    X       = [(item[0] + 4*item[1] + item[2])/6 for i in range(0, len(dataset)) for item in dataset[i] ]
    X       = np.asarray(X)
    X       = np.reshape(X, (len(dataset), len(dataset)))
    for i in range(0, len(dataset)):
        a, b, c = 1, 1, 1
        for j in range(0, len(dataset[i])):
            d, e, f = dataset[i][j]
            a, b, c = a*d, b*e, c*f
        row_sum.append( (a, b, c) )
    L, M, U = 0, 0, 0
    for i in range(0, len(row_sum)):
        a, b, c = row_sum[i]
        a, b, c = a**(1/len(dataset)), b**(1/len(dataset)), c**(1/len(dataset))
        s_row.append( ( a, b, c ) )
        L = L + a
        M = M + b
        U = U + c
    for i in range(0, len(s_row)):
        a, b, c = s_row[i]
        a, b, c = a*(U**-1), b*(M**-1), c*(L**-1)
        f_w.append( ( a, b, c ) )
        d_w.append( (a + b + c)/3 )
    n_w      = [item/sum(d_w) for item in d_w]
    vector   = np.sum(X*n_w, axis = 1)/n_w
    lamb_max = np.mean(vector)
    cons_ind = (lamb_max - X.shape[1])/(X.shape[1] - 1)
    rc       = cons_ind/inc_rat[X.shape[1]]
    return f_w, d_w, n_w, rc

###############################################################################