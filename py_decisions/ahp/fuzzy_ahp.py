###############################################################################

# Required Libraries

###############################################################################

# Function: Fuzzy AHP
def fuzzy_ahp_method(dataset):
    row_sum = []
    s_row   = []
    f_w     = []
    d_w     = []
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
    n_w = [item/sum(d_w) for item in d_w]  
    return f_w, d_w, n_w

###############################################################################
    
       
    
