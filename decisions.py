###############################################################################

# Required Libraries
import numpy as np

# ELECTRE
from py_decisions.electre.e_i       import electre_i
from py_decisions.electre.e_i_s     import electre_i_s
from py_decisions.electre.e_i_v     import electre_i_v
from py_decisions.electre.e_ii      import electre_ii
from py_decisions.electre.e_iii     import electre_iii
from py_decisions.electre.e_iv      import electre_iv
from py_decisions.electre.e_tri_b   import electre_tri_b

# PROMETHEE
from py_decisions.promethee.p_i     import promethee_i
from py_decisions.promethee.p_ii    import promethee_ii
from py_decisions.promethee.p_iii   import promethee_iii
from py_decisions.promethee.p_iv    import promethee_iv
from py_decisions.promethee.p_v     import promethee_v
from py_decisions.promethee.p_vi    import promethee_vi
from py_decisions.promethee.p_xgaia import promethee_gaia

##############################################################################

# ELECTRE I
    
# Parameters
c_hat = 1.00
d_hat = 0.40

W = [0.0780, 0.1180, 0.1570, 0.3140, 0.2350, 0.0390, 0.0590]

# Dataset
dataset = np.array([
                [1, 2, 1, 5, 2, 2, 4],   #a1
                [3, 5, 3, 5, 3, 3, 3],   #a2
                [3, 5, 3, 5, 3, 2, 2],   #a3
                [1, 2, 2, 5, 1, 1, 1],   #a4
                [1, 1, 3, 5, 4, 1, 5]    #a5
                ])

# Dataset
dataset = np.array([
                [10,  8, 10],   #a1
                [10, 10, 10],   #a2
                [ 1,  1,  1],   #a3
                [ 7,  7,  5],   #a4
                [ 7,  7,  7]    #a5
                ])

concordance, discordance, dominance, kernel, dominated = electre_i(dataset, W = W, remove_cycles = True, c_hat = 0.75, d_hat = 0.50, graph = True)

##############################################################################

# ELECTRE I_s
    
# Parameters
lambda_value = 0.7

Q = [2000,   2,   1,   1,   1,  50, 0.1]
P = [3000,   5,   2,   3,   2,  82, 0.2]
V = [3500,   7,   3,   5,   6,  90, 0.5]
W = [0.3 , 0.1, 0.3, 0.1, 0.2, 0.2, 0.1]

# Dataset
dataset = np.array([
                [16000, 201, 8, 40, 5, 378, 31.3],   #a1
                [18000, 199, 8, 35, 5, 474, 33.0],   #a2
                [16000, 195, 8, 36, 1, 480, 33.9],   #a3
                [18000, 199, 8, 35, 5, 430, 33.1],   #a4
                [17000, 191, 8, 34, 1, 430, 34.4],   #a5
                [17000, 199, 8, 35, 4, 494, 32.0],   #a6
                [15000, 194, 8, 37, 3, 452, 33.8],   #a7
                [18000, 200, 8, 36, 6, 475, 33.8],   #a8
                [17000, 209, 7, 37, 3, 440, 30.9]    #a9
                ])

# Call Electre
global_concordance, discordance, kernel, credibility, dominated = electre_i_s(dataset, Q = Q, P = P, V = V, W = W, graph = True, lambda_value = lambda_value)

##############################################################################

# ELECTRE I_v
    
# Parameters
c_hat = 0.50

V = [2, 2, 2, 2]
W = [7, 3, 5, 6]

# Dataset
dataset = np.array([
                [15,  9, 6, 10],   #a1
                [10,  5, 7,  8],   #a2
                [22, 12, 1, 14],   #a3
                [31, 10, 6, 18],   #a4
                [ 8,  9, 0,  9]    #a5
                ])

# Call Electre
concordance, discordance, dominance, kernel, dominated = electre_i_v(dataset, V = V, W = W, remove_cycles = True, c_hat = c_hat, graph = True)

##############################################################################

# ELECTRE II
    
# Parameters
c_minus = 0.65
c_zero  = 0.75
c_plus  = 0.85

d_minus = 0.25
d_plus  = 0.50

W = [0.0780, 0.1180, 0.1570, 0.3140, 0.2350, 0.0390, 0.0590]

# Dataset
dataset = np.array([
                [1, 2, 1, 5, 2, 2, 4],   #a1
                [3, 5, 3, 5, 3, 3, 3],   #a2
                [3, 5, 3, 5, 3, 2, 2],   #a3
                [1, 2, 2, 5, 1, 1, 1],   #a4
                [1, 1, 3, 5, 4, 1, 5]    #a6
                ])

# Call Electre
concordance, discordance, dominance_s, dominance_w, rank_D, rank_A, rank_N, rank_P = electre_ii(dataset, W = W, c_minus = c_minus, c_zero = c_zero, c_plus = c_plus, d_minus = d_minus, d_plus = d_plus, graph = True)

##############################################################################

# ELECTRE III

# Parameters
Q = [0.30, 0.30, 0.30, 0.30]
P = [0.50, 0.50, 0.50, 0.50]
V = [0.70, 0.70, 0.70, 0.70]
W = [9.00, 8.24, 5.98, 8.48]

# Dataset
dataset = np.array([
                [8.84, 8.79, 6.43, 6.95],   #a1
                [8.57, 8.51, 5.47, 6.91],   #a2
                [7.76, 7.75, 5.34, 8.76],   #a3
                [7.97, 9.12, 5.93, 8.09],   #a4
                [9.03, 8.97, 8.19, 8.10],   #a5
                [7.41, 7.87, 6.77, 7.23]    #a6
                ])
    
# Call Electre
global_concordance, credibility, rank_D, rank_A, rank_N, rank_P = electre_iii(dataset, P = P, Q = Q, V = V, W = W, graph = True)


##############################################################################

# ELECTRE IV
    
# Parameters
    
Q = [ 10,  10,  10,  10,  10,  10,  10,  10]
P = [ 20,  20,  20,  20,  20,  20,  20,  20]
V = [100, 100, 100, 100, 100, 100, 100, 100]

# Dataset
dataset = np.array([
                [15, 80, 60, 30, 60, 50, 60,  70],   #a1
                [25,  0, 40, 30, 40, 40, 50, 140],   #a2
                [25,  0, 50, 30, 40, 40, 50, 140],   #a3
                [25,  0, 50, 30, 50, 40, 70, 140],   #a4
                [25,  0, 50, 30, 50, 40, 50, 140],   #a5
                [15, 20, 50, 30, 50, 60, 60, 100],   #a6
                [15, 80, 50, 50, 40, 90, 60, 100],   #a7
                ])

# Call Electre
credibility, rank_D, rank_A, rank_N, rank_P = electre_iv(dataset, P = P, Q = Q, V = V, graph = True)

##############################################################################

# ELECTRE Tri-B

# Dataset
dataset = np.array([
                [75, 67, 85, 82, 90],   #a1
                [28, 35, 70, 90, 95],   #a2
                [45, 60, 55, 68, 60]    #a3
                ])

Q = [ 5,  5,  5,  5,  5]
P = [10, 10, 10, 10, 10]
V = [30, 30, 30, 30, 30]
W = [ 1,  1,  1,  1,  1]
B = [[50, 48, 55, 55, 60], [70, 75, 80, 75, 85]]


classification = electre_tri_b(dataset, W , Q , P , V , B , cut_level = 0.75, verbose = False, rule = 'oc', graph = True)

##############################################################################

# PROMETHEE I
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5']

# Calling Promethee
p1 = promethee_i(dataset, W = W, Q = Q, S = S, P = P, F = F, graph = True)

##############################################################################

# PROMETHEE II
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5']

# Calling Promethee
p2 = promethee_ii(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, topn = 10, graph = True)

##############################################################################

# PROMETHEE III
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5']

# Calling Promethee
p3 =  promethee_iii(dataset, W = W, Q = Q, S = S, P = P, F = F, lmbd = 0.15, graph = True)

##############################################################################

# PROMETHEE IV
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5']

# Calling Promethee
p4 = promethee_iv(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, steps = 0.001, topn = 10, graph = True)

##############################################################################

# PROMETHEE V
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5']

# Constraint 1
criteria = 4

# Constraint 2
cost   = [10, 10, 15, 10, 10, 15]
budget = 50

# Constraint 3
forbidden = [['a5', 'a4']]
forbidden = [['a1', 'a4'], ['a1', 'a5']]

# Calling Promethee
p5 = promethee_v(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, criteria = criteria, cost = cost, budget = budget, forbidden = forbidden, iterations = 500)

##############################################################################

# PROMETHEE VI
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
F = ['t5', 't5', 't5', 't5']

W_lower = np.array([1.00, 1.00, 4.95, 1.00])
W_upper  = np.array([3.10, 1.50, 5.00, 5.00])

W_lower = np.array([5.00, 5.00, 1.00, 1.00])
W_upper  = np.array([9.00, 9.00, 5.00, 5.00])

# Calling Promethee
p6_minus, p6, p6_plus = promethee_vi(dataset, W_lower = W_lower, W_upper = W_upper, Q = Q, S = S, P = P, F = F, sort = True, topn = 10, iterations = 1000, graph = True)

##############################################################################

# PROMETHEE Gaia
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950], 
        [8.570, 8.510, 5.470, 6.910], 
        [7.760, 7.750, 5.340, 8.760], 
        [7.970, 9.120, 5.930, 8.090], 
        [9.030, 8.970, 8.190, 8.100], 
        [7.410, 7.870, 6.770, 7.230]
        ])

Q = np.array([0.3, 0.3, 0.3, 0.3])
S = np.array([0.4, 0.4, 0.4, 0.4])
P = np.array([0.5, 0.5, 0.5, 0.5])
W = np.array([9.00, 8.24, 5.98, 8.48])
F = ['t5', 't5', 't5', 't5']

# Calling Promethee
promethee_gaia(dataset, W = W, Q = Q, S = S, P = P, F = F)

##############################################################################
