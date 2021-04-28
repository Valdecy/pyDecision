###############################################################################

# Required Libraries
import numpy as np

# AHP
from py_decisions.ahp.ahp           import ahp_method

# COPRAS
from py_decisions.copras.copras     import copras_method

# Fuzzy AHP 
from py_decisions.ahp.fuzzy_ahp     import fuzzy_ahp_method

# Borda
from py_decisions.borda.borda       import borda_method

# BWM
from py_decisions.bwm.bwm           import bw_method

# CODAS
from py_decisions.codas.codas       import codas_method

# ARAS
from py_decisions.aras.aras       import aras_method

# CRITIC
from py_decisions.critic.critic     import critic_method

# DEMATEL
from py_decisions.dematel.dematel   import dematel_method

# Fuzzy DEMATEL
from py_decisions.dematel.fuzzy_dematel import fuzzy_dematel_method

# EDAS
from py_decisions.edas.edas         import edas_method

# Fuzzy EDAS
from py_decisions.edas.fuzzy_edas   import fuzzy_edas_method

# ELECTRE
from py_decisions.electre.e_i       import electre_i
from py_decisions.electre.e_i_s     import electre_i_s
from py_decisions.electre.e_i_v     import electre_i_v
from py_decisions.electre.e_ii      import electre_ii
from py_decisions.electre.e_iii     import electre_iii
from py_decisions.electre.e_iv      import electre_iv
from py_decisions.electre.e_tri_b   import electre_tri_b

# GRA
from py_decisions.gra.gra           import gra_method

# MOORA
from py_decisions.moora.moora       import moora_method

# PROMETHEE
from py_decisions.promethee.p_i     import promethee_i
from py_decisions.promethee.p_ii    import promethee_ii
from py_decisions.promethee.p_iii   import promethee_iii
from py_decisions.promethee.p_iv    import promethee_iv
from py_decisions.promethee.p_v     import promethee_v
from py_decisions.promethee.p_vi    import promethee_vi
from py_decisions.promethee.p_xgaia import promethee_gaia

# TOPSIS
from py_decisions.topsis.topsis       import topsis_method

# Fuzzy TOPSIS
from py_decisions.topsis.fuzzy_topsis import fuzzy_topsis_method

# VIKOR
from py_decisions.vikor.vikor         import vikor_method, ranking

# Fuzzy VIKOR
from py_decisions.vikor.fuzzy_vikor   import fuzzy_vikor_method

# WSM, WPM, WASPAS
from py_decisions.waspas.waspas       import waspas_method

##############################################################################

# AHP

# Parameters
weight_derivation = 'geometric'

# Dataset
dataset = np.array([
# g1     g2     g3    g4      g5     g6    g7
[1  ,   1/3,   1/5,   1  ,   1/4,   1/2,   3  ],   #g1
[3  ,   1  ,   1/2,   2  ,   1/3,   3  ,   3  ],   #g2
[5  ,   2  ,   1  ,   4  ,   5  ,   6  ,   5  ],   #g3
[1  ,   1/2,   1/4,   1  ,   1/4,   1  ,   2  ],   #g4
[4  ,   3  ,   1/5,   4  ,   1  ,   3  ,   2  ],   #g5
[2  ,   1/3,   1/6,   1  ,   1/3,   1  ,   1/3],   #g6
[1/3,   1/3,   1/5,   1/2,   1/2,   3  ,   1  ]    #g7
])

# Call AHP Function
weights, rc = ahp_method(dataset, wd = weight_derivation)

##############################################################################

# ARAS

# Weights
weights = np.array([0.28, 0.14, 0.05, 0.24, 0.19, 0.05, 0.05])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']

# Dataset
dataset = np.array([
                    [75.5, 420,	 74.2,	2.8,	21.4,	0.37,	0.16],   #a1
                    [95,   91,	 70,	2.68,	22.1,	0.33,	0.16],   #a2
                    [770,  1365, 189,	7.9,	16.9,	0.04,	0.08],   #a3
                    [187,  1120, 210,	7.9,	14.4,	0.03,	0.08],   #a4
                    [179,  875,	 112,	4.43,	9.4,	0.016,	0.09],   #a5
                    [239,  1190, 217,	8.51,	11.5,	0.31,	0.07],   #a6
                    [273,  200,	 112,	8.53,	19.9,	0.29,	0.06]    #a7
                    ])

# Call ARAS Function
rank = aras_method(dataset, weights, criterion_type, graph = True)

##############################################################################

# Fuzzy AHP

# Dataset
dataset = list([
    #          g1              g2                g3                  g4
    [ (  1,   1,   1), (  4,   5,   6), (  3,   4,   5), (  6,   7,   8) ],   #g1
    [ (1/6, 1/5, 1/4), (  1,   1,   1), (1/3, 1/2, 1/1), (  2,   3,   4) ],   #g2
    [ (1/5, 1/4, 1/3), (  1,   2,   3), (  1,   1,   1), (  2,   3,   4) ],   #g3
    [ (1/8, 1/7, 1/6), (1/4, 1/3, 1/2), (1/4, 1/3, 1/2), (  1,   1,   1) ]    #g4
    ])

# Call Fuzzy AHP Function        
fuzzy_weights, defuzzified_weights, normalized_weights = fuzzy_ahp_method(dataset)

##############################################################################

# Borda
 
# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min']

# Dataset
dataset = np.array([
                [7, 9, 9, 5],   #a1
                [8, 7, 8, 7],   #a2
                [9, 6, 8, 9],   #a3
                [6, 7, 8, 6]    #a4
                ])

# Call Borda Function
rank = borda_method(dataset, criterion_type, graph = True)

##############################################################################

# BWM
 
# Dataset
dataset = np.array([
                # g1   g2   g3   g4
                [250,  16,  12,  5],
                [200,  16,  8 ,  3],   
                [300,  32,  16,  4],
                [275,  32,  8 ,  4],
                [225,  16,  16,  2]
                ])

# Most Important Criteria
mic = np.array([1, 3, 4, 7])

# Least Important Criteria
lic = np.array([7, 5, 5, 1])

# Call BWM Function
weights =  bw_method(dataset, mic, lic, size = 50, iterations = 150)

##############################################################################

# CODAS

# Weights
weights = np.array([0.28, 0.14, 0.05, 0.24, 0.19, 0.05, 0.05])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']

# Dataset
dataset = np.array([
                    [75.5, 420,	 74.2,	2.8,	21.4,	0.37,	0.16],   #a1
                    [95,   91,	 70,	2.68,	22.1,	0.33,	0.16],   #a2
                    [770,  1365, 189,	7.9,	16.9,	0.04,	0.08],   #a3
                    [187,  1120, 210,	7.9,	14.4,	0.03,	0.08],   #a4
                    [179,  875,	 112,	4.43,	9.4,	0.016,	0.09],   #a5
                    [239,  1190, 217,	8.51,	11.5,	0.31,	0.07],   #a6
                    [273,  200,	 112,	8.53,	19.9,	0.29,	0.06]    #a7
                    ])

# Call CODAS Function
rank = codas_method(dataset, weights, criterion_type, lmbd = 0.02, graph = True)

##############################################################################

# COPRAS

# Weights
weights = np.array([0.28, 0.14, 0.05, 0.24, 0.19, 0.05, 0.05])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']

# Dataset
dataset = np.array([
                    [75.5, 420,	 74.2,	2.8,	21.4,	0.37,	0.16],   #a1
                    [95,   91,	 70,	2.68,	22.1,	0.33,	0.16],   #a2
                    [770,  1365, 189,	7.9,	16.9,	0.04,	0.08],   #a3
                    [187,  1120, 210,	7.9,	14.4,	0.03,	0.08],   #a4
                    [179,  875,	 112,	4.43,	9.4,	0.016,	0.09],   #a5
                    [239,  1190, 217,	8.51,	11.5,	0.31,	0.07],   #a6
                    [273,  200,	 112,	8.53,	19.9,	0.29,	0.06]    #a7
                    ])

# Call COPRAS Function
rank = copras_method(dataset, weights, criterion_type, graph = True)

##############################################################################

# CRITIC
 
# Load Criterion Type: 'max' or 'min'
criterion_type = ['min', 'max', 'max', 'max']

# Dataset
dataset = np.array([
                # g1   g2   g3   g4
                [250,  16,  12,  5],
                [200,  16,  8 ,  3],   
                [300,  32,  16,  4],
                [275,  32,  8 ,  4],
                [225,  16,  16,  2]
                ])

# Call CRITIC Function  
weights = critic_method(dataset, criterion_type)

###############################################################################

# DEMATEL

# Dataset # Scale: 0 (No Influence), 1 (Low Influence), 2 (Medium Influence), 3 (High Influence), 4 (Very High Influence)
dataset = np.array([
    # 'g1' 'g2' 'g3' 'g4'
    [  0,   1,   2,   0  ],   #g1
    [  3,   0,   4,   4  ],   #g2
    [  3,   2,   0,   1  ],   #g3
    [  4,   1,   2,   0  ]    #g4
    ])

# Call DEMATEL Function  
D_plus_R, D_minus_R, weights = dematel_method(dataset, size_x = 15, size_y = 10)

##############################################################################

# Fuzzy DEMATEL

# Dataset # Scale: No Influence (0, 0, 1/4), Low Influence (1/4, 1/2, 3/4), Medium Influence (1/4, 1/2, 3/4), High Influence (1/2, 3/4, 1), Very High Influence (3/4, 1, 1)

dataset = list([
    #          g1              g2                g3                  g4
    [  (  0,   0, 1/4),  (  0, 1/4, 1/2),  (1/4, 1/2, 3/4),  (  0,   0, 1/4)  ],   #g1
    [  (1/2, 3/4,   1),  (  0,   0, 1/4),  (3/4,   1,   1),  (3/4,   1,   1)  ],   #g2
    [  (1/2, 3/4,   1),  (1/4, 1/2, 3/4),  (  0,   0, 1/4),  (  0, 1/4, 1/2)  ],   #g3
    [  (3/4,   1,   1),  (  0, 1/4, 1/2),  (1/4, 1/2, 3/4),  (  0,   0, 1/4)  ]    #g4
    ])

# Call Fuzzy DEMATEL Function  
D_plus_R, D_minus_R, weights = fuzzy_dematel_method(dataset, size_x = 15, size_y = 10)

##############################################################################

# EDAS
 
# Weights
weights = np.array([ [0.35, 0.30, 0.20, 0.15] ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['min', 'max', 'max', 'max']

# Dataset
dataset = np.array([
                [250, 16, 12, 5],   #a1
                [200, 16, 8 , 3],   #a2
                [300, 32, 16, 4],   #a3
                [275, 32, 8 , 4],   #a4
                [225, 16, 16, 2]    #a5
                ])

# Call EDAS Function  
rank = edas_method(dataset, criterion_type, weights, graph = True)

##############################################################################

# Fuzzy EDAS
 
# Weigths
weights = list([
          [ (  0.1,   0.2,   0.3), (  0.7,   0.8,   0.9), (  0.3,   0.5,   0.8) ]    
    ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'min']

# Dataset
dataset = list([
    [ (  3,   6,   9), (  5,   8,   9), (  5,   7,   9) ],   #a1
    [ (  5,   7,   9), (  3,   7,   9), (  3,   5,   7) ],   #a2
    [ (  5,   8,   9), (  3,   5,   7), (  1,   2,   3) ],   #a3
    [ (  1,   2,   4), (  1,   4,   7), (  1,   2,   5) ]    #a4
    ])

# Call Fuzzy EDAS
rank = fuzzy_edas_method(dataset, criterion_type, weights, graph = True)

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

# Call Electre I Function
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

# Call Electre I_s Function
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

# Call Electre I_v Function
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

# Call Electre II Function
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
    
# Call Electre III Function
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

# Call Electre IV Function
credibility, rank_D, rank_A, rank_N, rank_P = electre_iv(dataset, P = P, Q = Q, V = V, graph = True)

##############################################################################

# ELECTRE Tri-B

# Parameters 
Q = [ 5,  5,  5,  5,  5]
P = [10, 10, 10, 10, 10]
V = [30, 30, 30, 30, 30]
W = [ 1,  1,  1,  1,  1]
B = [[50, 48, 55, 55, 60], [70, 75, 80, 75, 85]]

# Dataset
dataset = np.array([
                [75, 67, 85, 82, 90],   #a1
                [28, 35, 70, 90, 95],   #a2
                [45, 60, 55, 68, 60]    #a3
                ])

# Call Electre Tri-B Function
classification = electre_tri_b(dataset, W , Q , P , V , B , cut_level = 0.75, verbose = False, rule = 'oc', graph = True)

##############################################################################

# GRA

# Weights
weights = np.array([ [0.35, 0.30, 0.20, 0.15] ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['min', 'max', 'max', 'max']

# Dataset
dataset = np.array([
                [250, 16, 12, 5],   #a1
                [200, 16, 8 , 3],   #a2
                [300, 32, 16, 4],   #a3
                [275, 32, 8 , 4],   #a4
                [225, 16, 16, 2]    #a5
                ])

# Call GRA Function
gra_grade = gra_method(dataset, criterion_type, weights, epsilon = 0.5, graph = True)

##############################################################################

# MOORA

# Weights
weights = np.array([0.297, 0.025, 0.035, 0.076, 0.154, 0.053, 0.104, 0.017, 0.025, 0.214])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'min', 'min']

# Dataset
dataset = np.array([
                    [3.5, 6, 1256, 4, 16, 3, 17.3, 8, 2.82, 4100],   #a1
                    [3.1, 4, 1000, 2, 8,  1, 15.6, 5, 3.08, 3800],   #a2
                    [3.6, 6, 2000, 4, 16, 3, 17.3, 5, 2.9,  4000],   #a3
                    [3,   4, 1000, 2, 8,  2, 17.3, 5, 2.6,  3500],   #a4
                    [3.3, 6, 1008, 4, 12, 3, 15.6, 8, 2.3,  3800],   #a5
                    [3.6, 6, 1000, 2, 16, 3, 15.6, 5, 2.8,  4000],   #a6
                    [3.5, 6, 1256, 2, 16, 1, 15.6, 6, 2.9,  4000]    #a7
                   ])

# Call MOORA Function
rank = moora_method(dataset, weights, criterion_type, graph = True)

##############################################################################

# PROMETHEE I

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])

# Call Promethee I
p1 = promethee_i(dataset, W = W, Q = Q, S = S, P = P, F = F, graph = True)

##############################################################################

# PROMETHEE II

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])


# Call Promethee II
p2 = promethee_ii(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, topn = 10, graph = True)

##############################################################################

# PROMETHEE III

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])

# Call Promethee III
p3 =  promethee_iii(dataset, W = W, Q = Q, S = S, P = P, F = F, lmbd = 0.15, graph = True)

##############################################################################

# PROMETHEE IV

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])

# Call Promethee IV
p4 = promethee_iv(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, steps = 0.001, topn = 10, graph = True)

##############################################################################

# PROMETHEE V

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
W = [9.00, 8.24, 5.98, 8.48]
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

# Constraint 1
criteria = 4

# Constraint 2
cost   = [10, 10, 15, 10, 10, 15]
budget = 50

# Constraint 3
forbidden = [['a1', 'a4'], ['a1', 'a5']]

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])

# Call Promethee V
p5 = promethee_v(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, criteria = criteria, cost = cost, budget = budget, forbidden = forbidden, iterations = 500)

##############################################################################

# PROMETHEE VI

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5]
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

W_lower = np.array([5.00, 5.00, 1.00, 1.00])
W_upper  = np.array([9.00, 9.00, 5.00, 5.00])

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])

# Call Promethee VI
p6_minus, p6, p6_plus = promethee_vi(dataset, W_lower = W_lower, W_upper = W_upper, Q = Q, S = S, P = P, F = F, sort = True, topn = 10, iterations = 1000, graph = True)

##############################################################################

# PROMETHEE Gaia

# Parameters 
Q = np.array([0.3, 0.3, 0.3, 0.3])
S = np.array([0.4, 0.4, 0.4, 0.4])
P = np.array([0.5, 0.5, 0.5, 0.5])
W = np.array([9.00, 8.24, 5.98, 8.48])
F = ['t5', 't5', 't5', 't5'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form

# Dataset
dataset = np.array([
        [8.840, 8.790, 6.430, 6.950],  #a1
        [8.570, 8.510, 5.470, 6.910],  #a2
        [7.760, 7.750, 5.340, 8.760],  #a3
        [7.970, 9.120, 5.930, 8.090],  #a4
        [9.030, 8.970, 8.190, 8.100],  #a5
        [7.410, 7.870, 6.770, 7.230]   #a6
        ])

# Call Promethee Gaia
promethee_gaia(dataset, W = W, Q = Q, S = S, P = P, F = F)

##############################################################################

# TOPSIS
 
# Weights
weights = np.array([ [0.1, 0.4, 0.3, 0.2] ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min']

# Dataset
dataset = np.array([
                [7, 9, 9, 8],   #a1
                [8, 7, 8, 7],   #a2
                [9, 6, 8, 9],   #a3
                [6, 7, 8, 6]    #a4
                ])

# Call TOPSIS
relative_closeness = topsis_method(dataset, weights, criterion_type, graph = True)

##############################################################################

# Fuzzy TOPSIS
 
# Weigths
weights = list([
          [ (  0.1,   0.2,   0.3), (  0.7,   0.8,   0.9), (  0.3,   0.5,   0.8) ]    
    ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'min']

# Dataset
dataset = list([
    [ (  3,   6,   9), (  5,   8,   9), (  5,   7,   9) ],   #a1
    [ (  5,   7,   9), (  3,   7,   9), (  3,   5,   7) ],   #a2
    [ (  5,   8,   9), (  3,   5,   7), (  1,   2,   3) ],   #a3
    [ (  1,   2,   4), (  1,   4,   7), (  1,   2,   5) ]    #a4
    ])

# Call Fuzzy TOPSIS
relative_closeness = fuzzy_topsis_method(dataset, weights, criterion_type, graph = True)

##############################################################################

# VIKOR
 
# Weights
weights = np.array([ [0.35, 0.30, 0.20, 0.15] ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['min', 'max', 'max', 'max']

# Dataset
dataset = np.array([
                [250, 16, 12, 5],   #a1
                [200, 16, 8 , 3],   #a2
                [300, 32, 16, 4],   #a3
                [275, 32, 8 , 4],   #a4
                [225, 16, 16, 2]    #a5
                ])

# Call VIKOR
s, r, q, c_solution = vikor_method(dataset, weights, criterion_type, strategy_coefficient = 0.5, graph = False)

# Graph Solutions
ranking(s) 
ranking(r) 
ranking(q) 
ranking(c_solution) # Final Solution

##############################################################################

# Fuzzy VIKOR
 
# Weigths
weights = list([
          [ (  0.1,   0.2,   0.3), (  0.7,   0.8,   0.9), (  0.3,   0.5,   0.8) ]    
    ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'min']

# Dataset
dataset = list([
    [ (  3,   6,   9), (  5,   8,   9), (  5,   7,   9) ],   #a1
    [ (  5,   7,   9), (  3,   7,   9), (  3,   5,   7) ],   #a2
    [ (  5,   8,   9), (  3,   5,   7), (  1,   2,   3) ],   #a3
    [ (  1,   2,   4), (  1,   4,   7), (  1,   2,   5) ]    #a4
    ])

# Call Fuzzy VIKOR
s, r, q, c_solution = fuzzy_vikor_method(dataset, weights, criterion_type, strategy_coefficient = 0.5, graph = True)

# Graph Solutions
ranking(s) 
ranking(r) 
ranking(q) 
ranking(c_solution) # Final Solution

##############################################################################

# WSM, WPM, WASPAS
 
# Weights
weights = np.array([ [0.35, 0.30, 0.20, 0.15] ])

# Load Criterion Type: 'max' or 'min'
criterion_type = ['min', 'max', 'max', 'max']

# Dataset
dataset = np.array([
                [250, 16, 12, 5],   #a1
                [200, 16, 8 , 3],   #a2
                [300, 32, 16, 4],   #a3
                [275, 32, 8 , 4],   #a4
                [225, 16, 16, 2]    #a5
                ])
# Lambda
lambda_value = 0.5

# Call WASPAS Function  
wsm, wpm, waspas = waspas_method(dataset, criterion_type, weights, lambda_value)

###############################################################################
