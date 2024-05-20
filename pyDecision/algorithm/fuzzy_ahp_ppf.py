###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# PPF AHP (Proportional Picture Fuzzy sets combined with the Analytic Hierarchy Process)
def ppf_ahp_method(comparison_matrix):
    
    inc_rat = np.array([0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59])
    
    ################################################
    def ppf_to_triple(comparison_matrix):
        triple_matrix  = []
        for row in comparison_matrix:
            triple_row = []
            for k1, k2 in row:
                if (k1 == 0 and k2 == 0):
                    triple_row.append((1.0, 0.0, 0.0))  
                else:
                    pi    = 1 / (1 + k1 + k2)
                    mu    = k1 * pi
                    theta = k2 * pi
                    triple_row.append((mu, pi, theta))  
            triple_matrix.append(triple_row)
        return triple_matrix

    def triple_to_crisp(triple_matrix):
        crisp_matrix = np.zeros((len(triple_matrix), len(triple_matrix)))
        for i, row in enumerate(triple_matrix):
            for j, (mu, pi, theta) in enumerate(row):
                if (i == j):
                    crisp_matrix[i, j] = 1.0
                else:
                    score   = 10 * (mu - pi/2 - theta)
                    score_n = 1 / (10 * (theta - pi/2 - mu))
                    if (score > 0):
                        crisp_matrix[i, j] = score
                    else:
                        crisp_matrix[i, j] = score_n
        return crisp_matrix
    
    def normalize_matrix(crisp_matrix):
        n_matrix = crisp_matrix / crisp_matrix.sum(axis = 0)
        return n_matrix

    def consistency_ratio(crisp_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(crisp_matrix)
        eigenvalues_real          = np.real(eigenvalues)
        lamb_max_index            = np.argmax(eigenvalues_real)
        lamb_max                  = eigenvalues_real[lamb_max_index]
        cons_ind                  = (lamb_max - crisp_matrix.shape[1]) / (crisp_matrix.shape[1] - 1)
        rc                        = cons_ind / inc_rat[crisp_matrix.shape[1]]
        return rc

    def calculate_weights(n_matrix):
        weights = n_matrix.mean(axis = 1)
        return weights
    ################################################

    triple_matrix = ppf_to_triple(comparison_matrix)
    crisp_matrix  = triple_to_crisp(triple_matrix)
    n_matrix      = normalize_matrix(crisp_matrix)
    weights       = calculate_weights(n_matrix)
    rc            = consistency_ratio(crisp_matrix)
    return weights, rc

###############################################################################
