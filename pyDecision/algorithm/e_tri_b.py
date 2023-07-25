###############################################################################

# Required Libraries
import copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy  as np

from sklearn.decomposition import TruncatedSVD

###############################################################################
    
# Function: Concordance Matrices and Vectors
def concordance_matrices_vectors(performance_matrix, number_of_profiles, number_of_alternatives, B, P, Q, W):         
    n_rows = number_of_profiles * number_of_alternatives
    n_cols = performance_matrix.shape[1]
    # Concordance Matrix x_b
    concordance_matrix = np.zeros((n_rows, n_cols))
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, concordance_matrix .shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, concordance_matrix.shape[1]):
            if (B[count, j] - performance_matrix[alternative, j] >= P[0, j]):
                concordance_matrix[i, j] = 0
            elif (B[count, j] - performance_matrix[alternative, j] < Q[0, j]):
                concordance_matrix[i, j] = 1
            else:
                concordance_matrix[i, j] = (P[0, j] - B[count, j] + performance_matrix[alternative, j])/(P[0, j] - Q[0, j])     
    # Concordance Vector x_b
    concordance_vector = np.zeros((n_rows, 1))
    for i in range(0, concordance_vector.shape[0]):
        for j in range(0, concordance_matrix.shape[1]):
            concordance_vector[i, 0] = concordance_vector[i, 0] + concordance_matrix[i, j]*W[j]
        if (W.sum(axis = 0) != 0):
            concordance_vector[i, 0] = concordance_vector[i, 0]/W.sum(axis = 0)           
    # Concordance Matrix b_x
    concordance_matrix_inv = np.zeros((n_rows, n_cols))
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, concordance_matrix_inv.shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, concordance_matrix_inv.shape[1]):
            if (-B[count, j] + performance_matrix[alternative, j] >= P[0, j]):
                concordance_matrix_inv[i, j] = 0
            elif (-B[count, j] + performance_matrix[alternative, j] < Q[0, j]):
                concordance_matrix_inv[i, j] = 1
            else:
                concordance_matrix_inv[i, j] = (P[0, j] + B[count, j] - performance_matrix[alternative, j])/(P[0, j] - Q[0, j])        
    # Concordance Vector b_x
    concordance_vector_inv = np.zeros((n_rows, 1))
    for i in range(0, concordance_vector_inv.shape[0]):
        for j in range(0, concordance_matrix_inv.shape[1]):
            concordance_vector_inv[i, 0] = concordance_vector_inv[i, 0] + concordance_matrix_inv[i, j]*W[j]
        if (W.sum(axis = 0) != 0):
            concordance_vector_inv[i, 0] = concordance_vector_inv[i, 0]/W.sum(axis = 0)    
    return concordance_matrix, concordance_matrix_inv, concordance_vector, concordance_vector_inv

# Function: Discordance Matrices
def discordance_matrices(performance_matrix, number_of_profiles, number_of_alternatives, B, P, V):
    n_rows = number_of_profiles * number_of_alternatives
    n_cols = performance_matrix.shape[1]
    # Discordance Matrix x_b
    disconcordance_matrix = np.zeros((n_rows, n_cols))
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, disconcordance_matrix.shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, disconcordance_matrix.shape[1]):
            if (B[count, j] - performance_matrix[alternative, j] < P[0, j]):
                disconcordance_matrix[i, j] = 0
            elif (B[count, j] - performance_matrix[alternative, j] >= V[0, j]):
                disconcordance_matrix[i, j] = 1
            else:
                disconcordance_matrix[i, j] = (-P[0, j] + B[count, j] - performance_matrix[alternative, j])/(V[0, j] - P[0, j])  
    # Discordance Matrix b_x
    disconcordance_matrix_inv = np.zeros((n_rows, n_cols))    
    count = B.shape[0] - 1
    alternative = -number_of_alternatives       
    for i in range(0, disconcordance_matrix_inv.shape[0]):
        if (i > 0 and i % number_of_alternatives == 0):
                count = count - 1
        if (i > 0 and i % number_of_alternatives != 0):
            alternative = alternative + 1
        elif (i > 0 and i % number_of_alternatives == 0):
            alternative = -number_of_alternatives 
        for j in range(0, disconcordance_matrix_inv.shape[1]):
            if (-B[count, j] + performance_matrix[alternative, j] < P[0, j]):
                disconcordance_matrix_inv[i, j] = 0
            elif (-B[count, j] + performance_matrix[alternative, j] >= V[0, j]):
                disconcordance_matrix_inv[i, j] = 1
            else:
                disconcordance_matrix_inv[i, j] = (-P[0, j] - B[count, j] + performance_matrix[alternative, j])/(V[0, j] - P[0, j])
    return disconcordance_matrix, disconcordance_matrix_inv

# Function: Credibility Vectors
def credibility_vectors(number_of_profiles, number_of_alternatives, concordance_matrix, concordance_matrix_inv, concordance_vector, concordance_vector_inv, disconcordance_matrix, disconcordance_matrix_inv):
    n_rows = number_of_profiles * number_of_alternatives  
    # Credibility Vector x_b
    credibility_vector = np.zeros((n_rows, 1))
    for i in range(0, credibility_vector.shape[0]):
        credibility_vector[i, 0] = concordance_vector[i, 0]
        for j in range(0, concordance_matrix.shape[1]):
            if (disconcordance_matrix[i, j] > concordance_vector[i, 0]):
                value = (1 - disconcordance_matrix[i, j])/(1 - concordance_vector[i, 0])
                credibility_vector[i, 0] = credibility_vector[i, 0]*value  
    # Credibility Vector b_x        
    credibility_vector_inv = np.zeros((n_rows, 1))
    for i in range(0, credibility_vector_inv.shape[0]):
        credibility_vector_inv[i, 0] = concordance_vector_inv[i, 0]
        for j in range(0, concordance_matrix_inv.shape[1]):
            if (disconcordance_matrix_inv[i, j] > concordance_vector_inv[i, 0]):
                value = (1 - disconcordance_matrix_inv[i, j])/(1 - concordance_vector_inv[i, 0])
                credibility_vector_inv[i, 0] = credibility_vector_inv[i, 0]*value
    return credibility_vector, credibility_vector_inv

# Function: Fuzzy Logic
def fuzzy_logic(number_of_profiles, number_of_alternatives, credibility_vector, credibility_vector_inv, cut_level):
    n_rows = number_of_profiles * number_of_alternatives 
    fuzzy_vector = []
    fuzzy_matrix = [[]]* number_of_alternatives 
    for i in range(0, n_rows):
        if (credibility_vector[i, 0] >= cut_level and credibility_vector_inv[i, 0] >= cut_level):
            fuzzy_vector.append('I')
        elif (credibility_vector[i, 0] >= cut_level and credibility_vector_inv[i, 0] <  cut_level):
            fuzzy_vector.append('>')
        elif (credibility_vector[i, 0] <  cut_level and credibility_vector_inv[i, 0] >= cut_level):
            fuzzy_vector.append('<')
        elif (credibility_vector[i, 0] <  cut_level and credibility_vector_inv[i, 0] <  cut_level):
            fuzzy_vector.append('R')
    
    fm = [fuzzy_vector[x:x+number_of_alternatives] for x in range(0, len(fuzzy_vector), number_of_alternatives)]
    for j in range(number_of_profiles-1, -1,-1):
        for i in range(0, number_of_alternatives):
            fuzzy_matrix[i] = fuzzy_matrix[i] + [fm[j][i]]
    return fuzzy_matrix

# Function: Classification
def classification_algorithm(number_of_profiles, number_of_alternatives, fuzzy_matrix, rule, verbose = True):
    classification = []
    if (rule == 'pc'):
        # Pessimist Classification
        for i1 in range(0, number_of_alternatives):
            class_i = number_of_profiles
            count   = 0
            for i2 in range(0, number_of_profiles):
                count = count + 1
                if (fuzzy_matrix[i1][i2] == '>'):
                    class_i = int(number_of_profiles - count)
            classification.append(class_i)
            if (verbose == True):
                print('a' + str(i1 + 1) + ' = ' + 'C' + str(class_i))  
    elif(rule == 'oc'):
        # Optimistic Classification
        for i1 in range(0, number_of_alternatives):
            class_i = 0
            count   = 0
            for i2 in range(number_of_profiles - 1, -1, -1):
                count = count + 1
                if (fuzzy_matrix[i1][i2] == '<'):
                    class_i = int(count)
            classification.append(class_i)
            if (verbose == True):
                print('a' + str(i1 + 1) + ' = ' + 'C' + str(class_i))    
    return classification   

# Function: Plot Projected Points 
def plot_points(data, classification):
    plt.style.use('ggplot')
    colors = {'A':'#bf77f6', 'B':'#fed8b1', 'C':'#d1ffbd', 'D':'#f08080', 'E':'#3a18b1', 'F':'#ff796c', 'G':'#04d8b2', 'H':'#ffb07c', 'I':'#aaa662', 'J':'#0485d1', 'K':'#fffe7a', 'L':'#b0dd16', 'M':'#85679', 'N':'#12e193', 'O':'#82cafc', 'P':'#ac9362', 'Q':'#f8481c', 'R':'#c292a1', 'S':'#c0fa8b', 'T':'#ca7b80', 'U':'#f4d054', 'V':'#fbdd7e', 'W':'#ffff7e', 'X':'#cd7584', 'Y':'#f9bc08', 'Z':'#c7c10c'}
    classification_ = copy.deepcopy(classification)
    color_leg = {}
    if (data.shape[1] == 2):
        data_proj = np.copy(data)
    else:
        tSVD      = TruncatedSVD(n_components = 2, n_iter = 100, random_state = 42)
        tSVD_proj = tSVD.fit_transform(data)
        data_proj = np.copy(tSVD_proj)
        #variance  = sum(np.var(tSVD_proj, axis = 0) / np.var(tSVD_proj, axis = 0).sum())
    class_list  = list(set(classification_))
    for i in range(0, len(classification_)):
        classification_[i] = str(classification_[i])
    for i in range(0, len(classification_)):
        for j in range(0, len(class_list)):
            classification_[i] = classification_[i].replace(str(class_list[j]), str(chr(ord('A') + class_list[j])))
    class_list = list(set(classification_))
    class_list.sort() 
    for i in range(0, len(class_list)):
        color_leg[class_list[i]] = colors[class_list[i]]
    patchList = []
    for key in color_leg:
        data_key = mpatches.Patch(color = color_leg[key], label = key)
        patchList.append(data_key)
    for i in range(0, data_proj.shape[0]):
        plt.text(data_proj[i, 0], data_proj[i, 1], 'x' + str(i+1), size = 10, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = colors[classification_[i]],))
    plt.gca().legend(handles = patchList, loc = 'center left', bbox_to_anchor = (1.05, 0.5))
    axes = plt.gca()
    xmin = np.amin(data_proj[:,0])
    xmax = np.amax(data_proj[:,0])
    axes.set_xlim([xmin*0.7, xmax*1])
    ymin = np.amin(data_proj[:,1])
    ymax = np.amax(data_proj[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin*0.7, ymax*1])
    plt.show()
    return
    
# Function: ELECTRE TRI-B
def electre_tri_b(performance_matrix, W = [], Q = [], P = [], V = [], B = [], cut_level = 1.0, verbose = True, rule = 'pc', graph = False):  
    # Loading Parameters
    if (isinstance(B[0], list)):
        number_of_profiles = len(B)
    else:
        number_of_profiles = 1
    number_of_alternatives = performance_matrix.shape[0]    
    p_vector = np.zeros((1, performance_matrix.shape[1]))
    q_vector = np.zeros((1, performance_matrix.shape[1]))
    v_vector = np.zeros((1, performance_matrix.shape[1]))    
    for i in range(0, p_vector.shape[1]):
        p_vector[0][i] = P[i]
        q_vector[0][i] = Q[i]
        v_vector[0][i] = V[i]   
    w_vector = np.array(W)
    b_matrix = np.array(B)
    if (isinstance(B[0], list)):
        b_matrix = np.array(B) 
    else:
        b_matrix = np.zeros((1, performance_matrix.shape[1]))
        for i in range(0, performance_matrix.shape[1]):
            b_matrix[0][i] = B[i]
     
    # Algorithm       
    concordance_matrix, concordance_matrix_inv, concordance_vector, concordance_vector_inv = concordance_matrices_vectors(performance_matrix = performance_matrix, number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, B = b_matrix, P = p_vector, Q = q_vector, W = w_vector)    
    
    disconcordance_matrix, disconcordance_matrix_inv = discordance_matrices(performance_matrix = performance_matrix, number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, B = b_matrix, P = p_vector, V = v_vector)    
    
    credibility_vector, credibility_vector_inv = credibility_vectors(number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, concordance_matrix = concordance_matrix, concordance_matrix_inv = concordance_matrix_inv, concordance_vector = concordance_vector, concordance_vector_inv = concordance_vector_inv, disconcordance_matrix = disconcordance_matrix, disconcordance_matrix_inv = disconcordance_matrix_inv) 
    
    fuzzy_matrix = fuzzy_logic(number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, credibility_vector = credibility_vector, credibility_vector_inv = credibility_vector_inv, cut_level = cut_level)  
    
    classification = classification_algorithm(number_of_profiles = number_of_profiles, number_of_alternatives = number_of_alternatives, fuzzy_matrix = fuzzy_matrix, rule = rule, verbose = verbose)
    
    if (graph == True):
        plot_points(performance_matrix, classification)
    return classification

###############################################################################