###############################################################################

# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from scipy.stats import rankdata

###############################################################################

from pyDecision.algorithm.bwm          import bw_method
from pyDecision.algorithm.bwm_s        import simplified_bw_method
from pyDecision.algorithm.cilos        import cilos_method
from pyDecision.algorithm.critic       import critic_method
from pyDecision.algorithm.entropy      import entropy_method
from pyDecision.algorithm.fucom        import fucom_method
from pyDecision.algorithm.idocriw      import idocriw_method
from pyDecision.algorithm.merec        import merec_method
from pyDecision.algorithm.psi_m        import mpsi_method
from pyDecision.algorithm.roc          import roc_method
from pyDecision.algorithm.rrw          import rrw_method
from pyDecision.algorithm.rsw          import rsw_method
from pyDecision.algorithm.seca         import seca_method

from pyDecision.algorithm.fuzzy_bwm    import fuzzy_bw_method
from pyDecision.algorithm.fuzzy_critic import fuzzy_critic_method
from pyDecision.algorithm.fuzzy_fucom  import fuzzy_fucom_method
from pyDecision.algorithm.fuzzy_merec  import fuzzy_merec_method

from pyDecision.algorithm.aras         import aras_method
from pyDecision.algorithm.borda        import borda_method
from pyDecision.algorithm.cocoso       import cocoso_method
from pyDecision.algorithm.codas        import codas_method
from pyDecision.algorithm.copeland     import copeland_method
from pyDecision.algorithm.copras       import copras_method
from pyDecision.algorithm.cradis       import cradis_method
from pyDecision.algorithm.edas         import edas_method
from pyDecision.algorithm.gra          import gra_method
from pyDecision.algorithm.mabac        import mabac_method
from pyDecision.algorithm.macbeth      import macbeth_method
from pyDecision.algorithm.mairca       import mairca_method
from pyDecision.algorithm.mara         import mara_method
from pyDecision.algorithm.marcos       import marcos_method
from pyDecision.algorithm.maut         import maut_method
from pyDecision.algorithm.moora        import moora_method
from pyDecision.algorithm.moosra       import moosra_method
from pyDecision.algorithm.multimoora   import multimoora_method
from pyDecision.algorithm.ocra         import ocra_method
from pyDecision.algorithm.oreste       import oreste_method
from pyDecision.algorithm.piv          import piv_method
from pyDecision.algorithm.p_ec         import ec_promethee
from pyDecision.algorithm.p_ii         import promethee_ii
from pyDecision.algorithm.p_iv         import promethee_iv
from pyDecision.algorithm.psi          import psi_method
from pyDecision.algorithm.rov          import rov_method
from pyDecision.algorithm.rafsi        import rafsi_method
from pyDecision.algorithm.saw          import saw_method
from pyDecision.algorithm.spotis       import spotis_method
from pyDecision.algorithm.todim        import todim_method
from pyDecision.algorithm.topsis       import topsis_method
from pyDecision.algorithm.vikor        import vikor_method
from pyDecision.algorithm.waspas       import waspas_method
from pyDecision.algorithm.wisp         import wisp_method

from pyDecision.algorithm.fuzzy_aras   import fuzzy_aras_method
from pyDecision.algorithm.fuzzy_copras import fuzzy_copras_method
from pyDecision.algorithm.fuzzy_edas   import fuzzy_edas_method
from pyDecision.algorithm.fuzzy_moora  import fuzzy_moora_method
from pyDecision.algorithm.fuzzy_ocra   import fuzzy_ocra_method
from pyDecision.algorithm.fuzzy_topsis import fuzzy_topsis_method
from pyDecision.algorithm.fuzzy_vikor  import fuzzy_vikor_method
from pyDecision.algorithm.fuzzy_waspas import fuzzy_waspas_method
 
###############################################################################

# Function: Tranpose Dictionary
def transpose_dict(rank_count_dict):
    transposed_dict = {}
    list_length     = len(next(iter(rank_count_dict.values())))
    for i in range(list_length):
        transposed_dict[i+1] = [values[i] for values in rank_count_dict.values()]
    return transposed_dict

# Function: Plot Ranks
def plot_rank_freq(ranks, size_x = 8, size_y = 10):
    alternative_labels = [f'a{i+1}' for i in range(ranks.shape[0])]
    rank_count_dict    = {i+1: [0]*ranks.shape[0] for i in range(0, ranks.shape[0])}
    for i in range(0, ranks.shape[0]):
        for j in range(0, ranks.shape[1]):
            rank = int(ranks.iloc[i, j])
            rank_count_dict[i+1][rank-1] = rank_count_dict[i+1][rank-1] + 1
    rank_count_dict = transpose_dict(rank_count_dict)
    fig, ax         = plt.subplots(figsize = (size_x, size_y))
    colors          = plt.cm.get_cmap('tab20', ranks.shape[0])
    bottom          = np.zeros(len(alternative_labels))
    for rank, counts in rank_count_dict.items():
        bars   = ax.barh(alternative_labels, counts, left = bottom, color = colors(rank-1))
        bottom = bottom + counts  
        for rect, c in zip(bars, counts):
            if (c > 0): 
                width = rect.get_width()
                ax.text(width/2 + rect.get_x(), rect.get_y() + rect.get_height() / 2, f"r{rank} ({c})", ha = 'center', va = 'center', color = 'black')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer = True))
    ax.tick_params(axis = 'y', which = 'both', pad = 25) 
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Alternative')
    ax.set_title('Rank Frequency per Alternative')
    plt.show()
    return

# Function: Calc & Plot Correlation. correlation_method = 'kendall', correlation_method = 'spearman', correlation_method = 'pearson'
def corr_viz(df, correlation_method = 'kendall', size = 10, font_size = 10, graph = True):
    corr_df = df.corr(method = correlation_method)
    if (graph == True):
        plt.figure(figsize = (size, size))
        cax = plt.matshow(corr_df, cmap = 'YlGnBu', fignum = 1)
        plt.colorbar(cax)
        plt.xticks(np.arange(len(corr_df.columns)), corr_df.columns, rotation = 90)
        plt.yticks(np.arange(len(corr_df.columns)), corr_df.columns)
        for i in range(len(corr_df.columns)):
                    for j in range(len(corr_df.columns)):
                        plt.gca().add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill = False, edgecolor = 'black', lw = 0.2))
        for (i, j), z in np.ndenumerate(corr_df):
            plt.text(j, i, '{:0.2f}'.format(z), ha = 'center', va = 'center', fontsize = font_size, bbox = dict(boxstyle = 'round', facecolor = 'white', edgecolor = '0.1'))
    return corr_df

###############################################################################

# Function: Compare Weights Crisp
def compare_weights(dataset, criterion_type, custom_methods = [], custom_weigths = [], methods_list = [], mic = [], lic = [], criteria_priority = [], criteria_rank = [], alpha = 0.5, beta = 3):
    if ('all' in methods_list):
        methods_list = ['bwm', 'bwm_s', 'cilos', 'critic', 'entropy', 'fucom', 'idocriw', 'merec', 'mpsi', 'roc', 'rrw', 'rsw', 'seca']
    if (len(custom_methods) > 0):
        methods_list = custom_methods + methods_list 
    X       = np.zeros((dataset.shape[1], len(methods_list)))
    j       = 0
    for i in range(0, len(custom_weigths)):
        X[:,j] = custom_weigths[i]
        j      = j + 1 
        print(custom_methods[i], ': Done!')
    for method in methods_list:
        if (method == 'bwm' or method == 'all'):
            w      = bw_method(mic, lic, False)
            X[:,j] = w
            j      = j + 1
            print('BWM: Done!')
        if (method == 'bwm_s' or method == 'all'):
            _, w   = simplified_bw_method(mic, lic, alpha, False)
            X[:,j] = w
            j      = j + 1
            print('Simplified BWM: Done!')
        if (method == 'cilos' or method == 'all'):
            w      = cilos_method(dataset, criterion_type)
            X[:,j] = w
            j      = j + 1
            print('CILOS: Done!')
        if (method == 'critic' or method == 'all'):
            w      = critic_method(dataset, criterion_type)
            X[:,j] = w
            j      = j + 1
            print('CRITIC: Done!')
        if (method == 'entropy' or method == 'all'):
            w      = entropy_method(dataset, criterion_type)
            X[:,j] = w
            j      = j + 1
            print('Entropy: Done!')
        if (method == 'fucom' or method == 'all'):
            w      = fucom_method(criteria_rank, criteria_priority, True, False)
            X[:,j] = w
            j      = j + 1
            print('FUCOM: Done!')
        if (method == 'idocriw' or method == 'all'):
            w      = idocriw_method(dataset, criterion_type, False)
            X[:,j] = w
            j      = j + 1
            print('IDOCRIW: Done!')
        if (method == 'merec' or method == 'all'):
            w      = merec_method(dataset, criterion_type)
            X[:,j] = w
            j      = j + 1
            print('MEREC: Done!')
        if (method == 'mpsi' or method == 'all'):
            w      = mpsi_method(dataset, criterion_type)
            X[:,j] = w
            j      = j + 1
            print('MPSI: Done!')
        if (method == 'roc' or method == 'all'):
            w      = roc_method(criteria_rank)
            X[:,j] = w
            j      = j + 1
            print('ROC: Done!')
        if (method == 'rrw' or method == 'all'):
            w      = rrw_method(criteria_rank)
            X[:,j] = w
            j      = j + 1
            print('RRW: Done!')
        if (method == 'rsw' or method == 'all'):
            w      = rsw_method(criteria_rank)
            X[:,j] = w
            j      = j + 1
            print('RSW: Done!')
        if (method == 'seca' or method == 'all'):
            w      = seca_method(dataset, criterion_type, beta)
            X[:,j] = w
            j      = j + 1
            print('SECA: Done!')
    X = pd.DataFrame(X, index = ['g'+str(i+1) for i in range(0, X.shape[0])], columns = methods_list)    
    return X

# Function: Compare Weights Fuzzy
def compare_weights_fuzzy(dataset = [], criterion_type = [], custom_methods = [], custom_weigths = [], methods_list = [], criteria_priority = [], criteria_rank = [], mic = [], lic = [], eps_penalty = 1, n_starts = 250):
    if ('all' in methods_list):
        methods_list = ['fuzzy_bwm', 'fuzzy_critic', 'fuzzy_fucom', 'fuzzy_merec']
    if (len(custom_methods) > 0):
        methods_list = custom_methods + methods_list 
    X = np.zeros((np.max((len(criterion_type), len(criteria_rank), len(mic))), len(methods_list)))
    j = 0
    for i in range(0, len(custom_weigths)):
        X[:,j] = custom_weigths[i]
        j      = j + 1 
        print(custom_methods[i], ': Done!')
    for method in methods_list:
        if (method == 'fuzzy_bwm' or method == 'all'):
            _, _, _, w = fuzzy_bw_method(mic, lic, eps_penalty, False)
            X[:,j]     = w
            j          = j + 1
            print('Fuzzy BWM: Done!')
        if (method == 'fuzzy_critic' or method == 'all'):
            w          = fuzzy_critic_method(dataset, criterion_type)
            X[:,j]     = w
            j          = j + 1
            print('Fuzzy CRITIC: Done!')
        if (method == 'fuzzy_fucom' or method == 'all'):
            _, w       = fuzzy_fucom_method(criteria_rank, criteria_priority, n_starts, True, False)
            X[:,j]     = w
            j          = j + 1
            print('Fuzzy FUCOM: Done!')
        if (method == 'fuzzy_merec' or method == 'all'):
            w          = fuzzy_merec_method(dataset, criterion_type)
            X[:,j]     = w
            j          = j + 1
            print('Fuzzy MEREC: Done!')
    X = pd.DataFrame(X, index = ['g'+str(i+1) for i in range(0, X.shape[0])], columns = methods_list) 
    return X
    
# Function: Compare Ranks Crisp
def compare_ranks_crisp(dataset, weights, criterion_type, utility_functions = [], custom_methods = [], custom_ranks = [], methods_list = [], L = 0.5, lmbd = 0.02, epsilon = 0.5, step_size = 1, teta = 1, strategy_coefficient = 0.5, Q = [], S = [], P = [], F = [], custom_sets = [], iterations = 1000, lambda_value = 0.5, alpha = 0.4, s_min = [], s_max = [], ideal = [], anti_ideal = [], n_i = 1, n_k = 6):
    if ('all' in methods_list):
        methods_list = ['aras', 'borda', 'cocoso', 'codas', 'copeland', 'copras', 'cradis', 'edas', 'gra', 'mabac', 'macbeth', 'mairca', 'mara', 'marcos', 'maut', 'moora', 'moosra', 'multimoora', 'ocra', 'oreste', 'piv', 'promethee_ii', 'promethee_iv', 'ec_promethee', 'psi', 'rafsi', 'rov', 'saw', 'spotis', 'todim', 'topsis', 'vikor', 'wsm', 'wpm', 'waspas', 'wisp', 'simple wisp']
    if (len(custom_methods) > 0):
        methods_list = custom_methods + methods_list 
    graph   = False
    verbose = False
    X       = np.zeros((dataset.shape[0], len(methods_list)))
    j       = 0
    for i in range(0, len(custom_ranks)):
        X[:,j] = custom_ranks[i]
        j      = j + 1 
        print(custom_methods[i], ': Done!')
    for method in methods_list:
        if (method == 'aras' or method == 'all'):
            rank   = aras_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('ARAS: Done!')
        if (method == 'borda' or method == 'all'):
            rank   = borda_method(dataset, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Borda: Done!')
        if (method == 'cocoso' or method == 'all'):
            rank   = cocoso_method(dataset, criterion_type, weights, L, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('CoCoSo: Done!')
        if (method == 'codas' or method == 'all'):
            rank   = codas_method(dataset, weights, criterion_type, lmbd, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('CODAS: Done!')
        if (method == 'copras' or method == 'all'):
            rank   = copras_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('COPRAS: Done!')
        if (method == 'copeland' or method == 'all'):
            rank   = copeland_method(dataset, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('COPELAND: Done!')
        if (method == 'cradis' or method == 'all'):
            rank   = cradis_method(dataset, criterion_type, weights, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('CRADIS: Done!')
        if (method == 'edas' or method == 'all'):
            rank   = edas_method(dataset, criterion_type, weights, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('EDAS: Done!')
        if (method == 'gra' or method == 'all'):
            rank   = gra_method(dataset, criterion_type, weights, epsilon, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('GRA: Done!')
        if (method == 'mabac' or method == 'all'):
            rank   = mabac_method(dataset, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('MABAC: Done!')
        if (method == 'macbeth' or method == 'all'):
            rank   = macbeth_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('MACBETH: Done!')
        if (method == 'mairca' or method == 'all'):
            rank   = mairca_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('MAIRCA: Done!')
        if (method == 'mara' or method == 'all'):
            rank   = mara_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('MARA: Done!')
        if (method == 'marcos' or method == 'all'):
            rank   = marcos_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('MARCOS: Done!')
        if (method == 'maut' or method == 'all'):
            rank   = maut_method(dataset, weights, criterion_type, utility_functions, step_size, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('MAUT: Done!')
        if (method == 'moora' or method == 'all'):
            rank   = moora_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('MOORA: Done!')
        if (method == 'moosra' or method == 'all'):
            rank   = moosra_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('MOOSRA: Done!')
        if (method == 'multimoora' or method == 'all'):
            _, _, rank = multimoora_method(dataset, criterion_type, graph)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('MULTIMOORA: Done!')
        if (method == 'ocra' or method == 'all'):
            rank   = ocra_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('OCRA: Done!')
        if (method == 'oreste' or method == 'all'):
            rank   = oreste_method(dataset, weights, criterion_type, alpha, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('ORESTE: Done!')
        if (method == 'piv' or method == 'all'):
            rank   = piv_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('PIV: Done!')
        if (method == 'promethee_ii' or method == 'all'):
            data_adj = np.copy(dataset)
            for k in range(0, dataset.shape[1]):
                if (criterion_type[k] == 'min'):
                    data_adj[:,k] = 1/data_adj[:,k]
            rank   = promethee_ii(data_adj, weights, Q, S, P, F, False, 0, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('PROMETHEE II: Done!')
        if (method == 'promethee_iv' or method == 'all'):
            data_adj = np.copy(dataset)
            for k in range(0, dataset.shape[1]):
                if (criterion_type[k] == 'min'):
                    data_adj[:,k] = 1/data_adj[:,k]
            rank   = promethee_iv(data_adj, weights, Q, S, P, F, False, 0.001, 0, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('PROMETHEE IV: Done!')
        if (method == 'ec_promethee' or method == 'all'):
            _, _, rank, _ = ec_promethee(dataset, criterion_type, custom_sets, Q, S, P, F, iterations, verbose)
            X[:,j] = np.sum(rank, axis = 0)
            j      = j + 1
            print('EC PROMETHEE: Done!')
        if (method == 'psi' or method == 'all'):
            rank   = psi_method(dataset, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('PSI: Done!')
        if (method == 'rov' or method == 'all'):
            rank   = rov_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('ROV: Done!')
        if (method == 'rafsi' or method == 'all'):
            rank   = rafsi_method(dataset, weights, criterion_type, ideal, anti_ideal, n_i, n_k, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('RAFSI: Done!')
        if (method == 'saw' or method == 'all'):
            rank   = saw_method(dataset, criterion_type, weights, graph, verbose)
            X[:,j] = rank[:,1]
            j      = j + 1
            print('SAW: Done!')
        if (method == 'spotis' or method == 'all'):
            rank   = spotis_method(dataset, criterion_type, weights, s_min,s_max, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('SPOTIS: Done!')
        if (method == 'todim' or method == 'all'):
            rank   = todim_method(dataset, criterion_type, weights, teta, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('TODIM: Done!')
        if (method == 'topsis' or method == 'all'):
            rank   = topsis_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('TOPSIS: Done!')
        if (method == 'vikor' or method == 'all'):
            _, _, rank, _ = vikor_method(dataset, weights, criterion_type, strategy_coefficient, graph, verbose)
            X[:,j] = -rank[:,0]
            j      = j + 1
            print('VIKOR: Done!')
        if (method == 'wsm' or method == 'all'):
            w, _, _ = waspas_method(dataset, criterion_type, weights, lambda_value, graph)
            X[:,j] = w
            j      = j + 1
            print('WSM: Done!')
        if (method == 'wpm' or method == 'all'):
            _, w, _ = waspas_method(dataset, criterion_type, weights, lambda_value, graph)
            X[:,j] = w
            j      = j + 1
            print('WPM: Done!')
        if (method == 'waspas' or method == 'all'):
            _, _, w = waspas_method(dataset, criterion_type, weights, lambda_value, graph)
            X[:,j] = w
            j      = j + 1
            print('WASPAS: Done!')
        if (method == 'wisp' or method == 'all'):
            w      = wisp_method(dataset, criterion_type, weights, False, False, False)
            X[:,j] = w
            j      = j + 1
            print('WISP: Done!')
        if (method == 'simple wisp' or method == 'all'):
            w      = wisp_method(dataset, criterion_type, weights, True, False, False)
            X[:,j] = w
            j      = j + 1
            print('Simple WISP: Done!')
    ranked = np.zeros_like(X)
    for i in range(0, len(methods_list)):
        if (methods_list[i] in ['borda', 'cradis', 'mairca', 'mara', 'oreste', 'piv', 'spotis']):
            ranked[:, i] = X.shape[0] + 1 - rankdata(-X[:, i], method = 'max')
        else:
            ranked[:, i] = X.shape[0] + 1 - rankdata(X[:, i], method = 'max')
    X      = pd.DataFrame(X, index = ['a'+str(i+1) for i in range(0, X.shape[0])], columns = methods_list)    
    ranked = pd.DataFrame(ranked, index = ['a'+str(i+1) for i in range(0, X.shape[0])], columns = methods_list)
    return X, ranked

# Function: Compare Ranks Fuzzy
def compare_ranks_fuzzy(dataset, weights, criterion_type, custom_methods = [], custom_ranks = [], methods_list = [], strategy_coefficient = 0.5):
    if ('all' in methods_list):
        methods_list = ['fuzzy_aras', 'fuzzy_copras', 'fuzzy_edas', 'fuzzy_moora', 'fuzzy_ocra', 'fuzzy_topsis', 'fuzzy_vikor', 'fuzzy_wsm', 'fuzzy_wpm', 'fuzzy_waspas']
    if (len(custom_methods) > 0):
        methods_list = custom_methods + methods_list 
    graph   = False
    verbose = False
    X       = np.zeros((len(dataset), len(methods_list)))
    j       = 0
    for i in range(0, len(custom_ranks)):
        X[:,j] = custom_ranks[i]
        j      = j + 1 
        print(custom_methods[i], ': Done!')
    for method in methods_list:
        if (method == 'fuzzy_aras' or method == 'all'):
            rank   = fuzzy_aras_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Fuzzy ARAS: Done!')
        if (method == 'fuzzy_copras' or method == 'all'):
            rank   = fuzzy_copras_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Fuzzy COPRAS: Done!')
        if (method == 'fuzzy_edas' or method == 'all'):
            rank   = fuzzy_edas_method(dataset, criterion_type, weights, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Fuzzy EDAS: Done!')
        if (method == 'fuzzy_moora' or method == 'all'):
            rank   = fuzzy_moora_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Fuzzy MOORA: Done!')
        if (method == 'fuzzy_ocra' or method == 'all'):
            rank   = fuzzy_ocra_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Fuzzy OCRA: Done!')
        if (method == 'fuzzy_topsis' or method == 'all'):
            rank   = fuzzy_topsis_method(dataset, weights, criterion_type, graph, verbose)
            X[:,j] = rank
            j      = j + 1
            print('Fuzzy TOPSIS: Done!')
        if (method == 'fuzzy_vikor' or method == 'all'):
            _, _, rank, _ = fuzzy_vikor_method(dataset, weights, criterion_type, strategy_coefficient, graph, verbose)
            X[:,j] = -rank[:,0]
            j      = j + 1
            print('Fuzzy VIKOR: Done!')
        if (method == 'fuzzy_wsm' or method == 'all'):
            w, _, _ = fuzzy_waspas_method(dataset, criterion_type, weights, graph) 
            X[:,j] = w
            j      = j + 1
            print('Fuzzy WSM: Done!')
        if (method == 'fuzzy_wpm' or method == 'all'):
            _, w, _ = fuzzy_waspas_method(dataset, criterion_type, weights, graph)
            X[:,j] = w
            j      = j + 1
            print('Fuzzy WPM: Done!')
        if (method == 'fuzzy_waspas' or method == 'all'):
            _, _, w = fuzzy_waspas_method(dataset, criterion_type, weights, graph)
            X[:,j] = w
            j      = j + 1
            print('Fuzzy WASPAS: Done!')
    ranked = np.zeros_like(X)
    for i in range(0, X.shape[1]):
        ranked[:, i] = X.shape[0] + 1 - rankdata(X[:, i], method = 'max')
    X      = pd.DataFrame(X, index = ['a'+str(i+1) for i in range(0, X.shape[0])], columns = methods_list)    
    ranked = pd.DataFrame(ranked, index = ['a'+str(i+1) for i in range(0, X.shape[0])], columns = methods_list)
    return X, ranked

###############################################################################
