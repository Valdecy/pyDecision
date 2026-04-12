
###############################################################################

# Required Libraries
import numpy as np

from dataclasses import dataclass
from scipy.stats import norm

###############################################################################

# Function: Default Values
@dataclass(frozen = True)
class config:
    qlow:              float = 0.10
    qhigh:             float = 0.90
    eta:               float = 1.0
    q_threshold:       float = 0.0
    base_h_scale:      float = 0.35
    veto_h_multiplier: float = 1.5
    veto_s_multiplier: float = 0.5
    eps:               float = 1e-12
    local_k:           int   = 3
#    use_graph:         bool  = False
#    graph_k:           int   = 3
#    graph_lambda:      float = 1.0

# Function: Weight Normalization
def _normalize_weights(weights):
    w = np.asarray(weights, dtype = float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D sequence.")
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return w / total

# Function: Criteria Type
def _encode_directions(directions, n_criteria):
    d = np.asarray([1.0 if item in ("max", "benefit", 1, True, "+") else -1.0 for item in directions], dtype = float)
    if d.ndim != 1 or len(d) != n_criteria:
        raise ValueError("weights and directions must have the same length.")
    return d

###############################################################################

# Function: Perfomance Matrix Normalization
def quantile_normalize(X, directions, config):
    ql              = np.quantile(X, config.qlow,  axis = 0)
    qh              = np.quantile(X, config.qhigh, axis = 0)
    Z               = (X - ql) / (qh - ql + config.eps)
    Z               = np.clip(Z, 0.0, 1.0)
    cost_mask       = directions < 0
    Z[:, cost_mask] = 1.0 - Z[:, cost_mask]
    return Z, ql, qh

# Function:  Pairwise Distance
def pairwise_weighted_distance(Z, weights):
    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt(np.sum(weights[None, None, :] * diff**2, axis = 2))

# Function: Pairwise Preference
def pairwise_preference(Z, i, k, rho, base_h, weights, config):
    h       = adaptive_bandwidth(base_h, rho[i], rho[k])
    prefs   = criterion_preference(Z[i], Z[k], h, config)
    c_value = concordance(prefs, weights, config)
    v_value = veto_factor(Z[i], Z[k], h, weights, config)
    return c_value * v_value

# Function: Scales
def local_scales(D, config):
    n   = D.shape[0]
    k   = min(config.local_k, max(1, n - 1))
    rho = np.zeros(n, dtype = float)
    for i in range(0, n):
        vals   = np.sort(D[i][np.arange(n) != i])
        rho[i] = np.mean(vals[:k]) if len(vals) else 1.0
    positive = rho[rho > 0]
    med      = np.median(positive) if len(positive) > 0 else 1.0
    return np.maximum(rho / (med + config.eps), 0.25)

# Function: Bandwith
def base_bandwidths(Z, config):
    h = np.zeros(Z.shape[1], dtype = float)
    for j in range(0, Z.shape[1]):
        if Z.shape[0] > 1:
            std_j = np.std(Z[:, j], ddof = 1)
            iqr_j = np.subtract(*np.percentile(Z[:, j], [75, 25]))
        else:
            std_j = 0.0
            iqr_j = 0.0
        scale = max(std_j, iqr_j / 1.349, 0.05)
        h[j]  = config.base_h_scale * scale + 1e-6
    return h

# Function: Bandwith
def adaptive_bandwidth(base_h, rho_i, rho_k):
    factor = max((rho_i + rho_k) / 2.0, 0.25)
    return base_h * factor

# Function: Preferences
def criterion_preference(x, y, h, config):
    z = (x - y - config.q_threshold) / (h + config.eps)
    return norm.cdf(z)

# Function: Concordance
def concordance(prefs, weights, config):
    prefs = np.clip(prefs, 1e-12, 1.0)
    eta   = config.eta
    if abs(eta) < 1e-10:
        return float(np.exp(np.sum(weights * np.log(prefs))))
    return float(np.sum(weights * (prefs**eta)) ** (1.0 / eta))

# Function: Veto
def veto_factor(x, y, h, weights, config):
    loss   = np.maximum(0.0, y - x)
    v      = config.veto_h_multiplier * h
    s      = np.maximum(config.veto_s_multiplier * h, 1e-6)
    logits = (v - loss) / s
    vals   = 1.0 / (1.0 + np.exp(-logits))
    vals   = np.clip(vals, 1e-12, 1.0)
    return float(np.exp(np.sum(weights * np.log(vals))))

# Function:  Flow
def net_flow(Z, weights, config):
    n      = Z.shape[0]
    D      = pairwise_weighted_distance(Z, weights)
    rho    = local_scales(D, config)
    base_h = base_bandwidths(Z, config)
    P      = np.zeros((n, n), dtype = float)
    for i in range(0, n):
        for k in range(0, n):
            if i == k:
                continue
            P[i, k] = pairwise_preference(Z, i, k, rho, base_h, weights, config)
    f = (P.sum(axis=1) - P.sum(axis=0)) / max(1, n - 1)
    return f, P, D, rho, base_h

###############################################################################

# Graph Functions
 
#def estimate_graph_sigma(D, config):
#    n         = D.shape[0]
#    ksig      = min(config.local_k, max(1, n - 1))
#    ksig_vals = []
#    for i in range(0, n):
#        vals = np.sort(D[i][np.arange(n) != i])
#        ksig_vals.extend(vals[:ksig].tolist())
#    positive_distances = D[D > 0]
#    fallback           = float(np.median(positive_distances)) if positive_distances.size else 1.0
#    sigma              = float(np.median(ksig_vals)) if len(ksig_vals) > 0 else fallback
#    return max(sigma, 1e-6)
 
#def build_sparse_similarity_graph(D, config):
#    n         = D.shape[0]
#    sigma     = estimate_graph_sigma(D, config)
#    S_dense   = np.exp(-(D**2) / (2 * sigma**2 + config.eps))
#    np.fill_diagonal(S_dense, 0.0)
#    k         = min(config.graph_k, max(1, n - 1))
#    S         = np.zeros_like(S_dense)
#    neighbors = []
#    for i in range(0, n):
#        idx = np.argsort(D[i])[1 : k + 1]
#        neighbors.append(set(idx.tolist()))
#    for i in range(0, n):
#        for j in range(0, n):
#            if i != j and (j in neighbors[i] or i in neighbors[j]):
#                S[i, j] = S_dense[i, j]
#    S = np.maximum(S, S.T)
#    return S, sigma

#def apply_graph_regularization(base_scores, Z, weights, config):
#    n = len(base_scores)
#    if not config.use_graph or config.graph_lambda <= 0 or n <= 2:
#        return base_scores.copy(), None, None
#    D        = pairwise_weighted_distance(Z, weights)
#    S, sigma = build_sparse_similarity_graph(D, config)
#    degree   = np.sum(S, axis = 1)
#    L        = np.diag(degree) - S
#    A        = np.eye(n) + config.graph_lambda * L
#    z        = np.linalg.solve(A, base_scores)
#    return z, S, sigma

###############################################################################

# Function: 
def fit_score(X, weights, directions, config): #, graph_regularizer = False
    #cfg                     = replace(config or Config(), use_graph = graph_regularizer)
    cfg                     = config
    w                       = _normalize_weights(weights)
    d                       = _encode_directions(directions, n_criteria = X.shape[1])
    Z, ql, qh               = quantile_normalize(X, d, cfg)
    flow, P, D, rho, base_h = net_flow(Z, w, cfg)
    score                   = flow.copy()
    #score, S, sigma         = apply_graph_regularization(flow, Z, w, cfg)
    
    return {
                "normalized":    Z,
                "qlow":          ql,
                "qhigh":         qh,
                "score":         score,
                "pairwise_pref": P,
                "distance":      D,
                "local_scale":   rho,
                "base_h":        base_h,
                "weights":       w,
                #"config":        cfg,
                #"similarity":    S,
                #"graph_sigma":   sigma,
            }

# Function: Scores
def rank_scores(scores, labels):
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    order = np.argsort(scores)[::-1]
    return [(labels[i], float(scores[i]), int(r + 1)) for r, i in enumerate(order)]

###############################################################################

# Function: # Function: SABINA (Smooth Adaptive Bandwidth Integrated Net-flow Aggregation)
def sabina_method(X, weights, criteria_type, labels = None, config = config):
    res = fit_score(X = X, weights = weights, directions = criteria_type, config = config)
    if labels is not None:
        res["ranking"]         = rank_scores(res["score"], labels)
        res["predicted_order"] = [item[0] for item in res["ranking"]]
        res["order"]           = np.argsort(res['score'])[::-1] + 1
    return res['order'], res['score'], res

###############################################################################

