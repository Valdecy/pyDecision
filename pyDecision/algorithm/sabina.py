
###############################################################################

# Required Libraries
import numpy as np

from dataclasses import dataclass
from scipy.stats import norm
from scipy.special import expit, logsumexp

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
    rho_floor:         float = 0.25
#    use_graph:         bool  = False
#    graph_k:           int   = 3
#    graph_lambda:      float = 1.0

# Function: Weight Normalization
def _normalize_weights(weights):
    w = np.asarray(weights, dtype = float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D sequence.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return w / total

# Function: Criteria Type
def _encode_directions(directions, n_criteria):
    try:
        values = list(directions)
    except TypeError as exc:
        raise ValueError("directions must be a sequence.") from exc
    if len(values) != n_criteria:
        raise ValueError("directions must match the number of criteria.")
    benefits = ("max", "benefit", 1, True, "+")
    costs = ("min", "cost", -1, False, "-")
    encoded = []
    for item in values:
        if item in benefits:
            encoded.append(1.0)
        elif item in costs:
            encoded.append(-1.0)
        else:
            raise ValueError(f"Invalid criterion direction: {item!r}")
    return np.asarray(encoded, dtype=float)

###############################################################################

# Function: Perfomance Matrix Normalization
def quantile_normalize(X, directions, config):
    X = np.asarray(X, dtype = float)
    ql = np.quantile(X, config.qlow, axis = 0)
    qh = np.quantile(X, config.qhigh, axis = 0)
    full_lo = np.min(X, axis = 0)
    full_hi = np.max(X, axis = 0)
    denom = qh - ql
    constant = (full_hi - full_lo) <= config.eps
    Z = np.zeros_like(X, dtype = float)
    mode = []
    for j in range(X.shape[1]):
        if constant[j]:
            Z[:, j] = 0.5
            mode.append('constant')
        elif denom[j] > config.eps:
            Z[:, j] = np.clip((X[:, j] - ql[j]) / denom[j], 0.0, 1.0)
            mode.append('quantile')
        else:
            vals = X[:, j]
            order = np.argsort(vals, kind = 'mergesort')
            ranks = np.empty(len(vals), dtype = float)
            sv = vals[order]
            start = 0
            while start < len(vals):
                end = start + 1
                while end < len(vals) and sv[end] == sv[start]:
                    end += 1
                mid = 0.5*((start + 1) + end)
                ranks[order[start:end]] = mid
                start = end
            Z[:, j] = (ranks - 0.5) / len(vals)
            mode.append('ecdf')
    cost_mask = directions < 0
    Z[:, cost_mask] = 1.0 - Z[:, cost_mask]
    return Z, ql, qh, np.asarray(mode, dtype = object), constant

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
    return np.maximum(rho / med, config.rho_floor)

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
        scale = max(std_j, iqr_j / 1.349)
        h[j]  = config.base_h_scale * scale + 1e-6
    return h

# Function: Bandwith
def adaptive_bandwidth(base_h, rho_i, rho_k):
    factor = (rho_i + rho_k) / 2.0
    return base_h * factor

# Function: Preferences
def criterion_preference(x, y, h, config):
    z = (x - y - config.q_threshold) / h
    return norm.cdf(z)

# Function: Concordance
def concordance(prefs, weights, config):
    prefs = np.clip(prefs, 1e-300, 1.0)
    eta   = config.eta
    if abs(eta) < 1e-10:
        return float(np.exp(np.sum(weights * np.log(prefs))))
    # Log-space power mean avoids underflow for negative eta.
    positive = weights > 0
    return float(np.exp(logsumexp(np.log(weights[positive]) + eta * np.log(prefs[positive])) / eta))

# Function: Veto
def veto_factor(x, y, h, weights, config):
    loss   = np.maximum(0.0, y - x)
    v      = config.veto_h_multiplier * h
    s      = config.veto_s_multiplier * h
    logits = (v - loss) / s
    vals   = expit(logits)
    vals   = np.clip(vals, 1e-300, 1.0)
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
#    return sigma
 
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
    cfg = config
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty 2D decision matrix.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values only.")
    if not (0 <= cfg.qlow < cfg.qhigh <= 1):
        raise ValueError("Require 0 <= qlow < qhigh <= 1.")
    if cfg.base_h_scale <= 0 or cfg.veto_h_multiplier < 0 or cfg.veto_s_multiplier <= 0:
        raise ValueError("Invalid bandwidth/veto parameters.")
    if cfg.local_k < 1 or cfg.rho_floor <= 0 or cfg.eps <= 0:
        raise ValueError("local_k, rho_floor, and eps must be positive.")
    if not np.all(np.isfinite([cfg.eta, cfg.q_threshold, cfg.base_h_scale, cfg.veto_h_multiplier, cfg.veto_s_multiplier, cfg.rho_floor, cfg.eps])):
        raise ValueError("Configuration parameters must be finite.")
    w = _normalize_weights(weights)
    if len(w) != X.shape[1]:
        raise ValueError("weights must match the number of criteria.")
    d = _encode_directions(directions, n_criteria = X.shape[1])
    Z, ql, qh, norm_mode, constant = quantile_normalize(X, d, cfg)
    active = ~constant
    active &= w > 0
    if not np.any(active):
        n = X.shape[0]
        flow = np.zeros(n, dtype = float)
        P = np.zeros((n, n), dtype = float)
        D = np.zeros((n, n), dtype = float)
        rho = np.ones(n, dtype = float)
        base_h = np.array([], dtype = float)
    else:
        Za = Z[:, active]
        wa = _normalize_weights(w[active])
        flow, P, D, rho, base_h = net_flow(Za, wa, cfg)
    score = flow.copy()
    return {
                "normalized":    Z,
                "qlow":          ql,
                "qhigh":         qh,
                "normalization_mode": norm_mode,
                "active_criteria": active,
                "score":         score,
                "pairwise_pref": P,
                "distance":      D,
                "local_scale":   rho,
                "base_h":        base_h,
                "weights":       w,
            }

# Function: Scores
def rank_scores(scores, labels):
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    return [(labels[i], float(scores[i]), int(r + 1)) for r, i in enumerate(order)]

###############################################################################

# Function: SABINA (Smooth Adaptive Bandwidth Integrated Net-flow Aggregation)
def sabina_method(X, weights, criteria_type, labels = None, config = config):
    """Compute SABINA scores and a 1-based ranked list of alternative indices.

    Returns (order, scores, diagnostics). Higher scores are preferred.
    ``config`` accepts the class ``config`` or a ``config(...)`` instance.
    """
    res = fit_score(X = X, weights = weights, directions = criteria_type, config = config)
    res["order"] = np.argsort(-res["score"], kind="mergesort") + 1
    if labels is not None:
        res["ranking"]         = rank_scores(res["score"], labels)
        res["predicted_order"] = [item[0] for item in res["ranking"]]
    return res['order'], res['score'], res

###############################################################################

