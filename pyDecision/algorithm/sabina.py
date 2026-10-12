
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
    block_size:        int   = 128

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
    costs    = ("min", "cost", -1, False, "-")
    encoded  = []
    for item in values:
        if item in benefits:
            encoded.append(1.0)
        elif item in costs:
            encoded.append(-1.0)
        else:
            raise ValueError(f"Invalid criterion direction: {item!r}")
    return np.asarray(encoded, dtype = float)

###############################################################################

# Function: Performance Matrix Normalization
def quantile_normalize(X, directions, config):
    X        = np.asarray(X, dtype = float)
    ql       = np.quantile(X, config.qlow,  axis = 0)
    qh       = np.quantile(X, config.qhigh, axis = 0)
    full_lo  = np.min(X, axis = 0)
    full_hi  = np.max(X, axis = 0)
    denom    = qh - ql
    constant = (full_hi - full_lo) <= config.eps
    Z        = np.zeros_like(X, dtype = float)
    mode     = []
    for j in range(X.shape[1]):
        if constant[j]:
            Z[:, j] = 0.5
            mode.append('constant')
        elif denom[j] > config.eps:
            Z[:, j] = np.clip((X[:, j] - ql[j]) / denom[j], 0.0, 1.0)
            mode.append('quantile')
        else:
            # Collapsed interior quantiles: empirical midrank/ECDF transform
            vals  = X[:, j]
            order = np.argsort(vals, kind = 'mergesort')
            ranks = np.empty(len(vals), dtype = float)
            sv    = vals[order]
            start = 0
            while start < len(vals):
                end = start + 1
                while end < len(vals) and sv[end] == sv[start]:
                    end += 1
                ranks[order[start:end]] = 0.5*((start + 1) + end)
                start = end
            Z[:, j] = (ranks - 0.5) / len(vals)
            mode.append('ecdf')
    cost_mask       = directions < 0
    Z[:, cost_mask] = 1.0 - Z[:, cost_mask]
    return Z, ql, qh, np.asarray(mode, dtype = object), constant

# Function: Pairwise Distance (row blocks bound the temporary arrays to block_size x n x m)
def pairwise_weighted_distance(Z, weights, config):
    n = Z.shape[0]
    D = np.zeros((n, n), dtype = float)
    for first in range(0, n, config.block_size):
        last           = min(first + config.block_size, n)
        diff           = Z[first:last, None, :] - Z[None, :, :]
        D[first:last]  = np.sqrt(np.sum(weights[None, None, :] * diff**2, axis = 2))
    return D

# Function: Local Density Scales
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

# Function: Criterion Base Bandwidths
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

# Function: Net Flow (chunk-vectorized; weights are the strictly positive active weights)
def net_flow(Z, weights, config):
    n      = Z.shape[0]
    D      = pairwise_weighted_distance(Z, weights, config)
    rho    = local_scales(D, config)
    base_h = base_bandwidths(Z, config)
    log_w  = np.log(weights)
    P      = np.zeros((n, n), dtype = float)
    for first in range(0, n, config.block_size):
        last  = min(first + config.block_size, n)
        xi    = Z[first:last, None, :]
        xk    = Z[None, :, :]
        h     = base_h[None, None, :] * ((rho[first:last, None] + rho[None, :]) / 2.0)[:, :, None]
        # Criterion-wise smooth preference, Eq. (11)
        prefs = np.clip(norm.cdf((xi - xk - config.q_threshold) / h), 1e-300, 1.0)
        logp  = np.log(prefs)
        # Generalized-mean concordance in log space, Eq. (12)
        if abs(config.eta) < 1e-10:
            C = np.exp(np.sum(weights * logp, axis = -1))
        else:
            C = np.exp(logsumexp(log_w + config.eta * logp, axis = -1) / config.eta)
        # Smooth veto, Eqs. (14)-(17)
        loss   = np.maximum(0.0, xk - xi)
        logits = (config.veto_h_multiplier * h - loss) / (config.veto_s_multiplier * h)
        V      = np.exp(np.sum(weights * np.log(np.clip(expit(logits), 1e-300, 1.0)), axis = -1))
        P[first:last] = C * V
    np.fill_diagonal(P, 0.0)
    f = (P.sum(axis = 1) - P.sum(axis = 0)) / max(1, n - 1)
    return f, P, D, rho, base_h

###############################################################################

# Function: Fit and Score
def fit_score(X, weights, directions, config):
    X = np.asarray(X, dtype = float)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty 2D decision matrix.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values only.")
    if not np.all(np.isfinite([config.eta, config.q_threshold, config.base_h_scale, config.veto_h_multiplier, config.veto_s_multiplier, config.rho_floor, config.eps])):
        raise ValueError("Configuration parameters must be finite.")
    if not (0 <= config.qlow < config.qhigh <= 1):
        raise ValueError("Require 0 <= qlow < qhigh <= 1.")
    if config.q_threshold < 0:
        raise ValueError("q_threshold must be nonnegative.")
    if config.base_h_scale <= 0 or config.veto_h_multiplier <= 0 or config.veto_s_multiplier <= 0:
        raise ValueError("base_h_scale, veto_h_multiplier and veto_s_multiplier must be positive.")
    if config.rho_floor <= 0 or config.eps <= 0:
        raise ValueError("rho_floor and eps must be positive.")
    for name in ("local_k", "block_size"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    w = _normalize_weights(weights)
    if len(w) != X.shape[1]:
        raise ValueError("weights must match the number of criteria.")
    d                                = _encode_directions(directions, n_criteria = X.shape[1])
    Z, ql, qh, norm_mode, constant   = quantile_normalize(X, d, config)
    active                           = (~constant) & (w > 0)
    n                                = X.shape[0]
    if not np.any(active):
        flow   = np.zeros(n, dtype = float)
        P      = np.zeros((n, n), dtype = float)
        D      = np.zeros((n, n), dtype = float)
        rho    = np.ones(n, dtype = float)
        base_h = np.array([], dtype = float)
    else:
        wa                       = _normalize_weights(w[active])
        flow, P, D, rho, base_h  = net_flow(Z[:, active], wa, config)
    return {
                "normalized":         Z,
                "qlow":               ql,
                "qhigh":              qh,
                "normalization_mode": norm_mode,
                "active_criteria":    active,
                "score":              flow,
                "pairwise_pref":      P,
                "distance":           D,
                "local_scale":        rho,
                "base_h":             base_h,
                "weights":            w,
            }

# Function: Ranked Labels
def rank_scores(scores, labels):
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    order = np.argsort(-np.asarray(scores), kind = "mergesort")
    return [(labels[i], float(scores[i]), int(r + 1)) for r, i in enumerate(order)]

###############################################################################

# Function: SABINA (Smooth Adaptive Bandwidth Integrated Net-flow Aggregation)
def sabina_method(X, weights, criteria_type, labels = None, config = config):
    res          = fit_score(X = X, weights = weights, directions = criteria_type, config = config)
    res["order"] = np.argsort(-res["score"], kind = "mergesort") + 1
    if labels is not None:
        res["ranking"]         = rank_scores(res["score"], labels)
        res["predicted_order"] = [item[0] for item in res["ranking"]]
    return res["order"], res["score"], res

###############################################################################
