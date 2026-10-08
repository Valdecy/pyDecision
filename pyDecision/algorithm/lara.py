###############################################################################

# Required Libraries
import numpy as np
import networkx as nx
import matplotlib.cm       as cm
import matplotlib.colors   as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot   as plt

###############################################################################

# Function: Helper
def _safe_standardize(x, eps = 1e-12):
    x  = np.asarray(x, dtype = float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x), mu, sd
    return (x - mu) / sd, mu, sd

###############################################################################

# Function: Helper
def _labels(n, alt_labels):
    return alt_labels if alt_labels else [f"a{i+1}" for i in range(0, n)]

# Function: Graph
def plot_lara_graph(order, score, info, alt_labels = None, title = "Similarity Graph", figsize = (12.5, 8.5), seed = 42, node_scale = 1.0, savepath = None):
    score      = np.asarray(score, dtype = float)
    n          = len(score)
    S          = info["S_graph"]
    labels     = _labels(n, alt_labels)
    cmap_nodes = plt.colormaps["RdYlGn"]
    rank_pos   = np.empty(n, dtype = int)
    for r, a in enumerate(order):
        rank_pos[a - 1] = r
    node_cols = [cmap_nodes(1.0 - rank_pos[i] / max(n - 1, 1)) for i in range(0, n)]
    G         = nx.Graph()
    G.add_nodes_from(range(n))
    edges     = [(i, j, float(S[i, j])) for i in range(n) for j in range(i + 1, n) if S[i, j] > 1e-9]
    for i, j, w in edges:
        G.add_edge(i, j, weight = w)
    # Deterministic Fruchterman--Reingold layout for visualization only.
    # Edge affinities act as attractive weights; layout coordinates never enter LaRa.
    pos = nx.spring_layout(G, seed = seed, weight = "weight", k = 0.45, iterations = 500)
    for i in pos:
        pos[i][0] *= 0.88
    node_sizes = np.full(n, node_scale * 1900.0, dtype = float)
    if edges:
        ws             = np.array([w for _, _, w in edges])
        w_min, w_max   = ws.min(), ws.max()
        denom          = (w_max - w_min) + 1e-9
        widths         = 1.6  + 4.2  * (ws - w_min) / denom
        alphas         = 0.22 + 0.58 * (ws - w_min) / denom
    else:
        widths = alphas = []
    fig, ax = plt.subplots(figsize = figsize, facecolor = "white")
    ax.set_facecolor("#F8F8F8")
    ax.set_title(title, fontsize = 18, fontweight = "semibold", pad = 16, color = "#222222")
    ax.axis("off")
    for idx, (u, v) in enumerate(list(G.edges())):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color = [0.54, 0.58, 0.68, float(alphas[idx])], linewidth = float(widths[idx]), solid_capstyle = "round", zorder = 1)
    for i in range(0, n):
        x, y = pos[i]
        ax.scatter(x, y, s = node_sizes[i] * 1.16, color = "white", zorder = 2, linewidths = 0)
        ax.scatter(x, y, s = node_sizes[i], color = node_cols[i], zorder = 3, linewidths = 1.4, edgecolors = "white")
    for i, lbl in enumerate(labels):
        x, y = pos[i]
        bg   = node_cols[i]
        lum  = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        tc   = "white" if lum < 0.55 else "#222222"
        ax.text(x, y, lbl, ha = "center", va = "center", fontsize = 11.5, fontweight = "bold", color = tc, zorder = 5)
        ax.annotate(f"{score[i]:.2f}", xy = (x, y), xytext = (0, -24), textcoords = "offset points", ha = "center", va = "top", fontsize = 10, color = "#333333", zorder = 5, bbox = dict(boxstyle = "round,pad=0.18", facecolor = "white", edgecolor = "none", alpha = 0.88))
    rank_norm = mcolors.Normalize(vmin = 1, vmax = n)
    sm        = cm.ScalarMappable(cmap = plt.colormaps["RdYlGn_r"], norm = rank_norm)
    sm.set_array([])
    cbar      = fig.colorbar(sm, ax = ax, shrink = 0.82, pad = 0.02, aspect = 26)
    cbar.set_label("Final LaRa rank  (1 = best)", fontsize = 12, color = "#444444")
    cbar.set_ticks([1, n])
    cbar.set_ticklabels(["best", "worst"])
    cbar.ax.tick_params(labelsize = 10, color = "#888888")
    cbar.outline.set_visible(False)
    ax.margins(0.10, 0.14)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi = 300, bbox_inches = "tight", facecolor = "white")
    return fig, ax
 
# Function: Overview
def plot_lara_overview(order, score, info, alt_labels = None, title = "Ranking", figsize = (12, 6), top_k = 20, savepath = None):
    score      = np.asarray(score, dtype = float)
    n          = len(score)
    S          = info["S_graph"]
    labels     = _labels(n, alt_labels)
    ranked_idx = [a-1 for a in order]
    cmap_bar   = plt.colormaps["RdYlGn"]
    bar_colors = [cmap_bar(1.0 - r / max(n-1, 1)) for r in range(0, n)]
    fig        = plt.figure(figsize = figsize, facecolor = "white")
    gs         = gridspec.GridSpec(1, 2, figure = fig, width_ratios = [2, 1.6], wspace = 0.35)
    ax1        = fig.add_subplot(gs[0])
    ax2        = fig.add_subplot(gs[1])
    y_pos      = np.arange(n)
    fig.suptitle(title, fontsize = 13, fontweight = "semibold", y = 0.98, color = "#222222")
    ax1.set_facecolor("#F8F8F8")
    ax1.spines[["top","right","bottom"]].set_visible(False)
    ax1.spines["left"].set_color("#CCCCCC")
    ax1.barh(y_pos, [score[i] for i in ranked_idx], color = bar_colors, height = 0.72, edgecolor = "white", linewidth = 0.4)
    ax1.axvline(0, color = "#AAAAAA", lw = 0.8, zorder = 0)
    ax1.set_yticks(y_pos)
    if n <= 60:
        ax1.set_yticklabels([labels[i] for i in ranked_idx], fontsize = max(5, 9 - n//12))
    else:
        ax1.set_yticklabels([], fontsize=5)
        for y, i in list(enumerate(ranked_idx))[:5] + list(enumerate(ranked_idx))[-5:]:
            ax1.text(-0.02, y, labels[i], ha = "right", va = "center", fontsize = 6, color = "#555555", transform = ax1.get_yaxis_transform())
    ax1.invert_yaxis()
    ax1.set_xlabel("Standardised Z", fontsize = 9,  color = "#555555")
    ax1.set_title("Ranked Scores",   fontsize = 10, color = "#333333", pad = 6)
    ax1.tick_params(axis = "x", labelsize = 8, color = "#AAAAAA")
    ax1.tick_params(axis = "y", length = 0, pad = 20)
    ax1.grid(axis = "x", color = "#EEEEEE", linewidth = 0.6, zorder = 0)
    if n <= 30:
        for y, i in enumerate(ranked_idx):
            s      = score[i]
            offset = 0.02 * (score.max() - score.min())
            ax1.text(s + (offset if s >= 0 else -offset), y, f"{s:.3f}", va = "center", ha = "left" if s >= 0 else "right", fontsize = 6.5, color = "#333333")
    
    ax2.set_facecolor("white")
    degree = S.sum(axis = 1)
    if n > top_k:
        heat_idx = np.sort(np.argsort(-degree)[:top_k])
    else:
        heat_idx = np.arange(n)
    sub_S    = S[np.ix_(heat_idx, heat_idx)]
    sub_lbls = [labels[i] for i in heat_idx]
    nh       = len(heat_idx)
    ax2.imshow(sub_S, cmap = "Greys", vmin = 0, vmax = 1, aspect = "auto", interpolation = "nearest")
    if nh <= 20:
        for r in range(0, nh):
            for c in range(0, nh):
                v = sub_S[r, c]
                if v > 0.05:
                    ax2.text(c, r, f"{v:.2f}", ha = "center", va = "center", fontsize = 5.5, color = "white" if v > 0.50 else "#333333") 
    ax2.set_xticks(range(0, nh))
    ax2.set_yticks(range(0, nh))
    fs = max(5, 9 - nh//5) 
    ax2.set_xticklabels(sub_lbls, rotation = 45, ha = "right", fontsize = fs)
    ax2.set_yticklabels(sub_lbls, fontsize = fs)
    ax2.tick_params(length = 0)
    heat_title = ("Affinity Heatmap")
    ax2.set_title(heat_title, fontsize = 10, color = "#333333", pad = 6)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi = 180, bbox_inches = "tight", facecolor = "white")
    return fig, (ax1, ax2)


###############################################################################

# Function: Normalization
def quantile_normalize(Xraw, criteria_type = None, q_low = 0.10, q_high = 0.90, eps = 1e-12):
    Xraw  = np.asarray(Xraw, dtype = float)
    n, m  = Xraw.shape
    if criteria_type is None:
        criteria_type = ["max"] * m
    criteria_type = np.asarray(criteria_type, dtype = str)
    if criteria_type.shape != (m,):
        raise ValueError(f"criteria_type must contain exactly {m} entries.")
    allowed = {"max", "min"}
    invalid = sorted(set(criteria_type.tolist()) - allowed)
    if invalid:
        raise ValueError(
            "Unsupported criterion direction(s): " + ", ".join(invalid) +
            ". LaRa currently supports only 'max' and 'min'; transform target/interval criteria explicitly before use."
        )
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("Require 0 <= q_low < q_high <= 1.")
    if not np.all(np.isfinite(Xraw)):
        raise ValueError("dataset contains NaN or infinite values.")
    ql    = np.quantile(Xraw, q_low,  axis = 0)
    qh    = np.quantile(Xraw, q_high, axis = 0)
    denom = qh - ql
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    X     = (Xraw - ql) / denom
    X     = np.clip(X, 0.0, 1.0)
    is_min = (criteria_type == "min")
    if np.any(is_min):
        X[:, is_min] = 1.0 - X[:, is_min]
    return X

# Function: Distance
def weighted_pairwise_dist(X, w):
    X   = np.asarray(X, dtype = float)
    w   = np.asarray(w, dtype = float)
    Xw  = X * np.sqrt(w)
    d   = Xw[:, None, :] - Xw[None, :, :]
    return np.sqrt(np.sum(d * d, axis = -1))

# Function: Distance
def weighted_distance_to_point(X, point, w):
    X     = np.asarray(X,     dtype = float)
    point = np.asarray(point, dtype = float)
    w     = np.asarray(w,     dtype = float)
    diff  = (X - point[None, :]) * np.sqrt(w)[None, :]
    return np.sqrt(np.sum(diff * diff, axis = 1))

# Function: Sigma
def estimate_sigma_from_dist(dist, k_sigma = 3, eps = 1e-12):
    dist = np.asarray(dist, dtype = float)
    n    = dist.shape[0]
    if n <= 1:
        return 1.0
    k_eff       = max(1, min(k_sigma, n - 1))
    local_dists = []
    for i in range(0, n):
        row    = dist[i].copy()
        row[i] = np.inf          
        nbr_d  = np.sort(row)[:k_eff]
        local_dists.extend(nbr_d[np.isfinite(nbr_d)].tolist())
    if len(local_dists) == 0:
        return 1.0
    med = float(np.median(local_dists))
    if med > eps:
        return med
    nz = [d for d in local_dists if d > eps]
    if nz:
        return float(np.min(nz))
    return 1.0

# Function: RBF
def rbf_similarity_dense(dist, k_sigma = 3, sigma_override = None, eps = 1e-12):
    dist = np.asarray(dist, dtype = float)
    if sigma_override is not None:
        sigma = float(max(sigma_override, eps))
    else:
        sigma = float(max(estimate_sigma_from_dist(dist, k_sigma = k_sigma, eps = eps), eps))
    S = np.exp(-(dist * dist) / (2.0 * sigma * sigma + eps))
    np.fill_diagonal(S, 0.0)
    return S, sigma

# Function: KNN
def knn_sparse_similarity(S_dense, k = 5, mode = "union"):
    S_dense  = np.asarray(S_dense, dtype = float)
    n        = S_dense.shape[0]
    if n <= 1:
        return S_dense.copy()
    k_eff    = int(max(1, min(k, n - 1)))
    nbr_mask = np.zeros((n, n), dtype = bool)
    for i in range(0, n):
        row      = S_dense[i].copy()
        row[i]   = -np.inf
        nbrs     = np.argsort(-row, kind = "mergesort")[:k_eff]
        nbr_mask[i, nbrs] = True
    if mode == "mutual":
        keep = nbr_mask & nbr_mask.T
    else:                           
        keep = nbr_mask | nbr_mask.T
    S_sparse = np.where(keep, S_dense, 0.0)
    S_sparse = np.maximum(S_sparse, S_sparse.T)
    np.fill_diagonal(S_sparse, 0.0)
    return S_sparse

# Function: Scale
def two_scale_similarity(S_dense, k_local = 3, k_bridge = 2, alpha_bridge = 0.15, eps = 1e-12):
    S_local    = knn_sparse_similarity(S_dense, k = k_local,  mode = "mutual")
    S_bridge   = knn_sparse_similarity(S_dense, k = k_bridge, mode = "union")
    S_combined = S_local + alpha_bridge * S_bridge
    S_combined = np.maximum(S_combined, S_combined.T)
    np.fill_diagonal(S_combined, 0.0)
    return S_combined

###############################################################################

# Function: Component
def graph_components_from_similarity(S, eps = 1e-12):
    S       = np.asarray(S, dtype = float)
    n       = S.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps   = []
    for start in range(0, n):
        if visited[start]:
            continue
        stack          = [start]
        visited[start] = True
        comp           = []
        while stack:
            u    = stack.pop()
            comp.append(u)
            nbrs = np.where(S[u] > eps)[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps

# Function: Connect
def connect_components_with_dense_similarity(S_sparse, S_dense, eps = 1e-12):
    S_conn  = np.asarray(S_sparse, dtype = float).copy()
    S_dense = np.asarray(S_dense,  dtype = float)
    comps   = graph_components_from_similarity(S_conn, eps = eps)
    bridges = []
    while len(comps) > 1:
        best_w  = -np.inf
        best_uv = None
        for a in range(0, len(comps)):
            for b in range(a + 1, len(comps)):
                sub      = S_dense[np.ix_(comps[a], comps[b])]
                idx_flat = np.argmax(sub)
                w        = sub.flat[idx_flat]
                if w > best_w:
                    ia, ib  = np.unravel_index(idx_flat, sub.shape)
                    best_w  = float(w)
                    best_uv = (comps[a][ia], comps[b][ib])
        if best_uv is None or best_w <= eps:
            best_uv = (comps[0][0], comps[1][0])
            # Component detection uses S > eps, so a fallback bridge equal to eps
            # would remain invisible and could make this loop non-terminating.
            best_w  = float(np.nextafter(float(eps), np.inf))
        u, v             = best_uv
        S_conn[u, v]     = max(S_conn[u, v], best_w)
        S_conn[v, u]     = max(S_conn[v, u], best_w)
        bridges.append((int(u), int(v), float(best_w)))
        comps            = graph_components_from_similarity(S_conn, eps = eps)
    return S_conn, bridges

# Function: Laplacian
def combinatorial_laplacian(S):
    S   = np.asarray(S, dtype = float)
    deg = S.sum(axis = 1)
    L   = np.diag(deg) - S
    return L, deg

###############################################################################

# Function: Dominance
def dominance_relation_matrix(X_norm, eps_dom = 1e-6):
    X    = np.asarray(X_norm, dtype = float)
    n, m = X.shape
    if n == 0:
        return np.zeros((0, 0), dtype = bool)
    diff      = X[:, None, :] - X[None, :, :]
    ge        = np.all(diff >= -eps_dom, axis = 2)
    gt        = np.any(diff >   eps_dom, axis = 2)
    dominates = ge & gt
    np.fill_diagonal(dominates, False)
    return dominates

# Function: Score
def dominance_score(X_norm, eps_dom = 1e-6):
    dominates = dominance_relation_matrix(X_norm, eps_dom = eps_dom)
    wins      = dominates.sum(axis = 1)
    losses    = dominates.sum(axis = 0)
    return (wins - losses).astype(int)

# Function: Violation
def dominance_violations(score, dominates, tol = 1e-12):
    score     = np.asarray(score,     dtype = float)
    dominates = np.asarray(dominates, dtype = bool)
    edges     = np.argwhere(dominates)
    if edges.shape[0] == 0:
        return 0, 0.0
    gaps = score[edges[:, 1]] - score[edges[:, 0]]
    mask = gaps > tol
    if not np.any(mask):
        return 0, 0.0
    return int(mask.sum()), float(np.max(gaps[mask]))

# Function: Monotonicity
def enforce_dominance_monotonicity(score, dominates, tol = 1e-12, max_iter = 5000):
    g         = np.asarray(score,     dtype = float).copy()
    dominates = np.asarray(dominates, dtype = bool)
    edges     = np.argwhere(dominates)
    if edges.shape[0] == 0:
        return g
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer.")
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    for _ in range(0, int(max_iter)):
        gaps = g[j_idx] - g[i_idx]
        mask = gaps > tol
        if not np.any(mask):
            break
        delta = 0.5 * gaps * mask.astype(float)
        np.add.at(g, i_idx,  delta)
        np.add.at(g, j_idx, -delta)

    n_viol, max_viol = dominance_violations(g, dominates, tol = tol)
    if n_viol:
        raise RuntimeError(
            f"dominance repair did not converge within {int(max_iter)} iterations "
            f"({n_viol} inversion(s) remain; maximum inversion {max_viol:.3e})."
        )

    g, _, _ = _safe_standardize(g, eps = tol)
    n_viol, max_viol = dominance_violations(g, dominates, tol = tol)
    if n_viol:
        raise RuntimeError(
            f"dominance repair postcondition failed after standardization "
            f"({n_viol} inversion(s) remain; maximum inversion {max_viol:.3e})."
        )
    return g

###############################################################################

# Function: Reference
def select_existing_reference_indices(X_norm, w, dom, eps = 1e-12):
    X           = np.asarray(X_norm, dtype = float)
    w           = np.asarray(w,      dtype = float)
    dom         = np.asarray(dom,    dtype = float)
    n, m        = X.shape
    idx         = np.arange(n)
    ideal       = np.ones( m, dtype = float)
    antiideal   = np.zeros(m, dtype = float)
    d_best      = weighted_distance_to_point(X, ideal,     w)
    d_worst     = weighted_distance_to_point(X, antiideal, w)
    closeness   = d_worst / (d_best + d_worst + eps)
    best_order  = np.lexsort((idx, d_best,  -closeness, -dom))
    best_idx    = int(best_order[0])
    worst_order = np.lexsort((idx, d_worst,  closeness,  dom))
    worst_idx   = int(worst_order[0])

    if n > 1 and worst_idx == best_idx:
        for cand in worst_order:
            if int(cand) != best_idx:
                worst_idx = int(cand)
                break
    return best_idx, worst_idx

# Function: Prior
def build_reference_prior(X_norm, w, dom, reference_mode = "ideal", alpha_dom_prior = 0.5, adaptive_alpha = True, eps = 1e-12):
    X    = np.asarray(X_norm, dtype = float)
    w    = np.asarray(w,      dtype = float)
    dom  = np.asarray(dom,    dtype = float)
    n, m = X.shape

    if reference_mode not in {"ideal", "existing"}:
        raise ValueError("reference_mode must be either 'ideal' or 'existing'.")
    if not np.isfinite(float(alpha_dom_prior)) or not (0.0 <= float(alpha_dom_prior) <= 1.0):
        raise ValueError("alpha_dom_prior must be finite and lie in [0, 1].")

    if reference_mode == "ideal":
        best_idx    = None
        worst_idx   = None
        best_point  = np.ones( m, dtype = float)
        worst_point = np.zeros(m, dtype = float)
    else:
        best_idx, worst_idx = select_existing_reference_indices(X, w, dom, eps = eps)
        best_point          = X[best_idx].copy()
        worst_point         = X[worst_idx].copy()
    dist_to_best      = weighted_distance_to_point(X, best_point,  w)
    dist_to_worst     = weighted_distance_to_point(X, worst_point, w)
    closeness         = dist_to_worst / (dist_to_best + dist_to_worst + eps)
    ref_z,  _, ref_sd = _safe_standardize(closeness, eps = eps)
    dom_z,  _, dom_sd = _safe_standardize(dom,       eps = eps)

    corr_defined = bool(ref_sd > eps and dom_sd > eps)
    corr_val     = float(np.corrcoef(ref_z, dom_z)[0, 1]) if corr_defined else 0.0
    if corr_defined:
        corr_val = float(np.clip(corr_val, -1.0, 1.0))

    # Degenerate signals are handled explicitly. If dominance is constant it
    # contains no ordinal information, so the prior uses closeness alone. If
    # closeness is constant but dominance varies, adaptive mode falls back to
    # the only informative signal (dominance) rather than evaluating an
    # undefined Pearson correlation. If both are constant, the prior is the
    # zero vector and the final ranking is a deterministic tie ordering.
    if dom_sd <= eps:
        alpha_eff = 0.0
    elif ref_sd <= eps and adaptive_alpha:
        alpha_eff = 1.0
    elif adaptive_alpha:
        disagreement = 0.5 * (1.0 - corr_val)
        alpha_eff    = float(alpha_dom_prior) + disagreement * (1.0 - float(alpha_dom_prior))
        alpha_eff    = float(np.clip(alpha_eff, 0.0, 1.0))
    else:
        alpha_eff = float(alpha_dom_prior)

    prior       = (1.0 - alpha_eff) * ref_z + alpha_eff * dom_z
    prior, _, _ = _safe_standardize(prior, eps = eps)

    info = {
            "reference_mode":                         reference_mode,
            "best_reference_index":                   best_idx,
            "worst_reference_index":                  worst_idx,
            "best_reference_point":                   best_point.copy(),
            "worst_reference_point":                  worst_point.copy(),
            "reference_closeness":                    closeness.copy(),
            "dist_to_best_ref":                       dist_to_best.copy(),
            "dist_to_worst_ref":                      dist_to_worst.copy(),
            "alpha_dom_prior_requested":              float(alpha_dom_prior),
            "alpha_dom_prior_effective":              float(alpha_eff),
            "adaptive_alpha":                         bool(adaptive_alpha),
            "closeness_dom_correlation":              float(corr_val),
            "closeness_dom_correlation_defined":      corr_defined,
            "closeness_standard_deviation":           float(ref_sd),
            "dominance_standard_deviation":           float(dom_sd),
            }
    return prior, info

###############################################################################

# Function: Graph
def pairwise_graph_regularized_score(Lc, dominates, S_graph, prior_signal, lambda_graph = 1.0, mu_prior = 1.0, gamma_pair = 0.15, margin = 0.15, local_dominance = True, eps = 1e-10):
    Lc        = np.asarray(Lc,           dtype = float)
    dominates = np.asarray(dominates,    dtype = bool)
    S_graph   = np.asarray(S_graph,      dtype = float)
    u         = np.asarray(prior_signal, dtype = float)
    lambda_graph = float(lambda_graph)
    mu_prior     = float(mu_prior)
    gamma_pair   = float(gamma_pair)
    margin       = float(margin)
    if not np.isfinite(lambda_graph) or lambda_graph < 0.0:
        raise ValueError("lambda_graph must be finite and nonnegative.")
    if not np.isfinite(mu_prior) or mu_prior <= 0.0:
        raise ValueError("mu_prior must be finite and strictly positive.")
    if not np.isfinite(gamma_pair) or gamma_pair < 0.0:
        raise ValueError("gamma_pair must be finite and nonnegative.")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin (delta_dom) must be finite and nonnegative.")
    n         = Lc.shape[0]
    A         = lambda_graph * Lc + mu_prior * np.eye(n)
    b         = mu_prior * u.copy()
    edges     = np.argwhere(dominates)
    for i, j in edges:
        if local_dominance and S_graph[i, j] <= 0.0:
            continue
        gp         = gamma_pair
        A[i, i]    = A[i, i] + gp
        A[j, j]    = A[j, j] + gp
        A[i, j]    = A[i, j] - gp
        A[j, i]    = A[j, i] - gp
        b[i]       = b[i] + gp * margin
        b[j]       = b[j] - gp * margin
    A       = A + eps * np.eye(n)
    f       = np.linalg.solve(A, b)
    z, _, _ = _safe_standardize(f, eps = eps)
    return z

###############################################################################

# Function: LaRa (Laplacian Ranking)
def lara_method(dataset, W, criteria_type = None, lambda_graph = 1.0, k_graph = 3, reference_mode = "ideal", margin = 0.0, gamma_pair = 0.0,  q_low = 0.10, q_high = 0.90, use_sparse_graph = True, graph_mode = "union", auto_connect = True, use_two_scale_graph = False, k_local = 3, k_bridge = 2, alpha_bridge = 0.15, sigma_override = None, k_sigma = 3, eps_dom = 1e-6, local_dominance = True, alpha_dom_prior = 0.5, adaptive_alpha = True, mu_prior = 1.0, dominance_repair = True):
    Xraw                   = np.asarray(dataset, dtype = float)
    if Xraw.ndim != 2:
        raise ValueError("dataset must be a two-dimensional decision matrix.")
    n, m                   = Xraw.shape
    if n < 2 or m < 1:
        raise ValueError("dataset must contain at least two alternatives and one criterion.")
    W                      = np.asarray(W, dtype = float)
    if W.shape != (m,):
        raise ValueError(f"W must contain exactly {m} criterion weights.")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("criterion weights must be finite and nonnegative.")
    if float(W.sum()) <= 0.0:
        raise ValueError("at least one criterion weight must be positive.")

    if reference_mode not in {"ideal", "existing"}:
        raise ValueError("reference_mode must be either 'ideal' or 'existing'.")
    if graph_mode not in {"union", "mutual"}:
        raise ValueError("graph_mode must be either 'union' or 'mutual'.")
    if not isinstance(k_graph, (int, np.integer)) or isinstance(k_graph, bool) or int(k_graph) < 1:
        raise ValueError("k_graph must be a positive integer.")
    if not isinstance(k_local, (int, np.integer)) or isinstance(k_local, bool) or int(k_local) < 1:
        raise ValueError("k_local must be a positive integer.")
    if not isinstance(k_bridge, (int, np.integer)) or isinstance(k_bridge, bool) or int(k_bridge) < 1:
        raise ValueError("k_bridge must be a positive integer.")
    if not isinstance(k_sigma, (int, np.integer)) or isinstance(k_sigma, bool) or int(k_sigma) < 1:
        raise ValueError("k_sigma must be a positive integer.")
    if not np.isfinite(float(lambda_graph)) or float(lambda_graph) < 0.0:
        raise ValueError("lambda_graph must be finite and nonnegative.")
    if not np.isfinite(float(mu_prior)) or float(mu_prior) <= 0.0:
        raise ValueError("mu_prior must be finite and strictly positive.")
    if not np.isfinite(float(gamma_pair)) or float(gamma_pair) < 0.0:
        raise ValueError("gamma_pair must be finite and nonnegative.")
    if not np.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("margin (delta_dom) must be finite and nonnegative.")
    if not np.isfinite(float(alpha_dom_prior)) or not (0.0 <= float(alpha_dom_prior) <= 1.0):
        raise ValueError("alpha_dom_prior must be finite and lie in [0, 1].")
    if not np.isfinite(float(eps_dom)) or float(eps_dom) < 0.0:
        raise ValueError("eps_dom must be finite and nonnegative.")
    if not np.isfinite(float(alpha_bridge)) or float(alpha_bridge) < 0.0:
        raise ValueError("alpha_bridge must be finite and nonnegative.")
    if sigma_override is not None and (not np.isfinite(float(sigma_override)) or float(sigma_override) <= 0.0):
        raise ValueError("sigma_override must be finite and strictly positive when supplied.")

    X                      = quantile_normalize(Xraw, criteria_type, q_low, q_high, 1e-12)
    w                      = W / W.sum()
    active_criteria        = w > 0.0
    # A criterion assigned exactly zero metric weight is intentionally inactive:
    # it cannot re-enter the model indirectly through the Pareto-dominance signal.
    X_dom                  = X[:, active_criteria]
    dominates              = dominance_relation_matrix(X_dom, eps_dom = eps_dom)
    dom                    = dominance_score(X_dom,           eps_dom = eps_dom)
    prior_signal, ref_info = build_reference_prior(X_norm = X, w = w, dom = dom, reference_mode = reference_mode, alpha_dom_prior = alpha_dom_prior, adaptive_alpha = adaptive_alpha, eps = 1e-12)
    prior_order            = [i + 1 for i in np.argsort(-prior_signal, kind = "mergesort")]
    dist                   = weighted_pairwise_dist(X, w)
    S_dense, sigma_used    = rbf_similarity_dense(dist, k_sigma = k_sigma, sigma_override = sigma_override, eps = 1e-12)
    bridges                = []
    if use_sparse_graph:
        if use_two_scale_graph:
            S_graph          = two_scale_similarity(S_dense, k_local = k_local, k_bridge = k_bridge, alpha_bridge = alpha_bridge, eps = 1e-12)
        else:
            S_graph          = knn_sparse_similarity(S_dense, k = k_graph, mode = graph_mode)
        if auto_connect:
            S_graph, bridges = connect_components_with_dense_similarity(S_graph, S_dense, eps = 1e-12)
    else:
        S_graph = S_dense.copy()

    Lc, deg = combinatorial_laplacian(S_graph)
    score   = pairwise_graph_regularized_score(Lc = Lc, dominates = dominates, S_graph = S_graph, prior_signal = prior_signal, lambda_graph = lambda_graph, mu_prior = mu_prior, gamma_pair = gamma_pair, margin = margin, local_dominance = local_dominance, eps = 1e-10)
    n_viol_pre, max_viol_pre = dominance_violations(score, dominates, tol = 1e-12)
    if dominance_repair:
        score = enforce_dominance_monotonicity(score, dominates, tol = 1e-12, max_iter = 5000)
    order            = np.argsort(-score, kind = "mergesort")
    n_viol, max_viol = dominance_violations(score, dominates, tol = 1e-12)
    info             = {
                        "X":                             X.copy(),
                        "weights":                       w.copy(),
                        "reference_mode":                ref_info["reference_mode"],
                        "best_reference_index":          ref_info["best_reference_index"],
                        "worst_reference_index":         ref_info["worst_reference_index"],
                        "best_reference_point":          ref_info["best_reference_point"].copy(),
                        "worst_reference_point":         ref_info["worst_reference_point"].copy(),
                        "reference_closeness":           ref_info["reference_closeness"].copy(),
                        "dist_to_best_ref":              ref_info["dist_to_best_ref"].copy(),
                        "dist_to_worst_ref":             ref_info["dist_to_worst_ref"].copy(),
                        "alpha_dom_prior_requested":     ref_info["alpha_dom_prior_requested"],
                        "alpha_dom_prior_effective":     ref_info["alpha_dom_prior_effective"],
                        "adaptive_alpha":                ref_info["adaptive_alpha"],
                        "closeness_dom_correlation":     ref_info["closeness_dom_correlation"],
                        "closeness_dom_correlation_defined": ref_info["closeness_dom_correlation_defined"],
                        "closeness_standard_deviation":  ref_info["closeness_standard_deviation"],
                        "dominance_standard_deviation":  ref_info["dominance_standard_deviation"],
                        "active_criteria":               active_criteria.copy(),
                        "prior_signal":                  prior_signal.copy(),
                        "prior_order":                   prior_order,
                        "sigma_used":                    float(sigma_used),
                        "S_dense":                       S_dense.copy(),
                        "S_graph":                       S_graph.copy(),
                        "dominates":                     dominates.copy(),
                        "dominance_pairs":               int(np.sum(dominates)),
                        "dominance_violations_before":   int(n_viol_pre),
                        "max_dominance_violation_before":float(max_viol_pre),
                        "dominance_violations_after":    int(n_viol),
                        "max_dominance_violation_after": float(max_viol),
                        "score":                         score.copy(),
                        "order":                         [i + 1 for i in order],
                        "bridges_added":                 bridges,
                        "graph_is_sparse":               bool(use_sparse_graph),
                        "use_two_scale_graph":           bool(use_two_scale_graph),
                        "local_dominance":               bool(local_dominance),
                        "params": {
                            "q_low":              q_low,
                            "q_high":             q_high,
                            "lambda_graph":       float(lambda_graph),
                            "mu_prior":           float(mu_prior),
                            "gamma_pair":         float(gamma_pair),
                            "margin":             float(margin),
                            "k_graph":            int(k_graph),
                            "graph_mode":         graph_mode,
                            "alpha_dom_prior":    float(alpha_dom_prior),
                            "eps_dom":            float(eps_dom),
                            "local_dominance":    bool(local_dominance),
                            "use_two_scale_graph":bool(use_two_scale_graph),
                            "sigma_override":     sigma_override,
                        },
                    }
    return info["order"], info["score"], info

###############################################################################
