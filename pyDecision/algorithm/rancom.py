from __future__ import annotations

###############################################################################

# Required Libraries

import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

###############################################################################

Mode = Literal["ranking", "scores", "mac", "tournament"]

###############################################################################

# Class
@dataclass(frozen = True)
class RancomResult:
    mode:                   Mode
    n_criteria:             int
    criteria_labels:        List[str]
    ranking:                Optional[np.ndarray] = None
    scores:                 Optional[np.ndarray] = None
    mac:                    Optional[np.ndarray] = None
    scw:                    Optional[np.ndarray] = None
    weights:                Optional[np.ndarray] = None
    triads_T:               Optional[int]        = None
    triads_weak:            Optional[int]        = None
    triads_strong:          Optional[int]        = None
    triads_inconsistent:    Optional[int]        = None
    triad_consistency_CT_r: Optional[float]      = None

###############################################################################

# Helper Functions
def _as_1d_float_array(x, name):
    arr = np.asarray(x, dtype = float).reshape(-1)
    return arr

def _as_square_float_matrix(x, name) :
    mat = np.asarray(x, dtype = float)
    return mat

def _default_labels(n):
    return [f"c{i+1}" for i in range(n)]

def _to_zero_based(idx, one_indexed):
    return idx - 1 if one_indexed else idx

def _reciprocal(v):
    if v == 1.0:
        return 0.0
    if v == 0.0:
        return 1.0
    return 0.5

def _snap(a, tol):
    if abs(a - 0.0) <= tol:
        return 0.0
    if abs(a - 0.5) <= tol:
        return 0.5
    if abs(a - 1.0) <= tol:
        return 1.0
    return a

def validate_mac(mac):
    m = _as_square_float_matrix(mac, "mac")
    return m

def compute_scw(mac):
    m   = _as_square_float_matrix(mac, "mac")
    scw = m.sum(axis = 1)
    return scw

def compute_weights(scw):
    v     = _as_1d_float_array(scw, "scw")
    denom = float(v.sum())
    w     = v / denom
    return w

def triad_consistency_coefficient(mac, tol = 1e-12):
    m = validate_mac(mac)
    n = m.shape[0]
    weak_patterns: Dict[Tuple[float, float], set] = {
                                                        (1.0, 1.0): {0.5},
                                                        (1.0, 0.5): {0.5},
                                                        (0.5, 1.0): {0.5},
                                                        (0.5, 0.5): {1.0, 0.0},
                                                        (0.5, 0.0): {0.5},
                                                        (0.0, 0.5): {0.5},
                                                        (0.0, 0.0): {0.5},}
    strong_patterns: Dict[Tuple[float, float], set] = {
                                                        (1.0, 1.0): {0.0},
                                                        (1.0, 0.5): {0.0},
                                                        (0.5, 1.0): {0.0},
                                                        (0.5, 0.0): {1.0},
                                                        (0.0, 0.5): {1.0},
                                                        (0.0, 0.0): {1.0},}
    weak   = 0
    strong = 0
    total  = 0
    for i in range(0, n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                total = total + 1
                aij   = _snap(m[i, j], tol)
                ajk   = _snap(m[j, k], tol)
                aik   = _snap(m[i, k], tol)
                if (aij, ajk) in weak_patterns and aik in weak_patterns[(aij, ajk)]:
                    weak   = weak + 1
                if (aij, ajk) in strong_patterns and aik in strong_patterns[(aij, ajk)]:
                    strong = strong + 1
    inc     = weak + strong
    T       = total
    ct_r    = inc / T if T > 0 else 0.0
    details = {"t": n, "T": T, "T_weak_inc": weak, "T_strong_inc": strong, "T_inc": inc}
    return ct_r, details

###############################################################################

# Function: MAC Ranking
def build_mac_from_ranking(ranking):
    r   = _as_1d_float_array(ranking, "ranking")
    n   = r.size
    mac = np.zeros((n, n), dtype = float)

    for i in range(0, n):
        for j in range(0, n):
            if r[i] < r[j]:
                mac[i, j] = 1.0
            elif r[i] == r[j]:
                mac[i, j] = 0.5
            else:
                mac[i, j] = 0.0
    return mac

# Function: MAC Scores
def build_mac_from_scores(scores, higher_is_more_important):
    s   = _as_1d_float_array(scores, "scores")
    n   = s.size
    mac = np.zeros((n, n), dtype = float)

    for i in range(0, n):
        for j in range(0, n):
            si, sj = s[i], s[j]
            if not higher_is_more_important:
                si, sj = -si, -sj 

            if si > sj:
                mac[i, j] = 1.0
            elif si == sj:
                mac[i, j] = 0.5
            else:
                mac[i, j] = 0.0
    return mac

# Function: MAC Tournment
def build_mac_from_tournament(n_criteria, judgments, one_indexed = True, diag_value = 0.5):
    mac = np.full((n_criteria, n_criteria), np.nan, dtype = float)
    np.fill_diagonal(mac, float(diag_value))
    for i_raw, j_raw, v_raw in judgments:
        v         = float(v_raw)
        i         = _to_zero_based(int(i_raw), one_indexed)
        j         = _to_zero_based(int(j_raw), one_indexed)
        mac[i, j] = v
        vji       = _reciprocal(v)
        mac[j, i] = vji
    return mac

###############################################################################

# Function: RANCOM Method
def rancom_method(data, mode = "ranking", criteria_labels = None, higher_is_more_important = True, compute_triad_consistency = False):
    if mode == "ranking":
        ranking = _as_1d_float_array(data, "ranking")
        n       = ranking.size
        labels  = criteria_labels or _default_labels(n)
        mac     = build_mac_from_ranking(ranking)
        scw     = compute_scw(mac)
        weights = compute_weights(scw)
        ct_r    = None
        details = None
        if compute_triad_consistency:
            ct_r, details = triad_consistency_coefficient(mac)

        return RancomResult(mode = mode, n_criteria = n, criteria_labels = labels, ranking = ranking, mac = mac, scw = scw, weights = weights, triads_T = None if not details else details["T"], triads_weak = None if not details else details["T_weak_inc"], triads_strong = None if not details else details["T_strong_inc"], triads_inconsistent = None if not details else details["T_inc"], triad_consistency_CT_r = ct_r)

    if mode == "scores":
        scores  = _as_1d_float_array(data, "scores")
        n       = scores.size
        labels  = criteria_labels or _default_labels(n)

        mac     = build_mac_from_scores(scores, higher_is_more_important = higher_is_more_important)
        scw     = compute_scw(mac)
        weights = compute_weights(scw)
        ct_r    = None
        details = None
        if compute_triad_consistency:
            ct_r, details = triad_consistency_coefficient(mac)
        return RancomResult(mode = mode, n_criteria = n, criteria_labels = labels, scores = scores, mac = mac, scw = scw, weights = weights, triads_T = None if not details else details["T"], triads_weak = None if not details else details["T_weak_inc"], triads_strong = None if not details else details["T_strong_inc"], triads_inconsistent = None if not details else details["T_inc"], triad_consistency_CT_r = ct_r)

    if mode == "mac":
        mac     = validate_mac(data)
        n       = mac.shape[0]
        labels  = criteria_labels or _default_labels(n)
        scw     = compute_scw(mac)
        weights = compute_weights(scw)
        ct_r    = None
        details = None
        if compute_triad_consistency:
            ct_r, details = triad_consistency_coefficient(mac)
        return RancomResult(mode = mode, n_criteria = n, criteria_labels = labels, mac = mac, scw = scw, weights = weights, triads_T = None if not details else details["T"], triads_weak = None if not details else details["T_weak_inc"], triads_strong = None if not details else details["T_strong_inc"], triads_inconsistent = None if not details else details["T_inc"], triad_consistency_CT_r = ct_r)
    
    if mode == "tournament":
        n       = max(max(i, j) for (i, j, _) in data)
        labels  = criteria_labels or _default_labels(n)
        mac     = build_mac_from_tournament(n_criteria = n, judgments = data)
        scw     = compute_scw(mac)
        weights = compute_weights(scw)
        ct_r    = None
        details = None
        if compute_triad_consistency:
            ct_r, details = triad_consistency_coefficient(mac)
        return RancomResult(mode = mode, n_criteria = n, criteria_labels = labels, mac = mac, scw = scw, weights = weights, triads_T = None if not details else details["T"], triads_weak = None if not details else details["T_weak_inc"], triads_strong = None if not details else details["T_strong_inc"], triads_inconsistent = None if not details else details["T_inc"], triad_consistency_CT_r = ct_r)

###############################################################################