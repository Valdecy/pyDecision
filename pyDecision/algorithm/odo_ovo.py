###############################################################################

# Required Libraries
import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Function: Rank
def ranking(flow):
    if flow.shape[0] == 0:
        return
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0] - i
    for i in range(0, rank_xy.shape[0]):
        plt.text(
            rank_xy[i, 0],
            rank_xy[i, 1],
            'a' + str(int(flow[i, 0])),
            size = 12,
            ha   = 'center',
            va   = 'center',
            bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8)),
        )
    for i in range(0, rank_xy.shape[0] - 1):
        plt.arrow(
            rank_xy[i, 0],
            rank_xy[i, 1],
            rank_xy[i + 1, 0] - rank_xy[i, 0],
            rank_xy[i + 1, 1] - rank_xy[i, 1],
            head_width           = 0.01,
            head_length          = 0.2,
            overhang             = 0.0,
            color                = 'black',
            linewidth            = 0.9,
            length_includes_head = True,
        )
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:, 1])
    ymax = np.amax(rank_xy[:, 1])
    if ymin < ymax:
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin - 1, ymax + 1])
    plt.axis('off')
    plt.show()
    return

###############################################################################

def _validate_input(dataset, weights, criterion_type, critical_criterion):
    X = np.asarray(dataset)
    if X.ndim != 2:
        raise ValueError('dataset must be a 2D array')
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError('dataset must be numeric')
    X = X.astype(float)
    w = np.asarray(weights, dtype=float).flatten()
    if w.shape[0] != X.shape[1]:
        raise ValueError('weights length must be equal to the number of criteria')
    if np.sum(w) <= 0:
        raise ValueError('weights must sum to a positive number')
    w = w / np.sum(w)
    if len(criterion_type) != X.shape[1]:
        raise ValueError('criterion_type length must be equal to the number of criteria')
    for c in criterion_type:
        if c not in ['max', 'min']:
            raise ValueError("criterion_type entries must be 'max' or 'min'")

    if isinstance(critical_criterion, (list, tuple, np.ndarray)):
        critical_criterion = [int(idx) for idx in critical_criterion]
        if len(critical_criterion) == 0:
            raise ValueError('critical_criterion must contain at least one valid 0-based criterion index')
        for idx in critical_criterion:
            if idx < 0 or idx >= X.shape[1]:
                raise ValueError('critical_criterion must contain valid 0-based criterion indices')
    else:
        critical_criterion = int(critical_criterion)
        if critical_criterion < 0 or critical_criterion >= X.shape[1]:
            raise ValueError('critical_criterion must be a valid 0-based criterion index')
    return X, w, critical_criterion


def _threshold_value(x, threshold_method = 'mean', threshold_value = None):
    if threshold_method == 'mean':
        tau_0 = float(np.mean(x))
    elif threshold_method == 'median':
        tau_0 = float(np.median(x))
    elif threshold_method == 'fixed':
        if threshold_value is None:
            raise ValueError("threshold_method='fixed' requires threshold_value")
        tau_0 = float(threshold_value)
    elif threshold_method == 'quantile':
        if threshold_value is None or not (0 < threshold_value < 1):
            raise ValueError("threshold_method='quantile' requires 0 < threshold_value < 1")
        tau_0 = float(np.quantile(x, threshold_value))
    else:
        raise ValueError('unknown threshold_method')
    return tau_0

def _apply_operator(x, op, value):
    if op == '==':
        return x == value
    if op == '!=':
        return x != value
    if op == '>':
        return x > value
    if op == '>=':
        return x >= value
    if op == '<':
        return x < value
    if op == '<=':
        return x <= value
    raise ValueError('unsupported operator in logic_vetos')

def _admissibility_mask(X, criterion_type, critical_criterion, threshold_method = 'mean', threshold_value = None, hard_ranges = None, logic_vetos = None):
    mask = np.ones(X.shape[0], dtype = bool)
    if isinstance(critical_criterion, (list, tuple, np.ndarray)):
        tau_0 = {}
        for idx in critical_criterion:
            crit_values = X[:, idx]
            tau_j       = _threshold_value(crit_values, threshold_method = threshold_method, threshold_value = threshold_value)
            tau_0[idx]  = tau_j
            if criterion_type[idx] == 'max':
                mask = mask & (crit_values >= tau_j)
            else:
                mask = mask & (crit_values <= tau_j)
    else:
        crit_values = X[:, critical_criterion]
        tau_0       = _threshold_value(crit_values, threshold_method = threshold_method, threshold_value = threshold_value)
        if criterion_type[critical_criterion] == 'max':
            mask = mask & (crit_values >= tau_0)
        else:
            mask = mask & (crit_values <= tau_0)
    if hard_ranges is not None:
        for key, bounds in hard_ranges.items():
            lo, hi = bounds
            if key < 0 or key >= X.shape[1]:
                raise ValueError('hard_ranges contains an invalid criterion index')
            mask = mask & (X[:, key] >= lo) & (X[:, key] <= hi)
    if logic_vetos is not None:
        for rule in logic_vetos:
            trigger = np.ones(X.shape[0], dtype=bool)
            for condition in rule:
                idx = int(condition[0])
                op  = condition[1]
                val = condition[2]
                if idx < 0 or idx >= X.shape[1]:
                    raise ValueError('logic_vetos contains an invalid criterion index')
                trigger = trigger & _apply_operator(X[:, idx], op, val)
            mask = mask & (~trigger)
    return mask, tau_0

def _orient_admissible_matrix(X_adm, criterion_type, normalize = True):
    if X_adm.shape[0] == 0:
        raise ValueError('there are no admissible alternatives to evaluate')
    if normalize is False:
        for i in range(0, len(criterion_type)):
            if criterion_type[i] != 'max':
                raise ValueError("normalize=False is only valid when all criteria are already benefit-oriented ('max')")
        return np.copy(X_adm)
    X_oriented = np.zeros_like(X_adm, dtype = float)
    for j in range(0, X_adm.shape[1]):
        c_min = np.min(X_adm[:, j])
        c_max = np.max(X_adm[:, j])
        span  = c_max - c_min
        if span == 0:
            X_oriented[:, j] = 1.0
        elif criterion_type[j] == 'max':
            X_oriented[:, j] = (X_adm[:, j] - c_min) / span
        else:
            X_oriented[:, j] = (c_max - X_adm[:, j]) / span
    return X_oriented

def _distance_to_score(metric):
    return 1.0 / (1.0 + metric)

###############################################################################

# Function: ODO-OVO
def odo_ovo_method(dataset, weights, criterion_type, critical_criterion, threshold_method = 'mean', threshold_value = None, hard_ranges = None, logic_vetos = None, normalize = True, weighted = False, rank_by = 'l2', graph = True, verbose = True):
    X, w, critical_criterion = _validate_input(dataset, weights, criterion_type, critical_criterion)
    mask, tau_0              = _admissibility_mask(X, criterion_type, critical_criterion, threshold_method = threshold_method, threshold_value = threshold_value, hard_ranges = hard_ranges, logic_vetos = logic_vetos,)
    score                    = np.zeros(X.shape[0], dtype = float)
    l1_full                  = np.full(X.shape[0], np.nan)
    l2_full                  = np.full(X.shape[0], np.nan)
    linf_full                = np.full(X.shape[0], np.nan)
    combined_full            = np.full(X.shape[0], np.nan)
    admissible_idx           = np.where(mask)[0]
    if admissible_idx.shape[0] > 0:
        X_adm = np.copy(X[admissible_idx, :])
        X_adm = _orient_admissible_matrix(X_adm, criterion_type, normalize = normalize)
        if weighted is True:
            X_eval = X_adm * w
        else:
            X_eval = np.copy(X_adm)
        ideal      = np.max(X_eval, axis=0)
        diff       = np.abs(ideal - X_eval)
        l1         = np.sum(diff, axis=1)
        l2         = np.sqrt(np.sum(diff**2, axis=1))
        linf       = np.max(diff, axis=1)
        combined   = (l1 + l2 + linf) / 3.0
        metric_map = {'l1': l1, 'l2': l2, 'linf': linf, 'combined': combined,}
        if rank_by not in metric_map:
            raise ValueError("rank_by must be 'l1', 'l2', 'linf', or 'combined'")
        metric                        = metric_map[rank_by]
        score_adm                     = _distance_to_score(metric)
        score[admissible_idx]         = score_adm
        l1_full[admissible_idx]       = l1
        l2_full[admissible_idx]       = l2
        linf_full[admissible_idx]     = linf
        combined_full[admissible_idx] = combined

    if verbose is True:
        if isinstance(tau_0, dict):
            tau_print = {('c' + str(k + 1)): round(v, 6) for k, v in tau_0.items()}
        else:
            tau_print = round(tau_0, 6)
        print('ODO admissibility threshold (tau_0): ' + str(tau_print))
        print('Admissible alternatives: ' + str(list(map(int, admissible_idx + 1))))
        print('')
        for i in range(0, score.shape[0]):
            status = 'Admissible' if mask[i] else 'Inadmissible'
            print('a' + str(i + 1) + ': ' + str(round(score[i], 6)) + ' (' + status + ')')

    if graph is True and admissible_idx.shape[0] > 0:
        flow = np.copy(score[admissible_idx])
        flow = np.reshape(flow, (flow.shape[0], 1))
        flow = np.insert(flow, 0, admissible_idx + 1, axis=1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return score

###############################################################################
