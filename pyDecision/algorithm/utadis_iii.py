###############################################################################
# Required Libraries
import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp 
from sklearn.model_selection import train_test_split

###############################################################################

# Function: Build Grids
def build_grids(X, ai):
    gi_min = X.min(axis = 0)
    gi_max = X.max(axis = 0)
    grids  = [np.linspace(gi_min[i], gi_max[i], ai) for i in range(0, X.shape[1])]
    return grids

# Function: Build UW
def build_Uw(X, grids, ai):

    def w_index(i, j):
        return i * (ai - 1) + j

    n, m = X.shape
    n_w  = m * (ai - 1)
    Uw   = np.zeros((n, n_w))

    for a in range(0, n):
        for i in range(0, m):
            val  = X[a, i]
            grid = grids[i]
            if val >= grid[-1]:
                j = len(grid) - 2
            else:
                j = np.searchsorted(grid, val, side = "right") - 1
                j = min(max(j, 0), len(grid) - 2)
            g_j  = grid[j]
            g_j1 = grid[j + 1]
            t    = 0.0 if g_j1 == g_j else (val - g_j) / (g_j1 - g_j)
            for k in range(0, j):
                Uw[a, w_index(i, k)] = Uw[a, w_index(i, k)] + 1.0
            Uw[a, w_index(i, j)]     = Uw[a, w_index(i, j)] + t
    return Uw

###############################################################################

# Function: Utilities
def compute_utilities(X, grids, w, ai):

    def w_index(i, j):
        return i * (ai - 1) + j

    X      = np.asarray(X, float)
    n, m   = X.shape
    n_w    = m * (ai - 1)
    Uw_new = np.zeros((n, n_w))

    for a in range(0, n):
        for i in range(0, m):
            val  = X[a, i]
            grid = grids[i]
            if val >= grid[-1]:
                j = len(grid) - 2
            else:
                j = np.searchsorted(grid, val, side = "right") - 1
                j = min(max(j, 0), len(grid) - 2)
            g_j  = grid[j]
            g_j1 = grid[j + 1]
            t    = 0.0 if g_j1 == g_j else (val - g_j) / (g_j1 - g_j)
            for k in range(j):
                Uw_new[a, w_index(i, k)] = Uw_new[a, w_index(i, k)]  + 1.0
            Uw_new[a, w_index(i, j)]     = Uw_new[a, w_index(i, j)]  + t
    return Uw_new.dot(w)

# Function: Predict Classes
def predict_classes(utilities, thresholds, Q = None, eps = 1e-9):
    U          = np.asarray(utilities,  float)
    thresholds = np.asarray(thresholds, float)
    if Q is None:
        Q = len(thresholds) + 1
    preds = np.empty_like(U, dtype = int)
    for i, u in enumerate(U):
        if u >= thresholds[0] - eps:
            preds[i] = 1
        else:
            assigned = Q
            for k in range(2, Q):
                lower = thresholds[k - 1]
                upper = thresholds[k - 2]
                if (u >= lower - eps) and (u < upper - eps):
                    assigned = k
                    break
            preds[i] = assigned
    return preds

# Function: Predict New Classes 
def predict_classes_new(X_new, results, ai, Q, criterion_type = []):
    X_new = np.array(X_new, float)
    if (len(criterion_type) > 0):
        for j in range(0, X_new.shape[1]):
            if (criterion_type[j] == 'min'):
                X_new[:, j] = -X_new[:, j]
    U_new      = compute_utilities(X_new, results["grids"], results["w"], ai)
    y_pred_new = predict_classes(U_new, results["thresholds"], Q)
    return U_new, y_pred_new

###############################################################################

# Function: MILP
def milp_u_iii(Uw, y, delta = 1e-3, s = 1e-3, big_M = 1e4, lambda_margin = 1e-2):
    Uw                 = np.asarray(Uw, dtype = float)
    y                  = np.asarray(y, dtype = int)
    n, n_w             = Uw.shape
    Q                  = int(y.max())
    n_t                = Q - 1
    idx_w              = slice(0, n_w)
    idx_t              = slice(idx_w.stop, idx_w.stop + n_t)
    idx_U              = slice(idx_t.stop, idx_t.stop + n)
    idx_E              = slice(idx_U.stop, idx_U.stop + n)
    idx_d              = slice(idx_E.stop, idx_E.stop + n)
    n_var              = idx_d.stop
    c                  = np.zeros(n_var)
    c[idx_E]           = 1.0                         
    c[idx_d]           = -float(lambda_margin)       
    lb                 = np.full(n_var, -np.inf)
    ub                 = np.full(n_var,  np.inf)
    lb[idx_w]          = 0.0
    lb[idx_E]          = 0.0
    ub[idx_E]          = 1.0
    lb[idx_d]          = 0.0
    bounds             = Bounds(lb, ub)
    integrality        = np.zeros(n_var, dtype = int)
    integrality[idx_E] = 1  
    A_rows             = []
    lb_rows            = []
    ub_rows            = []
    for j in range(0, n):
        row                  = np.zeros(n_var)
        row[idx_w]           = -Uw[j]
        row[idx_U.start + j] = 1.0
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(0.0)
    row        = np.zeros(n_var)
    row[idx_w] = 1.0
    A_rows.append(row)
    lb_rows.append(1.0)
    ub_rows.append(1.0)
    if n_t >= 2:
        for k in range(n_t - 1):
            row                      = np.zeros(n_var)
            row[idx_t.start + k]     =  1.0
            row[idx_t.start + k + 1] = -1.0
            A_rows.append(row)
            lb_rows.append(delta)
            ub_rows.append(np.inf)
    for j in range(0, n):
        c_j   = y[j]
        u_idx = idx_U.start + j
        e_idx = idx_E.start + j
        if c_j == 1:
            row                  = np.zeros(n_var)
            row[u_idx]           = 1.0
            row[idx_t.start + 0] = -1.0
            row[e_idx]           = big_M
            A_rows.append(row)
            lb_rows.append(s)
            ub_rows.append(np.inf)
        elif 1 < c_j < Q:
            row                          = np.zeros(n_var)
            row[u_idx]                   = 1.0
            row[idx_t.start + (c_j - 1)] = -1.0
            row[e_idx] = big_M
            A_rows.append(row)
            lb_rows.append(s)
            ub_rows.append(np.inf)
            row                          = np.zeros(n_var)
            row[u_idx]                   = -1.0
            row[idx_t.start + (c_j - 2)] = 1.0
            row[e_idx] = big_M
            A_rows.append(row)
            lb_rows.append(s)
            ub_rows.append(np.inf)
        else:  
            row                        = np.zeros(n_var)
            row[u_idx]                 = -1.0
            row[idx_t.start + (Q - 2)] = 1.0
            row[e_idx]                 = big_M
            A_rows.append(row)
            lb_rows.append(s)
            ub_rows.append(np.inf)
    for j in range(0, n):
        c_j        = y[j]
        d_idx      = idx_d.start + j
        e_idx      = idx_E.start + j
        u_idx      = idx_U.start + j
        row        = np.zeros(n_var)
        row[d_idx] = 1.0
        row[e_idx] = big_M
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(big_M)
        if c_j == 1:
            row = np.zeros(n_var)
            row[d_idx]           = 1.0
            row[u_idx]           = -1.0
            row[idx_t.start + 0] = 1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
        elif 1 < c_j < Q:
            row                          = np.zeros(n_var)
            row[d_idx]                   = 1.0
            row[u_idx]                   = -1.0
            row[idx_t.start + (c_j - 1)] = 1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            row                          = np.zeros(n_var)
            row[d_idx]                   = 1.0
            row[u_idx]                   = 1.0
            row[idx_t.start + (c_j - 2)] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
        else:
            row                          = np.zeros(n_var)
            row[d_idx]                   = 1.0
            row[u_idx]                   = 1.0
            row[idx_t.start + (Q - 2)]   = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
    A           = np.vstack(A_rows)
    lb_arr      = np.array(lb_rows, dtype = float)
    ub_arr      = np.array(ub_rows, dtype = float)
    constraints = LinearConstraint(A, lb_arr, ub_arr)
    res         = milp(c = c, constraints = constraints, bounds = bounds, integrality = integrality)
    if not res.success:
        w_opt          = np.full(n_w, np.nan)
        thresholds_opt = np.full(n_t, np.nan)
        d_opt          = np.full(n, np.nan)
        E_opt          = np.full(n, np.nan)
        return w_opt, thresholds_opt, d_opt, E_opt, res.status
    x              = res.x
    w_opt          = x[idx_w]
    thresholds_opt = x[idx_t]
    d_opt          = x[idx_d]
    E_opt          = x[idx_E]
    return w_opt, thresholds_opt, d_opt, E_opt, res.status

###############################################################################

# Function: UTADIS III
def utadis_iii_method(X, y, criterion_type = [], ai = 5, train_size = 1.0, delta = 1e-3, s = 1e-3, big_M = 1e4, lambda_margin = 1e-2, verbose = True):
    X    = np.asarray(X, dtype = float)
    y    = np.asarray(y, dtype = int)
    n, m = X.shape
    Q    = int(y.max())
    if (len(criterion_type) > 0):
        for j in range(0, X.shape[1]):
            if (criterion_type[j] == 'min'):
                X[:, j] = -X[:, j]
    labels       = None
    labels_train = None
    labels_test  = None
    if train_size < 1.0:
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X, y, np.arange(len(y)), train_size = train_size, random_state = 42, stratify = y)
    else:
        X_train, y_train = X, y
        X_test           = None
        y_test           = None
        labels_train     = None
        labels_test      = None
    grids                                   = build_grids(X_train, ai)
    Uw_train                                = build_Uw(X_train, grids, ai)
    w, thresholds, d_train, E_train, status = milp_u_iii(Uw_train, y_train, delta = delta, s = s, big_M = big_M, lambda_margin = lambda_margin)
    U_train                                 = compute_utilities(X_train, grids, w, ai)
    y_pred_train                            = predict_classes(U_train, thresholds, Q)
    train_accuracy                          = (y_pred_train == y_train).mean()
    results                                 = {"grids": grids, "w": w, "thresholds": thresholds, "utilities_train": U_train, "y_pred_train": y_pred_train, "train_accuracy": train_accuracy, "solver_status": status, "X_train": X_train, "y_train": y_train, "labels_train": labels_train}
    if X_test is not None:
        U_test        = compute_utilities(X_test, grids, w, ai)
        y_pred_test   = predict_classes(U_test, thresholds, Q)
        test_accuracy = (y_pred_test == y_test).mean()
        results.update({"X_test": X_test, "y_test": y_test, "utilities_test": U_test, "y_pred_test": y_pred_test, "test_accuracy": test_accuracy, "labels_test": labels_test})
    if verbose:
        print("=== UTADIS III ===")
        print(f"Number of alternatives (total): {n}")
        print(f"Number of criteria: {m}")
        print(f"Number of classes (Q): {Q}")
        print(f"ai (grid points per criterion): {ai}")
        print(f"delta: {delta}, s: {s}")
        if train_size < 1.0:
            print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        else:
            print("Train on full dataset (no test split).")
        print("MILP status:", status)
        print("Learned thresholds:", thresholds)
        print("Training accuracy:", train_accuracy)
        if X_test is not None:
            print("Test accuracy:", test_accuracy)
        for idx, (u_val, true_c, pred_c) in enumerate(zip(U_train, y_train, y_pred_train)):
            label  = (labels_train[idx] if (train_size < 1.0 and labels is not None) else (labels[idx] if labels is not None else idx))
            print(f"Train {label}: Class = {true_c}, Predicted = {pred_c}, U = {u_val:.6f}")
        if X_test is not None:
            print("--- Test assignments ---")
            for idx, (u_val, true_c, pred_c) in enumerate(zip(results["utilities_test"], y_test, y_pred_test)):
                label = labels_test[idx] if (labels is not None) else idx
                print(f"Test {label}: Class = {true_c}, Predicted = {pred_c}, U = {u_val:.6f}")
    return results

###############################################################################