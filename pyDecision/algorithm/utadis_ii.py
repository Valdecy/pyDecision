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
def milp_u_ii(Uw, y, ai, delta = 1e-4, s = 1e-4, big_M = 2.0, eps = 1e-6):
    
    def u_index(k):
        return n_w + (k - 1)
    
    y               = np.asarray(y, int)
    n, n_w          = Uw.shape
    Q               = int(y.max())
    n_u             = Q - 1
    next_var        = n_w + n_u
    sigma_plus_idx  = {}
    sigma_minus_idx = {}

    for a in range(0, n):
        cls = y[a]
        if cls == 1:
            sigma_plus_idx[a]  = next_var
            next_var           = next_var + 1
        elif cls == Q:
            sigma_minus_idx[a] = next_var
            next_var           = next_var + 1
        else:
            sigma_plus_idx[a]  = next_var
            next_var           = next_var + 1
            sigma_minus_idx[a] = next_var
            next_var           = next_var + 1

    E_idx = {}
    for a in range(0, n):
        E_idx[a] = next_var
        next_var = next_var + 1
    n_vars = next_var
    c      = np.zeros(n_vars, dtype = float)
    for a in range(0, n):
        c[E_idx[a]] = 1.0
    lb = np.zeros(n_vars, dtype = float)
    ub = np.full(n_vars, np.inf, dtype = float)
    for k in range(1, Q):
        idx     = u_index(k)
        ub[idx] = 1.0
    for idx in sigma_plus_idx.values():
        ub[idx] = big_M
    for idx in sigma_minus_idx.values():
        ub[idx] = big_M
    integrality = np.zeros(n_vars, dtype = int)
    for a in range(0, n):
        idx              = E_idx[a]
        ub[idx]          = 1.0
        integrality[idx] = 1
    bounds        = Bounds(lb, ub)
    A_eq          = np.zeros((1, n_vars), dtype = float)
    A_eq[0, :n_w] = 1.0
    b_eq          = np.array([1.0], dtype = float)
    A_ub_rows     = []
    b_ub          = []
    for k in range(2, Q):
        row                 = np.zeros(n_vars, dtype = float)
        row[u_index(k - 1)] = -1.0   
        row[u_index(k)]     =  1.0   
        A_ub_rows.append(row)
        b_ub.append(-s)
    for a in range(0, n):
        cls    = y[a]
        Uw_row = Uw[a, :]
        if cls == 1:
            sp_idx          = sigma_plus_idx[a]
            row             = np.zeros(n_vars, dtype = float)
            row[:n_w]       = -Uw_row
            row[u_index(1)] = row[u_index(1)] + 1.0
            row[sp_idx]     = row[sp_idx] - 1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)
            row             = np.zeros(n_vars, dtype = float)
            row[sp_idx]     = 1.0
            row[E_idx[a]]   = -big_M
            A_ub_rows.append(row)
            b_ub.append(0.0)
            row             = np.zeros(n_vars, dtype = float)
            row[sp_idx]     = -1.0
            row[E_idx[a]]   = eps
            A_ub_rows.append(row)
            b_ub.append(0.0)
        elif cls == Q:
            sm_idx              = sigma_minus_idx[a]
            row                 = np.zeros(n_vars, dtype = float)
            row[:n_w]           = Uw_row
            row[sm_idx]         = row[sm_idx] - 1.0
            row[u_index(Q - 1)] = row[u_index(Q - 1)] - 1.0
            A_ub_rows.append(row)
            b_ub.append(-delta)
            row                 = np.zeros(n_vars, dtype = float)
            row[sm_idx]         = 1.0
            row[E_idx[a]]       = -big_M
            A_ub_rows.append(row)
            b_ub.append(0.0)
            row                 = np.zeros(n_vars, dtype = float)
            row[sm_idx]         = -1.0
            row[E_idx[a]]       = eps
            A_ub_rows.append(row)
            b_ub.append(0.0)
        else:
            k                   = cls
            sp_idx              = sigma_plus_idx[a]
            sm_idx              = sigma_minus_idx[a]
            row                 = np.zeros(n_vars, dtype = float)
            row[:n_w]           = Uw_row
            row[sm_idx]         = row[sm_idx] - 1.0
            row[u_index(k - 1)] = row[u_index(k - 1)] - 1.0
            A_ub_rows.append(row)
            b_ub.append(-delta)
            row                 = np.zeros(n_vars, dtype = float)
            row[:n_w]           = -Uw_row 
            row[sp_idx]         = row[sp_idx] - 1.0
            row[u_index(k)]     = row[u_index(k)] + 1.0
            A_ub_rows.append(row)
            b_ub.append(0.0)
            row = np.zeros(n_vars, dtype = float)
            row[sp_idx]         = 1.0
            row[sm_idx]         = 1.0
            row[E_idx[a]]       = -big_M
            A_ub_rows.append(row)
            b_ub.append(0.0)
            row                 = np.zeros(n_vars, dtype = float)
            row[sp_idx]         = -1.0
            row[sm_idx]         = -1.0
            row[E_idx[a]]       = eps
            A_ub_rows.append(row)
            b_ub.append(0.0)
    A_ub        = np.vstack(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars), dtype = float)
    b_ub        = np.asarray(b_ub, dtype = float)
    constraints = []
    if A_eq.size > 0:
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
    if A_ub.size > 0:
        lb_ub = -np.inf * np.ones_like(b_ub)
        constraints.append(LinearConstraint(A_ub, lb_ub, b_ub))
    res = milp(c = c, integrality = integrality, bounds = bounds, constraints = constraints)
    if not res.success:
        raise RuntimeError(f"UTADIS II MILP did not converge: {res.message}")
    x_opt      = res.x
    w          = x_opt[:n_w]
    thresholds = x_opt[n_w:n_w + n_u]
    status = f"Status = {res.status}, Message = {res.message}"
    return w, thresholds, status

###############################################################################

# Function: UTADIS II
def utadis_ii_method(X, y, criterion_type = [], ai = 5, delta = 1e-4, s = 1e-4, train_size = 1.0, verbose = True):
    X = np.asarray(X, float)
    if (len(criterion_type) > 0):
        for j in range(0, X.shape[1]):
            if (criterion_type[j] == 'min'):
                X[:, j] = -X[:, j]
    y = np.asarray(y, int)
    Q = int(y.max())
    if train_size < 1.0:
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X, y, np.arange(len(y)), train_size = train_size, random_state = 42, stratify = y)
    else:
        X_train, y_train = X, y
        X_test           = None
        y_test           = None
        labels_train     = None
        labels_test      = None
    grids                 = build_grids(X_train, ai)
    Uw                    = build_Uw(X_train, grids, ai)
    w, thresholds, status = milp_u_ii(Uw, y_train, ai, delta = delta, s = s)
    U_train               = compute_utilities(X_train, grids, w, ai)
    y_pred_train          = predict_classes(U_train, thresholds, Q)
    train_accuracy        = (y_pred_train == y_train).mean()
    results               = {"grids": grids, "w": w, "thresholds": thresholds, "utilities_train": U_train, "y_pred_train": y_pred_train, "train_accuracy": train_accuracy, "solver_status": status}

    if X_test is not None:
        U_test        = compute_utilities(X_test, grids, w, ai)
        y_pred_test   = predict_classes(U_test, thresholds, Q)
        test_accuracy = (y_pred_test == y_test).mean()
        results.update({"X_test": X_test, "y_test": y_test, "utilities_test": U_test, "y_pred_test": y_pred_test, "test_accuracy": test_accuracy, "labels_train": labels_train, "labels_test": labels_test})

    if verbose:
        print("=== UTADIS II ===")
        print(f"Number of alternatives (total): {len(y)}")
        print(f"Number of criteria: {X.shape[1]}")
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
            labels = None
            label  = (labels_train[idx] if (train_size < 1.0 and labels is not None) else (labels[idx] if labels is not None else idx))
            print(f"Train {label}: Class = {true_c}, Predicted = {pred_c}, U = {u_val:.6f}")
        if X_test is not None:
            print("--- Test assignments ---")
            for idx, (u_val, true_c, pred_c) in enumerate(zip(results["utilities_test"], y_test, y_pred_test)):
                label = labels_test[idx] if (labels is not None) else idx
                print(f"Test {label}: Class = {true_c}, Predicted = {pred_c}, U = {u_val:.6f}")
    return results

###############################################################################
