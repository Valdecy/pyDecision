"""
pyDecision.web.methods
======================

Comprehensive registry + dispatcher for the pyDecision MCDA library.

Every method is described by an INPUT SHAPE which tells the front-end what
kind of editor to render (decision matrix, pairwise matrix, fuzzy cells,
ranking list, etc.). The dispatcher in `run_method` knows how to call each
algorithm correctly and returns a uniform JSON-friendly result.

To prevent stray matplotlib pop-ups, we force the Agg backend and stub
plt.show before the algorithm modules are imported.
"""
import contextlib
import json
import io
import os
import math
import traceback
import base64

# Non-interactive matplotlib backend
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None  # belt-and-suspenders

import numpy as np


# ---------------------------------------------------------------------------
# Lazy import — pulled when first needed
# ---------------------------------------------------------------------------
_ALGOS_CACHE = None

def _load_algorithms():
    global _ALGOS_CACHE
    if _ALGOS_CACHE is not None:
        return _ALGOS_CACHE
    from pyDecision import algorithm as A
    _ALGOS_CACHE = A
    return A


# ===========================================================================
# REGISTRY
# ===========================================================================

REGISTRY = [
    # ----- Distance-to-Ideal -----
    {"key":"topsis","name":"TOPSIS","group":"Distance-to-Ideal","shape":"decision_matrix",
     "summary":"Technique for Order Preference by Similarity to Ideal Solution."},
    {"key":"vikor","name":"VIKOR","group":"Distance-to-Ideal","shape":"decision_matrix",
     "summary":"Compromise ranking — VIseKriterijumska Optimizacija.",
     "params":[{"key":"strategy_coefficient","label":"Strategy ν","type":"number",
                "default":0.5,"min":0,"max":1,"step":0.05}]},
    {"key":"codas","name":"CODAS","group":"Distance-to-Ideal","shape":"decision_matrix",
     "summary":"Combinative Distance-based ASsessment.",
     "params":[{"key":"lmbd","label":"Threshold τ","type":"number",
                "default":0.02,"min":0.01,"max":0.05,"step":0.005}]},
    {"key":"spotis","name":"SPOTIS","group":"Distance-to-Ideal","shape":"decision_matrix",
     "summary":"Stable Preference Ordering Towards Ideal Solution."},

    # ----- Ratio & Utility -----
    {"key":"saw","name":"SAW","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Simple Additive Weighting (Weighted Sum)."},
    {"key":"waspas","name":"WASPAS","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Weighted Aggregated Sum Product Assessment.",
     "params":[{"key":"lambda_value","label":"λ","type":"number",
                "default":0.5,"min":0,"max":1,"step":0.05}]},
    {"key":"moora","name":"MOORA","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Multi-Objective Optimization on basis of Ratio Analysis."},
    {"key":"moosra","name":"MOOSRA","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Multi-Objective Optimization on basis of Simple Ratio Analysis."},
    {"key":"multimoora","name":"MULTIMOORA","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Aggregates ratio, reference-point and multiplicative MOORA."},
    {"key":"aras","name":"ARAS","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Additive Ratio ASsessment."},
    {"key":"copras","name":"COPRAS","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"COmplex PRoportional ASsessment."},
    {"key":"edas","name":"EDAS","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Evaluation based on Distance from Average Solution."},
    {"key":"marcos","name":"MARCOS","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Measurement of Alternatives and Ranking based on Compromise Solution."},
    {"key":"mabac","name":"MABAC","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Multi-Attributive Border Approximation area Comparison.",
     "ignores_weights":True},
    {"key":"mairca","name":"MAIRCA","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Multi-Attributive Ideal-Real Comparative Analysis."},
    {"key":"mara","name":"MARA","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Magnitude of the Area for the Ranking of Alternatives."},
    {"key":"ocra","name":"OCRA","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Operational Competitiveness RAtings."},
    {"key":"piv","name":"PIV","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Proximity Indexed Value."},
    {"key":"rov","name":"ROV","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Range Of Value."},
    {"key":"lmaw","name":"LMAW","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Logarithm Methodology of Additive Weights."},
    {"key":"macbeth","name":"MACBETH","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Measuring Attractiveness by a Categorical-Based Evaluation Technique."},
    {"key":"wisp","name":"WISP","group":"Ratio & Utility","shape":"decision_matrix",
     "summary":"Integrated Simple Weighted Sum Product."},
    {"key":"psi","name":"PSI","group":"Ratio & Utility","shape":"decision_matrix_no_weights",
     "summary":"Preference Selection Index — purely data driven."},
    {"key":"maut","name":"MAUT","group":"Ratio & Utility","shape":"decision_matrix_utility",
     "summary":"Multi-Attribute Utility Theory with named utility curves."},
    {"key":"smart","name":"SMART","group":"Ratio & Utility","shape":"smart",
     "summary":"Simple Multi-Attribute Rating Technique with grade scaling."},

    # ----- Compromise -----
    {"key":"cocoso","name":"CoCoSo","group":"Compromise","shape":"decision_matrix",
     "summary":"Combined Compromise Solution.",
     "params":[{"key":"L","label":"Strategy L","type":"number",
                "default":0.5,"min":0,"max":1,"step":0.05}]},
    {"key":"cradis","name":"CRADIS","group":"Compromise","shape":"decision_matrix",
     "summary":"Compromise Ranking of Alternatives from Distance to Ideal Solution."},
    {"key":"rafsi","name":"RAFSI","group":"Compromise","shape":"rafsi",
     "summary":"Ranking of Alternatives through Functional mapping of criterion sub-Intervals.",
     "params":[
         {"key":"n_i","label":"Min interval n_i","type":"number","default":1,"step":1},
         {"key":"n_k","label":"Max interval n_k","type":"number","default":6,"step":1}]},

    # ----- Behavioural -----
    {"key":"gra","name":"GRA","group":"Behavioural","shape":"decision_matrix",
     "summary":"Grey Relational Analysis.",
     "params":[{"key":"epsilon","label":"ζ","type":"number",
                "default":0.5,"min":0.1,"max":1,"step":0.1}]},
    {"key":"todim","name":"TODIM","group":"Behavioural","shape":"decision_matrix",
     "summary":"TOmada de Decisão Interativa Multicritério (Prospect-theory-based).",
     "params":[{"key":"teta","label":"θ (loss aversion)","type":"number",
                "default":1.0,"min":0.1,"max":5,"step":0.1}]},
    {"key":"oreste","name":"ORESTE","group":"Behavioural","shape":"decision_matrix",
     "summary":"Organisation, Rangement Et SynThèsE de données rElationnelles."},

    # ----- Aggregation (rank-based) -----
    {"key":"borda","name":"Borda","group":"Aggregation","shape":"decision_matrix",
     "summary":"Borda count from a decision matrix (criterion = voter)."},
    {"key":"copeland","name":"Copeland","group":"Aggregation","shape":"decision_matrix",
     "summary":"Copeland pairwise-victory aggregation."},
    {"key":"regime","name":"REGIME","group":"Aggregation","shape":"decision_matrix",
     "summary":"REGIME outranking — pairwise concordance dominance."},

    # ----- Pairwise weight derivation -----
    {"key":"ahp","name":"AHP","group":"Pairwise","shape":"pairwise",
     "summary":"Analytic Hierarchy Process — Saaty's pairwise comparisons.",
     "params":[{"key":"wd","label":"Derivation","type":"select","default":"geometric",
                "options":[["geometric","Geometric mean"],["mean","Arithmetic mean"],
                           ["max_eigen","Principal eigenvector"]]}]},
    {"key":"fuzzy_ahp","name":"Fuzzy AHP","group":"Pairwise","shape":"fuzzy_pairwise",
     "summary":"AHP with triangular fuzzy comparisons (Buckley)."},
    {"key":"ppf_ahp","name":"PPF-AHP","group":"Pairwise","shape":"ppf_pairwise",
     "summary":"Proportional Probabilistic Fuzzy AHP."},
    {"key":"anp","name":"ANP","group":"Pairwise","shape":"supermatrix",
     "summary":"Analytic Network Process — limit-supermatrix.",
     "params":[
         {"key":"max_iter","label":"Max iter","type":"number",
          "default":100,"min":10,"max":1000,"step":10},
         {"key":"cesaro","label":"Cesaro mean","type":"select","default":"true",
          "options":[["true","Yes"],["false","No"]]}]},

    # ----- Influence -----
    {"key":"dematel","name":"DEMATEL","group":"Influence","shape":"influence_matrix",
     "summary":"Decision Making Trial and Evaluation Laboratory."},
    {"key":"fuzzy_dematel","name":"Fuzzy DEMATEL","group":"Influence","shape":"fuzzy_influence_matrix",
     "summary":"DEMATEL with triangular fuzzy direct-influence judgments."},
    {"key":"wings","name":"WINGS","group":"Influence","shape":"influence_matrix",
     "summary":"Weighted Influence Non-linear Gauge System."},

    # ----- Best-Worst -----
    {"key":"bwm","name":"BWM","group":"Best-Worst","shape":"bwm",
     "summary":"Best-Worst Method (Rezaei).",
     "params":[{"key":"eps_penalty","label":"ε penalty","type":"number",
                "default":1,"min":0.1,"max":10,"step":0.1}]},
    {"key":"bwm_s","name":"BWM-S","group":"Best-Worst","shape":"bwm",
     "summary":"Simplified Best-Worst Method.",
     "params":[{"key":"alpha","label":"α","type":"number",
                "default":0.5,"min":0,"max":1,"step":0.05}]},
    {"key":"fuzzy_bwm","name":"Fuzzy BWM","group":"Best-Worst","shape":"fuzzy_bwm",
     "summary":"Best-Worst Method with triangular fuzzy preferences.",
     "params":[{"key":"eps_penalty","label":"ε penalty","type":"number",
                "default":1,"min":0.1,"max":10,"step":0.1}]},

    # ----- Criteria-rank weighting -----
    {"key":"fucom","name":"FUCOM","group":"Criteria-Rank","shape":"fucom",
     "summary":"Full Consistency Method — rank-then-priority weighting."},
    {"key":"fuzzy_fucom","name":"Fuzzy FUCOM","group":"Criteria-Rank","shape":"fuzzy_fucom",
     "summary":"FUCOM with triangular fuzzy priorities.",
     "params":[{"key":"n_starts","label":"Solver restarts","type":"number",
                "default":250,"min":10,"max":1000,"step":10}]},
    {"key":"roc","name":"ROC","group":"Criteria-Rank","shape":"criteria_rank",
     "summary":"Rank Order Centroid weights."},
    {"key":"rrw","name":"RRW","group":"Criteria-Rank","shape":"criteria_rank",
     "summary":"Rank Reciprocal Weights."},
    {"key":"rsw","name":"RSW","group":"Criteria-Rank","shape":"criteria_rank",
     "summary":"Rank Sum Weights."},
    {"key":"rancom","name":"RANCOM","group":"Criteria-Rank","shape":"criteria_rank",
     "summary":"RANking COMparison weighting."},

    # ----- Data-driven Weighting -----
    {"key":"entropy","name":"Entropy","group":"Weighting","shape":"weighting_only",
     "summary":"Shannon-entropy based objective weights.","weighting":True},
    {"key":"critic","name":"CRITIC","group":"Weighting","shape":"weighting_only",
     "summary":"Criteria Importance Through Inter-Criteria Correlation.","weighting":True},
    {"key":"merec","name":"MEREC","group":"Weighting","shape":"weighting_only",
     "summary":"Method based on Removal Effects of Criteria.","weighting":True},
    {"key":"cilos","name":"CILOS","group":"Weighting","shape":"weighting_only",
     "summary":"Criterion Impact LOSs.","weighting":True},
    {"key":"idocriw","name":"IDOCRIW","group":"Weighting","shape":"weighting_only",
     "summary":"Integrated Determination of Objective CRIteria Weights.","weighting":True},
    {"key":"mpsi","name":"MPSI","group":"Weighting","shape":"weighting_only",
     "summary":"Modified Preference Selection Index — variance based.","weighting":True},
    {"key":"fuzzy_critic","name":"Fuzzy CRITIC","group":"Weighting","shape":"fuzzy_weighting_only",
     "summary":"CRITIC adapted to triangular fuzzy decision matrices.","weighting":True},
    {"key":"fuzzy_merec","name":"Fuzzy MEREC","group":"Weighting","shape":"fuzzy_weighting_only",
     "summary":"MEREC adapted to triangular fuzzy decision matrices.","weighting":True},
    {"key":"seca","name":"SECA","group":"Weighting","shape":"decision_matrix_no_weights",
     "summary":"Simultaneous Evaluation of Criteria and Alternatives — returns weights.",
     "weighting":True,
     "params":[{"key":"beta","label":"β","type":"number",
                "default":3.0,"min":0.1,"max":10,"step":0.1}]},

    # ----- ELECTRE family -----
    {"key":"electre_i","name":"ELECTRE I","group":"ELECTRE","shape":"decision_matrix",
     "summary":"ELECTRE I — concordance/discordance kernel.",
     "params":[
         {"key":"c_hat","label":"c̄","type":"number","default":0.75,"min":0,"max":1,"step":0.05},
         {"key":"d_hat","label":"d̄","type":"number","default":0.50,"min":0,"max":1,"step":0.05}]},
    {"key":"electre_ii","name":"ELECTRE II","group":"ELECTRE","shape":"decision_matrix",
     "summary":"ELECTRE II — strong/weak outranking.",
     "params":[
         {"key":"c_minus","label":"c⁻","type":"number","default":0.65,"min":0,"max":1,"step":0.05},
         {"key":"c_zero","label":"c⁰","type":"number","default":0.75,"min":0,"max":1,"step":0.05},
         {"key":"c_plus","label":"c⁺","type":"number","default":0.85,"min":0,"max":1,"step":0.05},
         {"key":"d_minus","label":"d⁻","type":"number","default":0.25,"min":0,"max":1,"step":0.05},
         {"key":"d_plus","label":"d⁺","type":"number","default":0.50,"min":0,"max":1,"step":0.05}]},
    {"key":"electre_iii","name":"ELECTRE III","group":"ELECTRE","shape":"electre_qpv",
     "summary":"ELECTRE III — fuzzy outranking with Q/P/V thresholds."},
    {"key":"electre_iv","name":"ELECTRE IV","group":"ELECTRE","shape":"electre_qpv_no_w",
     "summary":"ELECTRE IV — outranking without criterion weights."},
    {"key":"electre_i_s","name":"ELECTRE I-S","group":"ELECTRE","shape":"electre_qpv",
     "summary":"ELECTRE I-S — concordance with Q/P/V thresholds.",
     "params":[{"key":"lambda_value","label":"λ","type":"number",
                "default":0.5,"min":0,"max":1,"step":0.05}]},
    {"key":"electre_i_v","name":"ELECTRE I-V","group":"ELECTRE","shape":"electre_v",
     "summary":"ELECTRE I — concordance with veto thresholds per criterion.",
     "params":[{"key":"c_hat","label":"c̄","type":"number",
                "default":0.75,"min":0,"max":1,"step":0.05}]},

    # ----- ELECTRE Tri (sorting) -----
    {"key":"electre_tri_b","name":"ELECTRE Tri-B","group":"Sorting","shape":"electre_tri",
     "summary":"ELECTRE Tri-B — sorting using boundary profiles.",
     "params":[{"key":"cut_level","label":"λ cut","type":"number",
                "default":0.75,"min":0.5,"max":1,"step":0.05}]},
    {"key":"electre_tri_c","name":"ELECTRE Tri-C","group":"Sorting","shape":"electre_tri_central",
     "summary":"ELECTRE Tri-C — sorting with central reference profiles.",
     "params":[{"key":"cut_level","label":"λ cut","type":"number",
                "default":0.6,"min":0.5,"max":1,"step":0.05}]},

    # ----- PROMETHEE -----
    {"key":"promethee_i","name":"PROMETHEE I","group":"PROMETHEE","shape":"promethee",
     "summary":"PROMETHEE I — partial preorder from φ⁺ and φ⁻ flows."},
    {"key":"promethee_ii","name":"PROMETHEE II","group":"PROMETHEE","shape":"promethee",
     "summary":"PROMETHEE II — net flow (complete ranking)."},
    {"key":"promethee_iii","name":"PROMETHEE III","group":"PROMETHEE","shape":"promethee",
     "summary":"PROMETHEE III — interval-based ranking.",
     "params":[{"key":"lmbd","label":"λ","type":"number",
                "default":0.15,"min":0,"max":1,"step":0.05}]},
    {"key":"promethee_iv","name":"PROMETHEE IV","group":"PROMETHEE","shape":"promethee",
     "summary":"PROMETHEE IV — continuous version.",
     "params":[{"key":"steps","label":"Steps","type":"number",
                "default":0.001,"min":0.0001,"max":0.01,"step":0.0001}]},
    {"key":"promethee_v","name":"PROMETHEE V","group":"PROMETHEE","shape":"promethee",
     "summary":"PROMETHEE V — net-flow ranking with portfolio selection constraints.",
     "params":[
         {"key":"max_selected","label":"Max selected","type":"number",
          "default":2,"min":1,"max":50,"step":1},
         {"key":"iterations","label":"GA iterations","type":"number",
          "default":500,"min":50,"max":5000,"step":50}]},
    {"key":"promethee_vi","name":"PROMETHEE VI","group":"PROMETHEE","shape":"promethee_vi",
     "summary":"PROMETHEE VI — ranking under interval-valued criterion weights.",
     "params":[
         {"key":"iterations","label":"Monte Carlo iterations","type":"number",
          "default":1000,"min":100,"max":10000,"step":100}]},
    {"key":"promethee_gaia","name":"PROMETHEE-GAIA","group":"PROMETHEE","shape":"promethee",
     "summary":"PROMETHEE with GAIA principal-component analysis."},

    # ----- Sorting -----
    {"key":"flowsort","name":"FlowSort","group":"Sorting","shape":"flowsort",
     "summary":"FlowSort — PROMETHEE-based assignment to ordered classes."},
    {"key":"cpp_tri","name":"CPP-Tri","group":"Sorting","shape":"cpp_tri",
     "summary":"Composition of Probabilistic Preferences for sorting.",
     "params":[
         {"key":"rule","label":"Rule","type":"select","default":"central",
          "options":[["central","Central"],["benevolent","Benevolent"],["strict","Strict"]]},
         {"key":"dist","label":"Distribution","type":"select","default":"normal",
          "options":[["normal","Normal"],["logit","Logit"],["beta-pert","Beta-PERT"],["empirical","Empirical"]]}]},
    {"key":"utadis_i","name":"UTADIS I","group":"Sorting","shape":"utadis",
     "summary":"UTADIS I — additive utility disaggregation for ordered class assignment.",
     "params":[
         {"key":"ai","label":"Grid points (ai)","type":"number",
          "default":5,"min":2,"max":20,"step":1},
         {"key":"delta","label":"δ","type":"number",
          "default":0.0001,"min":0.000001,"max":0.1,"step":0.0001},
         {"key":"s","label":"s","type":"number",
          "default":0.0001,"min":0.000001,"max":0.1,"step":0.0001}]},
    {"key":"utadis_ii","name":"UTADIS II","group":"Sorting","shape":"utadis",
     "summary":"UTADIS II — mixed-integer disaggregation for ordered class assignment.",
     "params":[
         {"key":"ai","label":"Grid points (ai)","type":"number",
          "default":5,"min":2,"max":20,"step":1},
         {"key":"delta","label":"δ","type":"number",
          "default":0.0001,"min":0.000001,"max":0.1,"step":0.0001},
         {"key":"s","label":"s","type":"number",
          "default":0.0001,"min":0.000001,"max":0.1,"step":0.0001}]},
    {"key":"utadis_iii","name":"UTADIS III","group":"Sorting","shape":"utadis",
     "summary":"UTADIS III — MILP-based disaggregation with assignment slack variables.",
     "params":[
         {"key":"ai","label":"Grid points (ai)","type":"number",
          "default":5,"min":2,"max":20,"step":1},
         {"key":"delta","label":"δ","type":"number",
          "default":0.001,"min":0.000001,"max":0.1,"step":0.0001},
         {"key":"s","label":"s","type":"number",
          "default":0.001,"min":0.000001,"max":0.1,"step":0.0001},
         {"key":"big_M","label":"Big-M","type":"number",
          "default":10000,"min":10,"max":1000000,"step":10},
         {"key":"lambda_margin","label":"λ margin","type":"number",
          "default":0.01,"min":0.000001,"max":1,"step":0.001}]},

    # ----- Specialised -----
    {"key":"odo_ovo","name":"ODO-OVO","group":"Specialised","shape":"decision_matrix_critical",
     "summary":"Optimal-Direction / Optimal-Value Orientation method.",
     "params":[
         {"key":"rank_by","label":"Rank by","type":"select","default":"l2",
          "options":[["l2","L2 distance"],["l1","L1 distance"]]},
         {"key":"weighted","label":"Weighted","type":"select","default":"false",
          "options":[["false","No"],["true","Yes"]]}]},
    {"key":"opa","name":"OPA","group":"Specialised","shape":"opa",
     "summary":"Ordinal Priority Approach — ranks experts × criteria × alternatives."},

    # ----- Fuzzy ranking -----
    {"key":"fuzzy_topsis","name":"Fuzzy TOPSIS","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"TOPSIS with triangular fuzzy performance scores."},
    {"key":"fuzzy_vikor","name":"Fuzzy VIKOR","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"VIKOR with triangular fuzzy performance scores.",
     "params":[{"key":"strategy_coefficient","label":"Strategy ν","type":"number",
                "default":0.5,"min":0,"max":1,"step":0.05}]},
    {"key":"fuzzy_aras","name":"Fuzzy ARAS","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"ARAS with triangular fuzzy performance scores."},
    {"key":"fuzzy_copras","name":"Fuzzy COPRAS","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"COPRAS with triangular fuzzy performance scores."},
    {"key":"fuzzy_edas","name":"Fuzzy EDAS","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"EDAS with triangular fuzzy performance scores."},
    {"key":"fuzzy_moora","name":"Fuzzy MOORA","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"MOORA with triangular fuzzy performance scores."},
    {"key":"fuzzy_ocra","name":"Fuzzy OCRA","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"OCRA with triangular fuzzy performance scores."},
    {"key":"fuzzy_waspas","name":"Fuzzy WASPAS","group":"Fuzzy Ranking","shape":"fuzzy_decision_matrix",
     "summary":"WASPAS with triangular fuzzy performance scores."},

    # ----- Additional implemented methods -----
    {"key":"electre_tri_nb","name":"ELECTRE Tri-nB","group":"Sorting","shape":"electre_tri",
     "summary":"ELECTRE Tri-nB — sorting with nested boundary profiles.",
     "params":[{"key":"cut_level","label":"λ cut","type":"number",
                "default":0.75,"min":0.5,"max":1,"step":0.05}]},
    {"key":"electre_tri_nc","name":"ELECTRE Tri-nC","group":"Sorting","shape":"electre_tri_central",
     "summary":"ELECTRE Tri-nC — sorting with nested central reference profiles.",
     "params":[{"key":"cut_level","label":"λ cut","type":"number",
                "default":0.6,"min":0.5,"max":1,"step":0.05}]},
    {"key":"ec_promethee","name":"EC-PROMETHEE","group":"PROMETHEE","shape":"promethee",
     "summary":"Entropy–CRITIC PROMETHEE with iterative weight uncertainty aggregation.",
     "params":[{"key":"iterations","label":"Iterations","type":"number",
                "default":50,"min":10,"max":500,"step":10}]},
    {"key":"lara","name":"LaRa","group":"Ranking","shape":"decision_matrix",
     "summary":"Laplacian-regularised ranking with dominance-aware graph smoothing."},
    {"key":"sabina","name":"SABINA","group":"Ranking","shape":"decision_matrix",
     "summary":"Scale-adaptive noncompensatory preference aggregation."},
]

CATEGORY_ORDER = [
    "Criteria Weighting",
    "Choice-focused",
    "Consensus",
    "Ranking",
    "Ordered classes",
    "Structural Analysis",
]

_CATEGORY_BY_KEY = {
    # Criteria weighting
    "bwm": "Criteria Weighting",
    "bwm_s": "Criteria Weighting",
    "fuzzy_bwm": "Criteria Weighting",
    "cilos": "Criteria Weighting",
    "critic": "Criteria Weighting",
    "fuzzy_critic": "Criteria Weighting",
    "entropy": "Criteria Weighting",
    "fucom": "Criteria Weighting",
    "fuzzy_fucom": "Criteria Weighting",
    "idocriw": "Criteria Weighting",
    "lmaw": "Criteria Weighting",
    "merec": "Criteria Weighting",
    "fuzzy_merec": "Criteria Weighting",
    "mpsi": "Criteria Weighting",
    "rancom": "Criteria Weighting",
    "roc": "Criteria Weighting",
    "rrw": "Criteria Weighting",
    "rsw": "Criteria Weighting",
    "seca": "Criteria Weighting",

    # Choice-focused
    "electre_i": "Choice-focused",
    "electre_i_s": "Choice-focused",
    "electre_i_v": "Choice-focused",
    "odo_ovo": "Choice-focused",

    # Consensus
    "borda": "Consensus",
    "copeland": "Consensus",

    # Ordered classes
    "cpp_tri": "Ordered classes",
    "electre_tri_b": "Ordered classes",
    "electre_tri_nb": "Ordered classes",
    "electre_tri_c": "Ordered classes",
    "electre_tri_nc": "Ordered classes",
    "flowsort": "Ordered classes",
    "utadis_i": "Ordered classes",
    "utadis_ii": "Ordered classes",
    "utadis_iii": "Ordered classes",

    # Structural analysis
    "promethee_gaia": "Structural Analysis",
    "dematel": "Structural Analysis",
    "fuzzy_dematel": "Structural Analysis",
    "wings": "Structural Analysis",
}

_NAME_OVERRIDES = {
    "bwm_s": "Simplified BWM",
    "ppf_ahp": "PPF AHP",
    "promethee_gaia": "PROMETHEE Gaia",
}

def _normalised_registry():
    out = []
    for item in REGISTRY:
        m = dict(item)
        m["group"] = _CATEGORY_BY_KEY.get(m["key"], "Ranking")
        if m["key"] in _NAME_OVERRIDES:
            m["name"] = _NAME_OVERRIDES[m["key"]]
        out.append(m)
    return out


_EXAMPLES_CACHE = None

def _load_examples():
    global _EXAMPLES_CACHE
    if _EXAMPLES_CACHE is not None:
        return _EXAMPLES_CACHE
    try:
        example_path = os.path.join(os.path.dirname(__file__), "examples.json")
        with open(example_path, "r", encoding="utf-8") as f:
            _EXAMPLES_CACHE = json.load(f)
    except Exception:
        _EXAMPLES_CACHE = {}
    return _EXAMPLES_CACHE


def get_example(key):
    return _load_examples().get(key)



# ===========================================================================
# Public registry helpers
# ===========================================================================

def list_methods():
    """Return all entries sorted alphabetically and grouped by the UI categories."""
    registry = _normalised_registry()
    alpha = sorted(registry, key=lambda m: m["name"].lower())
    grouped = {}
    for m in alpha:
        grouped.setdefault(m["group"], []).append(m)
    groups = [g for g in CATEGORY_ORDER if g in grouped]
    groups.extend(sorted(g for g in grouped.keys() if g not in CATEGORY_ORDER))
    return {
        "alphabetical": alpha,
        "grouped": grouped,
        "groups": groups,
    }


def get_method(key):
    for m in _normalised_registry():
        if m["key"] == key:
            return m
    return None


# ===========================================================================
# Output normalisation helpers
# ===========================================================================

def _flow_to_scores(flow, n, higher_is_better=True):
    """Normalise different return shapes to a flat list of length n in
    alternative-index order.

    Many pyDecision methods return one of:
       - 1-D ndarray of scores (already in alt order)
       - 2-D ndarray (n,2) of [alt_idx, score]   — alt_idx may be float or 'a1'/'A1'
       - list of (label, score) pairs
    """
    arr = np.asarray(flow, dtype=object)
    # Quick path: already 1-D numeric of length n
    if arr.ndim == 1:
        try:
            return [float(x) for x in arr]
        except (TypeError, ValueError):
            pass

    # 2-D path
    scores = np.zeros(n)
    if arr.ndim == 2:
        for row in arr:
            try:
                v0 = row[0]
                if isinstance(v0, str):
                    digits = ''.join(ch for ch in v0 if ch.isdigit())
                    idx = int(digits) - 1 if digits else -1
                else:
                    idx = int(v0) - 1
                if 0 <= idx < n:
                    scores[idx] = float(row[1])
            except (ValueError, TypeError, IndexError):
                continue
        return scores.tolist()

    # Last resort: try ravel
    try:
        flat = np.asarray(flow, dtype=float).ravel()
        if flat.size == n:
            return flat.tolist()
    except Exception:
        pass
    return scores.tolist()


def _make_ranking(scores, higher_is_better=True):
    """Return [{alternative_index, score, rank}, ...] sorted by rank.
    None / NaN scores sort to the bottom and serialize as None."""
    n = len(scores)
    sentinel = -np.inf if higher_is_better else np.inf
    arr = np.array([sentinel if (v is None or (isinstance(v, float) and math.isnan(v)))
                    else float(v) for v in scores], dtype=float)
    order = np.argsort(-arr if higher_is_better else arr)
    out = []
    for rank, idx in enumerate(order, start=1):
        raw = scores[int(idx)]
        out.append({
            "alternative_index": int(idx),
            "score": _safe_float(raw),
            "rank": int(rank),
        })
    return out


def _by_alternative(scores, higher_is_better=True):
    """Per-alternative ranking aligned to original alt order."""
    n = len(scores)
    sentinel = -np.inf if higher_is_better else np.inf
    arr = np.array([sentinel if (v is None or (isinstance(v, float) and math.isnan(v)))
                    else float(v) for v in scores], dtype=float)
    order = np.argsort(-arr if higher_is_better else arr)
    rank_of = np.zeros(n, dtype=int)
    for r, i in enumerate(order, start=1):
        rank_of[i] = r
    return [{"alternative_index": i, "score": _safe_float(scores[i]),
             "rank": int(rank_of[i])} for i in range(n)]


def _ok(method, scores, higher_is_better=True, stdout="", extras=None):
    safe_scores = [_safe_float(x) for x in scores]
    return {
        "ok": True,
        "method": method,
        "stdout": stdout,
        "result_kind": "ranking",
        "scores": safe_scores,
        "ranking": _make_ranking(safe_scores, higher_is_better),
        "by_alternative": _by_alternative(safe_scores, higher_is_better),
        "higher_is_better": bool(higher_is_better),
        "extras": extras or {},
    }


def _err(method, exc):
    return {
        "ok": False,
        "method": method,
        "error": f"{type(exc).__name__}: {exc}",
        "trace": traceback.format_exc(limit=4),
    }


def _safe_float(v):
    """Coerce a value to float, returning None for NaN/Inf so JSON stays valid."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_jsonable(obj):
    """Recursively convert numpy arrays/scalars to JSON-friendly values.
    NaN and ±Inf are replaced by None to keep the payload valid JSON."""
    if isinstance(obj, (np.ndarray,)):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return data


def _choice_result(method, stdout="", extras=None):
    return {
        "ok": True,
        "method": method,
        "stdout": stdout,
        "result_kind": "choice",
        "extras": extras or {},
    }


def _parse_alt_token(token):
    if isinstance(token, str):
        digits = ''.join(ch for ch in token if ch.isdigit())
        return int(digits) - 1 if digits else -1
    try:
        return int(token) - 1
    except Exception:
        return -1


def _parse_alt_group(group):
    if isinstance(group, (list, tuple, np.ndarray)):
        items = group
    else:
        items = [x.strip() for x in str(group).split(';') if x.strip()]
    out = []
    for item in items:
        idx = _parse_alt_token(item)
        if idx >= 0:
            out.append(idx)
    return out


def _distillation_ranks(levels, n):
    rank_arr = [None] * n
    next_rank = 1
    for level in levels or []:
        members = _parse_alt_group(level)
        if not members:
            continue
        avg_rank = float(sum(range(next_rank, next_rank + len(members))) / len(members))
        for idx in members:
            if 0 <= idx < n:
                rank_arr[idx] = avg_rank
        next_rank += len(members)
    return rank_arr


def _preorder_groups(rank_p):
    po_string = np.array(rank_p, dtype='U50').copy()
    n = po_string.shape[0]
    alts = [f'a{i+1}' for i in range(n)]
    i = po_string.shape[0] - 1
    while i >= 0:
        merged = False
        j = po_string.shape[1] - 1
        while j >= 0:
            if i != j and po_string[i, j] == 'I':
                po_string = np.delete(po_string, i, axis=0)
                po_string = np.delete(po_string, i, axis=1)
                alts[j] = str(alts[j] + '; ' + alts[i])
                del alts[i]
                merged = True
                break
            j -= 1
        if not merged:
            i -= 1
        else:
            i = min(i - 1, po_string.shape[0] - 1)
    po_matrix = np.zeros((po_string.shape[0], po_string.shape[1]))
    for i in range(po_string.shape[0]):
        for j in range(po_string.shape[1]):
            if po_string[i, j] == 'P+':
                po_matrix[i, j] = 1
    col_sum = np.sum(po_matrix, axis=1)
    alts_rank = [x for _, x in sorted(zip(col_sum, alts))]
    if np.sum(col_sum) != 0:
        alts_rank.reverse()
    return alts_rank


def _preorder_scores(rank_p, n):
    groups = _preorder_groups(rank_p)
    scores = np.zeros(n, dtype=float)
    next_rank = 1
    for group in groups:
        members = _parse_alt_group(group)
        if not members:
            continue
        avg_rank = float(sum(range(next_rank, next_rank + len(members))) / len(members))
        score = float(n + 1 - avg_rank)
        for idx in members:
            if 0 <= idx < n:
                scores[idx] = score
        next_rank += len(members)
    return scores.tolist(), groups


def _ordered_rank_labels(rank_in):
    if not rank_in:
        return [], None
    if isinstance(rank_in[0], str):
        labels = list(rank_in)
    else:
        vals = [int(v) for v in rank_in]
        n = len(vals)
        if sorted(vals) == list(range(1, n + 1)):
            labels = [f'g{v}' for v in vals]
        else:
            pairs = sorted(enumerate(vals), key=lambda p: p[1])
            labels = [f'g{i+1}' for i, _ in pairs]
    reorder = []
    for lab in labels:
        digits = ''.join(ch for ch in str(lab) if ch.isdigit())
        reorder.append(int(digits) - 1 if digits else -1)
    return labels, reorder


def _reorder_weights_by_labels(weights_arr, labels):
    w_arr = np.asarray(weights_arr, dtype=float).ravel()
    if not labels:
        return w_arr
    ordered = np.zeros_like(w_arr)
    for pos, lab in enumerate(labels):
        digits = ''.join(ch for ch in str(lab) if ch.isdigit())
        idx = int(digits) - 1 if digits else -1
        if 0 <= idx < ordered.size and pos < w_arr.size:
            ordered[idx] = w_arr[pos]
    return ordered


# ===========================================================================
# DISPATCHERS — ranking methods
# ===========================================================================

def run_method(key, dataset=None, weights=None, criterion_type=None, params=None,
               extra_inputs=None):
    """Universal dispatcher."""
    params = params or {}
    extra_inputs = extra_inputs or {}
    A = _load_algorithms()
    buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(buf):
            X = np.asarray(dataset, dtype=float) if dataset is not None else None
            w = np.asarray(weights, dtype=float) if weights is not None else None
            types = list(criterion_type) if criterion_type is not None else None

            # --------- Distance-to-Ideal ---------
            if key == "topsis":
                scores = A.topsis_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(scores, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "vikor":
                v = float(params.get("strategy_coefficient", 0.5))
                fs, fr, fq, sol = A.vikor_method(X, w, types,
                                                 strategy_coefficient=v,
                                                 graph=False, verbose=False)
                n = X.shape[0]
                s = _flow_to_scores(fs, n)
                r = _flow_to_scores(fr, n)
                q = _flow_to_scores(fq, n)
                return _build(_ok(key, q, False, buf.getvalue(),
                                  extras={"S": s, "R": r, "Q": q,
                                          "solution_label": str(sol)}))

            if key == "codas":
                lmbd = float(params.get("lmbd", 0.02))
                flow = A.codas_method(X, w, types, lmbd=lmbd,
                                       graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "spotis":
                # Auto-bounds from data (5% margin)
                margin = (X.max(axis=0) - X.min(axis=0))
                margin = np.where(margin == 0, 1.0, margin) * 0.05
                s_min = X.min(axis=0) - margin
                s_max = X.max(axis=0) + margin
                scores = A.spotis_method(X, types, w, s_min, s_max,
                                         graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(scores, X.shape[0]),
                                  False, buf.getvalue()))

            # --------- Ratio & Utility ---------
            if key == "saw":
                flow = A.saw_method(X, types, w, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  False, buf.getvalue()))

            if key == "waspas":
                lam = float(params.get("lambda_value", 0.5))
                # waspas returns (wsm, wpm, waspas) — three 1-D arrays
                wsm, wpm, waspas_scores = A.waspas_method(X, types, w, lam, graph=False)
                return _build(_ok(key, list(map(float, waspas_scores)),
                                  True, buf.getvalue(),
                                  extras={"WSM": list(map(float, wsm)),
                                          "WPM": list(map(float, wpm)),
                                          "WASPAS": list(map(float, waspas_scores))}))

            if key == "moora":
                flow = A.moora_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "moosra":
                flow = A.moosra_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "multimoora":
                # No weights, no verbose
                f1, f2, f3 = A.multimoora_method(X, types, graph=False)
                n = X.shape[0]
                ratio = list(map(float, np.asarray(f1).ravel())) if np.asarray(f1).ndim == 1 else _flow_to_scores(f1, n)
                ref   = list(map(float, np.asarray(f2).ravel())) if np.asarray(f2).ndim == 1 else _flow_to_scores(f2, n)
                mult  = list(map(float, np.asarray(f3).ravel())) if np.asarray(f3).ndim == 1 else _flow_to_scores(f3, n)
                # Borda dominance over the three sub-rankings (higher = better)
                rk_r = np.argsort(-np.array(ratio)).argsort() + 1
                rk_e = np.argsort( np.array(ref)).argsort() + 1   # smaller is better
                rk_m = np.argsort(-np.array(mult)).argsort() + 1
                borda = (n - rk_r) + (n - rk_e) + (n - rk_m)
                return _build(_ok(key, borda.tolist(), True, buf.getvalue(),
                                  extras={"Ratio": ratio, "Reference": ref,
                                          "Multiplicative": mult}))

            if key == "aras":
                flow = A.aras_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "copras":
                flow = A.copras_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "edas":
                flow = A.edas_method(X, types, w, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "marcos":
                flow = A.marcos_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "mabac":
                # MABAC ignores weights
                flow = A.mabac_method(X, types, graph=False, verbose=False)
                # mabac returns 1-D rank vector (np)
                return _build(_ok(key, np.asarray(flow).ravel().tolist(),
                                  False, buf.getvalue()))

            if key == "mairca":
                flow = A.mairca_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  False, buf.getvalue()))

            if key == "mara":
                flow = A.mara_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "ocra":
                flow = A.ocra_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "piv":
                flow = A.piv_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  False, buf.getvalue()))

            if key == "rov":
                flow = A.rov_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "lmaw":
                flow = A.lmaw_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "macbeth":
                flow = A.macbeth_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "wisp":
                flow_full = A.wisp_method(X, types, w, simplified=False, graph=False, verbose=False)
                flow_simple = A.wisp_method(X, types, w, simplified=True, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow_full, X.shape[0]),
                                  True, buf.getvalue(),
                                  extras={"standard_scores": _flow_to_scores(flow_full, X.shape[0]),
                                          "simplified_scores": _flow_to_scores(flow_simple, X.shape[0])}))

            if key == "psi":
                # psi_method(dataset, criterion_type, graph, verbose)
                flow = A.psi_method(X, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "maut":
                # extra_inputs must contain utility_functions list
                ufs = extra_inputs.get("utility_functions") or ['linear'] * X.shape[1]
                flow = A.maut_method(X, w, types, ufs, step_size=1,
                                     graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue(),
                                  extras={"utility_functions": ufs}))

            if key == "smart":
                grades = np.asarray(extra_inputs.get("grades", [3]*X.shape[1]),
                                    dtype=float)
                lower  = np.asarray(extra_inputs.get("lower",
                                    X.min(axis=0).tolist()), dtype=float)
                upper  = np.asarray(extra_inputs.get("upper",
                                    X.max(axis=0).tolist()), dtype=float)
                flow = A.smart_method(X, grades, lower, upper, types,
                                      graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "lara":
                order, score, info = A.lara_method(X, w, types)
                scores = np.asarray(score, dtype=float).ravel()
                extras = {"order": _to_jsonable(order),
                          "info": _to_jsonable(info)}
                try:
                    fig1, _ = A.plot_lara_overview(order, score, info, alt_labels=[f"A{i+1}" for i in range(X.shape[0])])
                    extras["overview_plot"] = _fig_to_base64(fig1)
                except Exception:
                    pass
                try:
                    fig2, _ = A.plot_lara_graph(order, score, info, alt_labels=[f"A{i+1}" for i in range(X.shape[0])])
                    extras["graph_plot"] = _fig_to_base64(fig2)
                except Exception:
                    pass
                return _build(_ok(key, scores.tolist(), True, buf.getvalue(), extras=extras))

            if key == "sabina":
                labels = [f"A{i+1}" for i in range(X.shape[0])]
                order, score, info = A.sabina_method(X, w, types, labels=labels)
                scores = np.asarray(score, dtype=float).ravel()
                return _build(_ok(key, scores.tolist(), True, buf.getvalue(),
                                  extras={"order": _to_jsonable(order),
                                          "info": _to_jsonable(info)}))

            # --------- Compromise ---------
            if key == "cocoso":
                L = float(params.get("L", 0.5))
                flow = A.cocoso_method(X, types, w, L=L, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "cradis":
                flow = A.cradis_method(X, types, w, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "rafsi":
                ni = float(params.get("n_i", 1))
                nk = float(params.get("n_k", 6))
                ideal = list(np.asarray(extra_inputs.get('ideal', []), dtype=float).ravel()) if extra_inputs.get('ideal') is not None else []
                anti  = list(np.asarray(extra_inputs.get('anti_ideal', []), dtype=float).ravel()) if extra_inputs.get('anti_ideal') is not None else []
                if len(ideal) != X.shape[1] or len(anti) != X.shape[1]:
                    ideal = X.max(axis=0).tolist()
                    anti  = X.min(axis=0).tolist()
                    for j, t in enumerate(types):
                        if t == "min":
                            ideal[j], anti[j] = anti[j], ideal[j]
                flow = A.rafsi_method(X, w, types, ideal=ideal,
                                       anti_ideal=anti, n_i=ni, n_k=nk,
                                       graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            # --------- Behavioural ---------
            if key == "gra":
                eps = float(params.get("epsilon", 0.5))
                flow = A.gra_method(X, types, w, epsilon=eps,
                                    graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "todim":
                teta = float(params.get("teta", 1.0))
                flow = A.todim_method(X, types, w, teta=teta,
                                      graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "oreste":
                flow = A.oreste_method(X, w, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  False, buf.getvalue()))

            # --------- Aggregation ---------
            if key == "borda":
                # No weights param in borda_method
                flow = A.borda_method(X, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  False, buf.getvalue()))

            if key == "copeland":
                # No weights param in copeland_method
                flow = A.copeland_method(X, types, graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "regime":
                # regime returns a string preference matrix (or per-pair scores)
                result = A.regime_method(X, w, types)
                # result is a string showing pairs — we'll surface that and run
                # a fallback Borda count on row-wins for a numeric ranking.
                # Build a dominance matrix from the matrix itself
                n = X.shape[0]
                dom = np.zeros((n, n), dtype=int)
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        score = 0.0
                        for k in range(X.shape[1]):
                            diff = X[i, k] - X[j, k]
                            if types[k] == "min":
                                diff = -diff
                            score += w[k] * (1 if diff > 0 else (-1 if diff < 0 else 0))
                        if score > 0: dom[i, j] = 1
                wins = dom.sum(axis=1).astype(float).tolist()
                return _build(_ok(key, wins, True, buf.getvalue() + str(result),
                                  extras={"dominance_matrix": dom.tolist()}))

            # --------- Pairwise ---------
            if key == "ahp":
                wd = params.get("wd", "geometric")
                wd_map = {"geometric": "g", "mean": "m", "max_eigen": "max_eigen"}
                M = np.asarray(extra_inputs["matrix"], dtype=float)
                # Compute all three derivations so the UI can show them
                # side-by-side (the user's selected wd is the primary).
                ahp_modes = []
                for mode_key, mode_label in [("mean", "Arithmetic mean"),
                                              ("geometric", "Geometric mean"),
                                              ("max_eigen", "Principal eigenvector")]:
                    try:
                        w_m, rc_m = A.ahp_method(M, wd=wd_map[mode_key])
                        ahp_modes.append({
                            "mode": mode_key,
                            "label": mode_label,
                            "weights": _to_jsonable(np.asarray(w_m, dtype=float).ravel()),
                            "consistency_ratio": _safe_float(rc_m),
                        })
                    except Exception as e:
                        ahp_modes.append({"mode": mode_key, "label": mode_label,
                                          "error": f"{type(e).__name__}: {e}"})
                # Pick the requested mode as the primary result
                primary = next((m for m in ahp_modes
                                if m.get("mode") == wd and "weights" in m),
                               next((m for m in ahp_modes if "weights" in m), None))
                if primary is None:
                    raise RuntimeError("AHP failed to compute weights for any derivation mode.")
                weights_ = primary["weights"]
                rc = primary["consistency_ratio"]
                n = len(weights_)
                return _weights_result(key, weights_, buf.getvalue(),
                                       extras={"consistency_ratio": rc,
                                               "selected_mode": wd,
                                               "ahp_modes": ahp_modes,
                                               "weight_labels": [f"Criterion {i+1}" for i in range(n)]})

            if key == "fuzzy_ahp":
                M = extra_inputs["matrix"]  # list of lists of [l,m,u]
                fw, dw, nw, rc = A.fuzzy_ahp_method(M)
                return _weights_result(key, nw, buf.getvalue(),
                                       extras={
                                           "fuzzy_weights": _to_jsonable(fw),
                                           "defuzzified_weights": _to_jsonable(dw),
                                           "consistency_ratio": float(rc),
                                       })

            if key == "ppf_ahp":
                # PPF-AHP expects each cell as a (k1, k2) pair. If the user
                # provides single numbers, convert v -> (v, 0).
                M_in = extra_inputs["matrix"]
                M_pairs = []
                for row in M_in:
                    new_row = []
                    for cell in row:
                        if isinstance(cell, (list, tuple)) and len(cell) == 2:
                            new_row.append((float(cell[0]), float(cell[1])))
                        else:
                            new_row.append((float(cell), 0.0))
                    M_pairs.append(new_row)
                result = A.ppf_ahp_method(M_pairs)
                if isinstance(result, tuple):
                    weights_ = np.asarray(result[0], dtype=float).ravel()
                    rc = float(result[1]) if len(result) > 1 else None
                else:
                    weights_ = np.asarray(result, dtype=float).ravel()
                    rc = None
                extras = {"consistency_ratio": rc} if rc is not None else {}
                return _weights_result(key, weights_, buf.getvalue(), extras=extras)

            if key == "anp":
                M = np.asarray(extra_inputs["matrix"], dtype=float)
                max_iter = int(params.get("max_iter", 100))
                cesaro = str(params.get("cesaro", "true")).lower() == "true"
                L = A.anp_method(M, max_iter=max_iter, cesaro=cesaro)
                # First column gives the limit weights
                weights_ = np.asarray(L)[:, 0]
                return _weights_result(key, weights_, buf.getvalue(),
                                       extras={"limit_matrix": _to_jsonable(L),
                                               "weight_labels": [f"Element {i+1}" for i in range(len(np.asarray(weights_).ravel()))]})

            # --------- Influence ---------
            if key == "dematel":
                M = np.asarray(extra_inputs["matrix"], dtype=float)
                D_plus_R, D_minus_R, weights_ = A.dematel_method(M, size_x=0, size_y=0)
                return _weights_result(key, weights_, buf.getvalue(),
                                       extras={"prominence": _to_jsonable(D_plus_R),
                                               "relation": _to_jsonable(D_minus_R)})

            if key == "fuzzy_dematel":
                M = extra_inputs["matrix"]
                D_plus_R, D_minus_R, weights_ = A.fuzzy_dematel_method(M, size_x=0, size_y=0)
                return _weights_result(key, weights_, buf.getvalue(),
                                       extras={"prominence": _to_jsonable(D_plus_R),
                                               "relation": _to_jsonable(D_minus_R)})

            if key == "wings":
                M = np.asarray(extra_inputs["matrix"], dtype=float)
                D_plus_R, D_minus_R, weights_ = A.wings_method(M, size_x=0, size_y=0)
                return _weights_result(key, weights_, buf.getvalue(),
                                       extras={"prominence": _to_jsonable(D_plus_R),
                                               "relation": _to_jsonable(D_minus_R)})

            # --------- Best-Worst ---------
            if key == "bwm":
                mic = np.asarray(extra_inputs["mic"], dtype=float)
                lic = np.asarray(extra_inputs["lic"], dtype=float)
                eps_p = float(params.get("eps_penalty", 1))
                result = A.bw_method(mic, lic, eps_penalty=eps_p, verbose=False)
                weights_ = result[1] if isinstance(result, tuple) else result
                weights_ = np.asarray(weights_, dtype=float).ravel()
                # pyDecision.bw_method currently returns only the weights.
                # Recompute a web-friendly consistency ratio here.
                mx = int(np.max(mic)) if mic.size else 1
                if mx <= 1:
                    cr = 0.0
                else:
                    cr_vals = []
                    for i in range(mic.shape[0]):
                        cr_vals.append((mic[i] * lic[i] - mx) / (mx**2 - mx))
                    cr = float(np.max(cr_vals)) if cr_vals else 0.0
                extras = {"consistency_ratio": cr}
                return _weights_result(key, weights_, buf.getvalue(), extras=extras)

            if key == "bwm_s":
                mic = np.asarray(extra_inputs["mic"], dtype=float)
                lic = np.asarray(extra_inputs["lic"], dtype=float)
                alpha = float(params.get("alpha", 0.5))
                result = A.simplified_bw_method(mic, lic, alpha=alpha, verbose=False)
                weights_ = result[1] if isinstance(result, tuple) else result
                cr = result[0] if isinstance(result, tuple) else None
                weights_ = np.asarray(weights_, dtype=float).ravel()
                extras = {"consistency_ratio": float(cr)} if cr is not None else {}
                return _weights_result(key, weights_, buf.getvalue(), extras=extras)

            if key == "fuzzy_bwm":
                # Fuzzy BWM expects each entry in mic/lic to be one of the
                # five linguistic triples — see source. We accept the user's
                # raw triples and snap each to the nearest valid one.
                priority_tuples = [(7/2,4,9/2), (5/2,3,7/2), (3/2,2,5/2),
                                   (2/3,1,3/2), (1,1,1)]
                def _snap(v):
                    arr = tuple(map(float, v))
                    # nearest by middle element
                    best = min(priority_tuples,
                               key=lambda t: abs(t[1]-arr[1]))
                    return best
                mic = [_snap(v) for v in extra_inputs["mic"]]
                lic = [_snap(v) for v in extra_inputs["lic"]]
                eps_p = float(params.get("eps_penalty", 1))
                result = A.fuzzy_bw_method(mic, lic,
                                            eps_penalty=eps_p, verbose=False)
                # Returns (eps, cr, fuzzy_weights, weights)
                if isinstance(result, tuple) and len(result) == 4:
                    eps, cr, fw_, dw_ = result
                    return _weights_result(key, dw_, buf.getvalue(),
                                           extras={"fuzzy_weights": _to_jsonable(fw_),
                                                   "consistency_ratio": float(cr),
                                                   "epsilon": float(eps)})
                if isinstance(result, tuple) and len(result) == 2:
                    fw_, dw_ = result
                    return _weights_result(key, dw_, buf.getvalue(),
                                           extras={"fuzzy_weights": _to_jsonable(fw_)})
                return _weights_result(key, result, buf.getvalue())

            # --------- Criteria-rank weighting ---------
            if key == "fucom":
                rank_in = extra_inputs["criteria_rank"]
                priority = extra_inputs["criteria_priority"]
                labels, _ = _ordered_rank_labels(rank_in)
                weights_ = A.fucom_method(labels, priority,
                                          sort_criteria=True, verbose=False)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "fuzzy_fucom":
                rank_in = extra_inputs["criteria_rank"]
                priority = extra_inputs["criteria_priority"]
                n_starts = int(params.get("n_starts", 250))
                labels, _ = _ordered_rank_labels(rank_in)
                result = A.fuzzy_fucom_method(labels, priority,
                                              n_starts=n_starts,
                                              sort_criteria=True, verbose=False)
                fuzzy_weights = None
                crisp_weights = result
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        fuzzy_weights, crisp_weights = result[0], result[1]
                    elif len(result) == 1:
                        crisp_weights = result[0]
                weights_ = np.asarray(crisp_weights, dtype=float).ravel()
                extras = {}
                if fuzzy_weights is not None:
                    extras["fuzzy_weights"] = _to_jsonable(fuzzy_weights)
                return _weights_result(key, weights_, buf.getvalue(), extras=extras)

            if key == "roc":
                rank_in = extra_inputs["criteria_rank"]
                labels, _ = _ordered_rank_labels(rank_in)
                weights_ = A.roc_method(labels)
                weights_ = _reorder_weights_by_labels(weights_, labels)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "rrw":
                rank_in = extra_inputs["criteria_rank"]
                labels, _ = _ordered_rank_labels(rank_in)
                weights_ = A.rrw_method(labels)
                weights_ = _reorder_weights_by_labels(weights_, labels)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "rsw":
                rank_in = extra_inputs["criteria_rank"]
                labels, _ = _ordered_rank_labels(rank_in)
                weights_ = A.rsw_method(labels)
                weights_ = _reorder_weights_by_labels(weights_, labels)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "rancom":
                rank_in = extra_inputs["criteria_rank"]
                # rancom takes a "ranking" mode list of integer ranks directly
                result = A.rancom_method(rank_in, mode="ranking")
                weights_ = np.asarray(result.weights, dtype=float)
                return _weights_result(key, weights_, buf.getvalue(),
                                       extras={"mac": _to_jsonable(result.mac)})

            # --------- Data-driven weighting ---------
            if key == "entropy":
                weights_ = A.entropy_method(X, types, graph=False, verbose=False) \
                           if _accepts(A.entropy_method, 'graph') \
                           else A.entropy_method(X, types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "critic":
                weights_ = A.critic_method(X, types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "merec":
                weights_ = A.merec_method(X, types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "cilos":
                weights_ = A.cilos_method(X, types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "idocriw":
                weights_ = A.idocriw_method(X, types, verbose=False)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "mpsi":
                weights_ = A.mpsi_method(X, types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "fuzzy_critic":
                weights_ = A.fuzzy_critic_method(extra_inputs["matrix"], types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "fuzzy_merec":
                weights_ = A.fuzzy_merec_method(extra_inputs["matrix"], types)
                return _weights_result(key, weights_, buf.getvalue())

            if key == "seca":
                beta = float(params.get("beta", 3.0))
                weights_ = A.seca_method(X, types, beta=beta)
                return _weights_result(key, weights_, buf.getvalue())

            # --------- ELECTRE ---------
            if key == "electre_i":
                c_hat = float(params.get("c_hat", 0.75))
                d_hat = float(params.get("d_hat", 0.50))
                concord, discord, dominance, kernel, dominated = A.electre_i(
                    X, w, remove_cycles=False, c_hat=c_hat, d_hat=d_hat, graph=False)
                kernel_idx = [_parse_alt_token(x) for x in kernel]
                dominated_idx = [_parse_alt_token(x) for x in dominated]
                return _build(_choice_result(key, buf.getvalue(), extras={
                    "choice_only": True,
                    "concordance": _to_jsonable(concord),
                    "discordance": _to_jsonable(discord),
                    "dominance": _to_jsonable(dominance),
                    "kernel": kernel_idx,
                    "dominated": dominated_idx,
                }))

            if key == "electre_i_v":
                c_hat = float(params.get("c_hat", 0.75))
                V = np.asarray(extra_inputs["V"], dtype=float)
                concord, discord, dominance, kernel, dominated = A.electre_i_v(
                    X, V, w, remove_cycles=False, c_hat=c_hat, graph=False)
                kernel_idx = [_parse_alt_token(x) for x in kernel]
                dominated_idx = [_parse_alt_token(x) for x in dominated]
                return _build(_choice_result(key, buf.getvalue(), extras={
                    "choice_only": True,
                    "concordance": _to_jsonable(concord),
                    "discordance": _to_jsonable(discord),
                    "dominance": _to_jsonable(dominance),
                    "kernel": kernel_idx,
                    "dominated": dominated_idx,
                }))

            if key == "electre_ii":
                kw = {k: float(params.get(k, default))
                      for k, default in [("c_minus",0.65),("c_zero",0.75),
                                         ("c_plus",0.85),("d_minus",0.25),
                                         ("d_plus",0.50)]}
                ret = A.electre_ii(X, w, graph=False, **kw)
                rank_D = ret[4] if len(ret) >= 5 else None
                rank_A = ret[5] if len(ret) >= 6 else None
                rank_P = ret[7] if len(ret) >= 8 else None
                n = X.shape[0]
                scores, preorder = _preorder_scores(rank_P, n)
                desc_ranks = _distillation_ranks(rank_D, n)
                asc_ranks = _distillation_ranks(rank_A, n)
                mean_ranks = [None if (d is None and a is None) else
                              (a if d is None else d if a is None else (d + a)/2)
                              for d, a in zip(desc_ranks, asc_ranks)]
                return _build(_ok(key, scores, True, buf.getvalue(),
                                  extras={"descending": _to_jsonable(rank_D),
                                          "ascending": _to_jsonable(rank_A),
                                          "median": _to_jsonable(preorder),
                                          "preorder": _to_jsonable(rank_P),
                                          "descending_ranks": _to_jsonable(desc_ranks),
                                          "ascending_ranks": _to_jsonable(asc_ranks),
                                          "mean_ranks": _to_jsonable(mean_ranks),
                                          "hide_top": True,
                                          "hide_full_ranking": True}))

            if key == "electre_iii":
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                V = list(np.asarray(extra_inputs["V"], dtype=float).ravel())
                try:
                    gc, cred, rank_D, rank_A, rank_N, rank_P = A.electre_iii(
                        X, P=P, Q=Q, V=V, W=list(w), graph=False)
                    n = X.shape[0]
                    scores, _ = _preorder_scores(rank_P, n)
                    return _build(_ok(key, scores, True, buf.getvalue(),
                                      extras={"global_concordance": _to_jsonable(gc),
                                              "credibility": _to_jsonable(cred),
                                              "rank_D": _to_jsonable(rank_D),
                                              "rank_A": _to_jsonable(rank_A),
                                              "rank_N": _to_jsonable(rank_N),
                                              "rank_P": _to_jsonable(rank_P),
                                              "electre_iii_view": True,
                                              "hide_top": True,
                                              "hide_full_ranking": True}))
                except Exception as inner:
                    from pyDecision.algorithm.e_iii import (
                        global_concordance_matrix, credibility_matrix)
                    gc = global_concordance_matrix(X, P=P, Q=Q, W=list(w))
                    cred = credibility_matrix(X, gc, P=P, V=V)
                    score = np.asarray(cred).sum(axis=1) - np.asarray(cred).sum(axis=0)
                    return _build(_ok(key, score.tolist(), True,
                                      buf.getvalue() + f"[fallback: {inner}]\n",
                                      extras={"global_concordance": _to_jsonable(gc),
                                              "credibility": _to_jsonable(cred),
                                              "rank_D": [],
                                              "rank_A": [],
                                              "rank_N": [],
                                              "rank_P": [],
                                              "electre_iii_view": True,
                                              "hide_top": True,
                                              "hide_full_ranking": True,
                                              "fallback": True}))

            if key == "electre_iv":
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                V = list(np.asarray(extra_inputs["V"], dtype=float).ravel())
                try:
                    cred, rank_D, rank_A, rank_N, rank_P = A.electre_iv(
                        X, P=P, Q=Q, V=V, graph=False)
                    n = X.shape[0]
                    scores, _ = _preorder_scores(rank_P, n)
                    return _build(_ok(key, scores, True, buf.getvalue(),
                                      extras={"credibility": _to_jsonable(cred),
                                              "rank_D": _to_jsonable(rank_D),
                                              "rank_A": _to_jsonable(rank_A),
                                              "rank_N": _to_jsonable(rank_N),
                                              "rank_P": _to_jsonable(rank_P),
                                              "electre_iv_view": True,
                                              "hide_top": True,
                                              "hide_full_ranking": True}))
                except Exception as inner:
                    n = X.shape[0]
                    try:
                        from pyDecision.algorithm.e_iv import m_count_matrices, credibility_matrix
                        m = m_count_matrices(X, P=P, Q=Q, V=V)
                        cred = credibility_matrix(*m)
                    except Exception:
                        cred = np.zeros((n, n))
                    score = np.asarray(cred).sum(axis=1) - np.asarray(cred).sum(axis=0)
                    return _build(_ok(key, score.tolist(), True,
                                      buf.getvalue() + f"[fallback: {inner}]\n",
                                      extras={"credibility": _to_jsonable(cred),
                                              "rank_D": [],
                                              "rank_A": [],
                                              "rank_N": [],
                                              "rank_P": [],
                                              "electre_iv_view": True,
                                              "hide_top": True,
                                              "hide_full_ranking": True,
                                              "fallback": True}))

            if key == "electre_i_s":
                lam = float(params.get("lambda_value", 0.5))
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                V = list(np.asarray(extra_inputs["V"], dtype=float).ravel())
                gc, disc, kernel, cred, dominated = A.electre_i_s(
                    X, Q, P, V, list(w), graph=False, lambda_value=lam)
                kernel_idx = [_parse_alt_token(x) for x in kernel]
                dominated_idx = [_parse_alt_token(x) for x in dominated]
                return _build(_choice_result(key, buf.getvalue(), extras={
                    "choice_only": True,
                    "kernel": kernel_idx,
                    "dominated": dominated_idx,
                    "concordance": _to_jsonable(gc),
                    "discordance": _to_jsonable(disc),
                    "credibility": _to_jsonable(cred),
                }))

            # --------- ELECTRE Tri ---------
            if key == "electre_tri_b":
                cut = float(params.get("cut_level", 0.75))
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                V = list(np.asarray(extra_inputs["V"], dtype=float).ravel())
                B = [list(map(float, row)) for row in extra_inputs["B"]]
                W_list = list(np.asarray(w, dtype=float).ravel())
                pc_raw = A.electre_tri_b(X, W=W_list, Q=Q, P=P, V=V, B=B,
                                         cut_level=cut, verbose=False,
                                         rule='pc', graph=False)
                oc_raw = A.electre_tri_b(X, W=W_list, Q=Q, P=P, V=V, B=B,
                                         cut_level=cut, verbose=False,
                                         rule='oc', graph=False)
                def _tri_vec(raw):
                    cls = _to_jsonable(raw)
                    vec = [None] * X.shape[0]
                    if isinstance(cls, list):
                        if cls and isinstance(cls[0], (list, tuple)) and len(cls[0]) >= 2:
                            for row in cls:
                                try:
                                    idx = int(''.join(c for c in str(row[0]) if c.isdigit())) - 1
                                except Exception:
                                    idx = -1
                                if 0 <= idx < X.shape[0]:
                                    vec[idx] = row[-1]
                        else:
                            for i, c in enumerate(cls[:X.shape[0]]):
                                vec[i] = c
                    return vec
                class_pc = _tri_vec(pc_raw)
                class_oc = _tri_vec(oc_raw)
                return _build_sorting(key, class_pc, buf.getvalue(),
                                      extras={"classification": class_pc,
                                              "pessimistic": class_pc,
                                              "optimistic": class_oc,
                                              "classification_by_rule": {"pc": class_pc, "oc": class_oc}})

            if key == "electre_tri_c":
                cut = float(params.get("cut_level", 0.6))
                Q = np.asarray(extra_inputs["Q"], dtype=float)
                P = np.asarray(extra_inputs["P"], dtype=float)
                V = np.asarray(extra_inputs.get("V", []), dtype=float)
                C = np.asarray(extra_inputs["C"], dtype=float)
                low, high = A.electre_tri_c(X, W=w, Q=Q, P=P,
                                            V=(V if V.size else []),
                                            C=C, cut_level=cut,
                                            verbose=False)
                low_j = _to_jsonable(low)
                high_j = _to_jsonable(high)
                return _build_sorting(key, low_j, buf.getvalue(),
                                      extras={"classification": low_j,
                                              "lowest_classes": low_j,
                                              "highest_classes": high_j,
                                              "pessimistic": low_j,
                                              "optimistic": high_j})

            if key == "electre_tri_nb":
                cut = float(params.get("cut_level", 0.75))
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                V = list(np.asarray(extra_inputs["V"], dtype=float).ravel())
                B = [list(map(float, row)) for row in extra_inputs["B"]]
                W_list = list(np.asarray(w, dtype=float).ravel())
                pc, pd = A.electre_tri_nb(X, B_sets=B, w=W_list, q=Q, p=P, v=V, lam=cut)
                pc_j = _to_jsonable(pc)
                pd_j = _to_jsonable(pd)
                arr = np.asarray(pc).ravel()
                class_per_alt = arr.tolist() if arr.size == X.shape[0] else [None] * X.shape[0]
                return _build_sorting(key, class_per_alt, buf.getvalue(),
                                      extras={"classification": class_per_alt,
                                              "pessimistic": pc_j,
                                              "optimistic": pd_j,
                                              "classification_by_rule": {"pc": pc_j, "pd": pd_j}})

            if key == "electre_tri_nc":
                cut = float(params.get("cut_level", 0.6))
                Q = np.asarray(extra_inputs["Q"], dtype=float)
                P = np.asarray(extra_inputs["P"], dtype=float)
                V = np.asarray(extra_inputs.get("V", []), dtype=float)
                C = np.asarray(extra_inputs["C"], dtype=float)
                low, high = A.electre_tri_nc(X, W=w, Q=Q, P=P,
                                             V=(V if V.size else np.zeros_like(Q)),
                                             C=C, cut_level=cut,
                                             verbose=False, tol=1e-9)
                low_j = _to_jsonable(low)
                high_j = _to_jsonable(high)
                return _build_sorting(key, low_j, buf.getvalue(),
                                      extras={"classification": low_j,
                                              "lowest_classes": low_j,
                                              "highest_classes": high_j,
                                              "pessimistic": low_j,
                                              "optimistic": high_j})

            # --------- PROMETHEE ---------
            if key in ("promethee_i","promethee_ii","promethee_iii",
                       "promethee_iv","promethee_v","promethee_vi","promethee_gaia"):
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                S = list(np.asarray(extra_inputs["S"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                F = list(extra_inputs.get("F", ['t1']*X.shape[1]))
                W_list = list(np.asarray(w, dtype=float).ravel())
                if key == "promethee_i":
                    cp = A.promethee_i(X, W_list, Q, S, P, F, graph=False)
                    # cp_matrix is a string preference matrix; derive a Borda
                    # score from "P+" (preferred over) entries
                    cp_arr = np.asarray(cp)
                    n = X.shape[0]
                    score = np.zeros(n)
                    for i in range(n):
                        for j in range(n):
                            v = str(cp_arr[i, j]) if cp_arr.ndim == 2 else ""
                            if v in ("P+", "I"):
                                score[i] += 1 if v == "P+" else 0.5
                    return _build(_ok(key, score.tolist(), True, buf.getvalue(),
                                      extras={"preference_matrix":
                                              cp_arr.tolist() if hasattr(cp_arr,'tolist') else cp,
                                              "promethee_partial": True,
                                              "promethee_partial_per_alt": True,
                                              "hide_top": True,
                                              "hide_full_ranking": True,
                                              "hide_scores": True}))
                if key == "promethee_ii":
                    flow = A.promethee_ii(X, W_list, Q, S, P, F, sort=False, topn=0,
                                           graph=False, verbose=False)
                    return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                      True, buf.getvalue()))
                if key == "promethee_iii":
                    lmbd = float(params.get("lmbd", 0.15))
                    cp = A.promethee_iii(X, W_list, Q, S, P, F, lmbd=lmbd,
                                          graph=False)
                    cp_arr = np.asarray(cp)
                    n = X.shape[0]
                    score = np.zeros(n)
                    for i in range(n):
                        for j in range(n):
                            v = str(cp_arr[i, j]) if cp_arr.ndim == 2 else ""
                            if v in ("P+", "I"):
                                score[i] += 1 if v == "P+" else 0.5
                    return _build(_ok(key, score.tolist(), True, buf.getvalue(),
                                      extras={"preference_matrix":
                                              cp_arr.tolist() if hasattr(cp_arr,'tolist') else cp,
                                              "promethee_partial_per_alt": True,
                                              "hide_top": True,
                                              "hide_scores": True}))
                if key == "promethee_iv":
                    steps = float(params.get("steps", 0.001))
                    flow = A.promethee_iv(X, W_list, Q, S, P, F, sort=False,
                                           steps=steps, topn=0,
                                           graph=False, verbose=False)
                    return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                      True, buf.getvalue()))
                if key == "promethee_v":
                    max_selected = int(params.get("max_selected", min(2, X.shape[0])))
                    iterations = int(params.get("iterations", 500))
                    flow = A.promethee_v(X, W_list, Q, S, P, F,
                                         sort=False,
                                         criteria=max_selected,
                                         cost=[],
                                         budget=0,
                                         forbidden=[],
                                         iterations=iterations,
                                         verbose=False)
                    flow_arr = np.asarray(flow, dtype=float)
                    scores = _flow_to_scores(flow_arr[:, :2], X.shape[0])
                    selected_mask = flow_arr[:, 2].astype(int).tolist() if flow_arr.ndim == 2 and flow_arr.shape[1] >= 3 else [0] * X.shape[0]
                    selected = [f"A{i+1}" for i, flag in enumerate(selected_mask) if flag == 1]
                    return _build(_ok(key, scores, True, buf.getvalue(),
                                      extras={"selected_mask": selected_mask,
                                              "selected_alternatives": selected,
                                              "max_selected": max_selected}))
                if key == "promethee_vi":
                    W_lower = np.asarray(extra_inputs.get("W_lower", []), dtype=float).ravel()
                    W_upper = np.asarray(extra_inputs.get("W_upper", []), dtype=float).ravel()
                    if W_lower.size != X.shape[1] or W_upper.size != X.shape[1]:
                        raise ValueError("PROMETHEE VI requires lower and upper weight vectors with one value per criterion.")
                    iterations = int(params.get("iterations", 1000))
                    flow_lower, flow_mid, flow_upper = A.promethee_vi(
                        X, W_lower, W_upper, Q, S, P, F,
                        sort=False, topn=0, iterations=iterations,
                        graph=False, verbose=False
                    )
                    return _build(_ok(key, _flow_to_scores(flow_mid, X.shape[0]),
                                      True, buf.getvalue(),
                                      extras={"lower_scores": _flow_to_scores(flow_lower, X.shape[0]),
                                              "favorable_scores": _flow_to_scores(flow_mid, X.shape[0]),
                                              "upper_scores": _flow_to_scores(flow_upper, X.shape[0]),
                                              "weight_lower": W_lower.tolist(),
                                              "weight_upper": W_upper.tolist(),
                                              "minus_scores": _flow_to_scores(flow_lower, X.shape[0]),
                                              "mid_scores": _flow_to_scores(flow_mid, X.shape[0]),
                                              "plus_scores": _flow_to_scores(flow_upper, X.shape[0]),
                                              "hide_top": True,
                                              "hide_full_ranking": True}))
                if key == "promethee_gaia":
                    flow = A.promethee_ii(X, W_list, Q, S, P, F, sort=False, topn=0,
                                           graph=False, verbose=False)
                    extras = {"note":"GAIA plot generated by promethee_gaia.",
                              "choice_only": True}
                    try:
                        A.promethee_gaia(X, W_list, Q, S, P, F)
                        extras['gaia_plot'] = _fig_to_base64(plt.gcf())
                    except Exception:
                        pass
                    return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                      True, buf.getvalue(), extras=extras))

            if key == "ec_promethee":
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                S = list(np.asarray(extra_inputs["S"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                F = list(extra_inputs.get("F", ['t1']*X.shape[1]))
                iterations = int(params.get("iterations", 50))
                wnorm_matrix, ranks_matrix, p2_matrix, sol = A.ec_promethee(
                    X, types, custom_sets=None, Q=Q, S=S, P=P, F=F,
                    iterations=iterations, verbose=False
                )
                sol_arr = np.asarray(sol, dtype=float).ravel()
                scores = (len(sol_arr) + 1 - sol_arr).tolist()
                extras = {"final_ranks": sol_arr.tolist(),
                          "weight_samples": _to_jsonable(wnorm_matrix),
                          "rank_samples": _to_jsonable(ranks_matrix),
                          "net_flow_samples": _to_jsonable(p2_matrix)}
                # Note: front-end now builds an HTML heatmap from rank_samples;
                # we no longer ship the matplotlib PNG.
                return _build(_ok(key, scores, True, buf.getvalue(), extras=extras))

            if key == "flowsort":
                profiles = [list(map(float, row)) for row in extra_inputs["profiles"]]
                Q = list(np.asarray(extra_inputs["Q"], dtype=float).ravel())
                S = list(np.asarray(extra_inputs["S"], dtype=float).ravel())
                P = list(np.asarray(extra_inputs["P"], dtype=float).ravel())
                F = list(extra_inputs.get("F", ['t1']*X.shape[1]))
                W_list = list(np.asarray(w, dtype=float).ravel())
                combos = {}
                for mode in ("central", "limiting"):
                    for rule in ("negative", "positive", "net"):
                        vec = A.flowsort_method(X, profiles, W_list, Q, S, P, F,
                                                mode=mode, rule=rule, verbose=False)
                        combos[f"{mode}_{rule}"] = _to_jsonable(np.asarray(vec).ravel().tolist())
                class_per_alt = combos.get("central_net", [None]*X.shape[0])
                return _build_sorting(key, class_per_alt, buf.getvalue(),
                                      extras={"classification": class_per_alt,
                                              "classification_by_mode_rule": combos})

            if key == "cpp_tri":
                rule = params.get("rule", "central")
                dist = params.get("dist", "normal")
                raw_profiles = extra_inputs["profiles"]
                # The algorithm's profiles_to_quantiles handles two shapes natively:
                #   * dict { "Profile 1": [[...], [...]], ... }  (averages sub-rows)
                #   * 2-D array [[...], [...], ...]
                # Don't np.asarray() the dict — that would crash with
                # "float() argument must be ... not 'dict'".
                if isinstance(raw_profiles, dict):
                    profiles = raw_profiles
                    num_cat = len(raw_profiles)
                else:
                    profiles = np.asarray(raw_profiles, dtype=float)
                    num_cat = profiles.shape[0]
                result = A.cpp_tri_method(X, weights=w, profiles=profiles,
                                           num_cat=num_cat,
                                           rule=rule, dist=dist, verbose=False)
                cls = _to_jsonable(result)
                n = X.shape[0]
                class_per_alt = [None]*n
                if isinstance(cls, list):
                    for i, c in enumerate(cls[:n]):
                        class_per_alt[i] = c
                return _build_sorting(key, class_per_alt, buf.getvalue(),
                                      extras={"classification": cls})

            if key in ("utadis_i", "utadis_ii", "utadis_iii"):
                y = np.asarray(extra_inputs.get("class_labels", []), dtype=int).ravel()
                if y.size != X.shape[0]:
                    raise ValueError("UTADIS requires one observed class label per alternative.")
                ai = int(params.get("ai", 5))
                delta = float(params.get("delta", 1e-4))
                s = float(params.get("s", 1e-4))
                if key == "utadis_i":
                    result = A.utadis_i_method(X, y, criterion_type=types,
                                               ai=ai, delta=delta, s=s,
                                               train_size=1.0, verbose=False)
                elif key == "utadis_ii":
                    result = A.utadis_ii_method(X, y, criterion_type=types,
                                                ai=ai, delta=delta, s=s,
                                                train_size=1.0, verbose=False)
                else:
                    big_M = float(params.get("big_M", 1e4))
                    lambda_margin = float(params.get("lambda_margin", 1e-2))
                    result = A.utadis_iii_method(X, y, criterion_type=types,
                                                 ai=ai, train_size=1.0,
                                                 delta=delta, s=s,
                                                 big_M=big_M,
                                                 lambda_margin=lambda_margin,
                                                 verbose=False)
                class_per_alt = _to_jsonable(result.get("y_pred_train", []) if isinstance(result, dict) else [])
                extras = _to_jsonable(result if isinstance(result, dict) else {"result": result})
                extras["observed_classes"] = y.tolist()
                return _build_sorting(key, class_per_alt, buf.getvalue(), extras=extras)

            # --------- Specialised ---------
            if key == "odo_ovo":
                critical = int(extra_inputs.get("critical_criterion", 0))
                rank_by = params.get("rank_by", "l2")
                weighted = str(params.get("weighted","false")).lower()=="true"
                flow = A.odo_ovo_method(X, w, types, critical_criterion=critical,
                                         weighted=weighted, rank_by=rank_by,
                                         graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, X.shape[0]),
                                  True, buf.getvalue()))

            if key == "opa":
                er = extra_inputs.get("experts_rank", [1])
                erc = extra_inputs.get("experts_rank_criteria", [])
                era = extra_inputs.get("experts_rank_alternatives", [])
                w_e, w_c, w_a = A.opa_method(experts_rank=er,
                                              experts_rank_criteria=erc,
                                              experts_rank_alternatives=era,
                                              graph=False, verbose=False)
                w_a_arr = np.asarray(w_a, dtype=float).ravel()
                return _build(_ok(key, w_a_arr.tolist(), True, buf.getvalue(),
                                  extras={"opa_view": True,
                                          "expert_weights": _to_jsonable(w_e),
                                          "criteria_weights": _to_jsonable(w_c),
                                          "alternative_weights": _to_jsonable(w_a),
                                          "hide_top": True,
                                          "hide_full_ranking": True}))

            # --------- Fuzzy ranking ---------
            # Fuzzy ranking methods expect weights as a 2-D list:
            # [[(l,m,u), (l,m,u), ...]]  — i.e. one row of fuzzy weights.
            def _wrap_fw(fw_in):
                # Fallback: pull from extra_inputs or weights param
                if fw_in is None:
                    fw_in = extra_inputs.get("fuzzy_weights")
                if fw_in is None and weights is not None:
                    fw_in = weights
                if fw_in is None:
                    return None
                # Convert to plain Python lists
                if hasattr(fw_in, 'tolist'):
                    fw_in = fw_in.tolist()
                # If already wrapped (list of list of triples)
                if (isinstance(fw_in, list) and fw_in and
                    isinstance(fw_in[0], (list, tuple)) and fw_in[0] and
                    isinstance(fw_in[0][0], (list, tuple))):
                    return [[list(c) for c in row] for row in fw_in]
                # If it's a list of triples — wrap once
                if (isinstance(fw_in, list) and fw_in and
                    isinstance(fw_in[0], (list, tuple))):
                    return [[list(c) for c in fw_in]]
                # Otherwise wrap once
                return [list(fw_in)]

            if key == "fuzzy_topsis":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                flow = A.fuzzy_topsis_method(M, fw, types,
                                              graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, len(M)),
                                  True, buf.getvalue()))

            if key == "fuzzy_vikor":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                v = float(params.get("strategy_coefficient", 0.5))
                M = extra_inputs["matrix"]
                fs, fr, fq, sol = A.fuzzy_vikor_method(M, fw, types,
                                                       strategy_coefficient=v,
                                                       graph=False, verbose=False)
                n = len(M)
                q = _flow_to_scores(fq, n)
                return _build(_ok(key, q, False, buf.getvalue(),
                                  extras={"S": _flow_to_scores(fs, n),
                                          "R": _flow_to_scores(fr, n),
                                          "Q": q,
                                          "solution_label": str(sol)}))

            if key == "fuzzy_aras":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                flow = A.fuzzy_aras_method(M, fw, types,
                                            graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, len(M)),
                                  True, buf.getvalue()))

            if key == "fuzzy_copras":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                flow = A.fuzzy_copras_method(M, fw, types,
                                              graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, len(M)),
                                  True, buf.getvalue()))

            if key == "fuzzy_edas":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                flow = A.fuzzy_edas_method(M, types, fw,
                                            graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, len(M)),
                                  True, buf.getvalue()))

            if key == "fuzzy_moora":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                flow = A.fuzzy_moora_method(M, fw, types,
                                             graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, len(M)),
                                  True, buf.getvalue()))

            if key == "fuzzy_ocra":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                flow = A.fuzzy_ocra_method(M, fw, types,
                                            graph=False, verbose=False)
                return _build(_ok(key, _flow_to_scores(flow, len(M)),
                                  True, buf.getvalue()))

            if key == "fuzzy_waspas":
                fw = _wrap_fw(extra_inputs.get("fuzzy_weights"))
                M = extra_inputs["matrix"]
                f_wsm, f_wpm, f_waspas = A.fuzzy_waspas_method(M, types, fw, graph=False)
                return _build(_ok(key, list(map(float, f_waspas)),
                                  True, buf.getvalue(),
                                  extras={"f_wsm": list(map(float, f_wsm)),
                                          "f_wpm": list(map(float, f_wpm)),
                                          "f_waspas": list(map(float, f_waspas))}))

            return {"ok": False, "method": key,
                    "error": f"Method '{key}' not yet implemented in dispatcher."}

    except Exception as exc:
        return _err(key, exc)


def _build(payload):
    return _to_jsonable(payload)


def _build_sorting(key, class_per_alt, stdout, extras=None):
    """Result for sorting/Tri methods."""
    return _to_jsonable({
        "ok": True,
        "method": key,
        "stdout": stdout,
        "result_kind": "sorting",
        "class_per_alternative": class_per_alt,
        "extras": extras or {},
    })


def _weights_result(key, weights_arr, stdout, extras=None):
    """Result for weight-producing methods."""
    w = np.asarray(weights_arr, dtype=float).ravel()
    if w.sum() > 0:
        w_norm = w / w.sum()
    else:
        w_norm = w
    return _to_jsonable({
        "ok": True,
        "method": key,
        "stdout": stdout,
        "result_kind": "weights",
        "weights": w_norm.tolist(),
        "weights_raw": w.tolist(),
        "extras": extras or {},
    })


def _accepts(fn, kw_name):
    try:
        import inspect
        return kw_name in inspect.signature(fn).parameters
    except Exception:
        return False
