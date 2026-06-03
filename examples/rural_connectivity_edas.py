"""
Rural Connectivity Technology Selection Using EDAS
==================================================
Decision problem
----------------
A regional authority must select the most suitable last-mile connectivity
technology for underserved rural and remote communities. Five candidates
are evaluated against five criteria:

    C1 Coverage area   (km²) – maximise
    C2 Deployment cost (USD/km) – minimise
    C3 Latency         (ms) – minimise
    C4 Reliability     (%)  – maximise
    C5 Scalability     (1-10 composite score) – maximise

Reference: Das, D., Nagalla, S., Paladugu, N. (2024). "Enhancing Connectivity
and Communication in Underserved Rural and Remote Areas – An EDAS-Based
Investment Assessment." Journal on Electronic and Automation Engineering,
3(4), 22–29.
"""

import numpy as np
from pyDecision.algorithm import edas_method

technologies = [
    'Satellite Internet',
    'LTE Expansion',
    'Microwave Relay',
    'Fiber Extension',
    'TV White Space',
]

# Rows = alternatives, columns = criteria
# [coverage(km²), deploy_cost(USD/km), latency(ms), reliability(%), scalability]
dataset = np.array([
    [5000,  3500,  620,  95.0, 6.5],   # Satellite Internet
    [1200,  8000,   45,  97.5, 8.0],   # LTE Expansion
    [ 800,  5500,   15,  96.0, 7.5],   # Microwave Relay
    [ 200, 18000,    5,  99.2, 9.5],   # Fiber Extension
    [2500,  1200,  110,  93.5, 7.0],   # TV White Space
])

weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

# 'max' = higher is better, 'min' = lower is better
criterion_type = ['max', 'min', 'min', 'max', 'max']

print('Rural Connectivity Technology Selection – EDAS')
print('=' * 48)
print(f'{"Technology":<20} {"EDAS Score":>10} {"Rank":>6}')
print('-' * 38)

scores = edas_method(dataset, criterion_type, weights, graph=False, verbose=False)
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

for rank, (idx, score) in enumerate(ranked, start=1):
    print(f'{technologies[idx]:<20} {score:>10.4f} {rank:>6}')

print()
best = ranked[0]
print(f'Recommended: {technologies[best[0]]} (score = {best[1]:.4f})')
