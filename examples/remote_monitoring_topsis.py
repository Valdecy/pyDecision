"""
Remote Monitoring Platform Selection Using TOPSIS
=================================================
Decision problem
----------------
A facilities team must choose an IoT-based remote monitoring platform
from five candidates. Five criteria drive the evaluation:

    C1 Sensor accuracy       (%) – maximise
    C2 Network latency       (ms) – minimise  (lower is better)
    C3 Power consumption     (W)  – minimise
    C4 Scalability score     (1-10) – maximise
    C5 Cost efficiency score (1-10) – maximise

Reference: Das, D., Paladugu, N., Nagalla, S. (2024). "Automation and Remote
Monitoring by using TOPSIS METHOD." Journal on Electronic and Automation
Engineering, 3(3), 16–24.
"""

import numpy as np
from pyDecision.algorithm import topsis_method

platforms = ['Platform-A', 'Platform-B', 'Platform-C', 'Platform-D', 'Platform-E']

# Rows = alternatives, columns = criteria
# [accuracy(%), latency(ms), power(W), scalability(1-10), cost_efficiency(1-10)]
dataset = np.array([
    [96.5, 18, 4.2, 8.5, 7.8],   # Platform-A
    [94.0, 12, 3.8, 9.0, 8.5],   # Platform-B
    [98.2, 25, 5.1, 7.5, 6.9],   # Platform-C
    [92.5, 10, 3.2, 9.5, 9.1],   # Platform-D
    [95.8, 20, 4.5, 8.0, 7.2],   # Platform-E
])

weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

# 'max' = higher is better, 'min' = lower is better
criterion_type = ['max', 'min', 'min', 'max', 'max']

print('Remote Monitoring Platform Selection – TOPSIS')
print('=' * 46)
print(f'{"Platform":<14} {"TOPSIS Score":>12} {"Rank":>6}')
print('-' * 34)

scores = topsis_method(dataset, weights, criterion_type, graph=False, verbose=False)
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

for rank, (idx, score) in enumerate(ranked, start=1):
    print(f'{platforms[idx]:<14} {score:>12.4f} {rank:>6}')

print()
best = ranked[0]
print(f'Recommended: {platforms[best[0]]} (score = {best[1]:.4f})')
