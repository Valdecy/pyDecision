"""
Bio-based Polymer Composite Formulation Ranking Using GRA
=========================================================
Decision problem
----------------
A materials engineer must select the optimal bio-based polymer composite
formulation from six candidates. Each formulation varies in fibre type,
matrix composition, and processing parameters. Six mechanical and
environmental criteria guide the selection:

    C1 Tensile strength    (MPa) – maximise
    C2 Flexural modulus    (GPa) – maximise
    C3 Impact resistance   (kJ/m²) – maximise
    C4 Water absorption    (%)  – minimise
    C5 Biodegradability    (%)  – maximise
    C6 Processing cost     (USD/kg) – minimise

Reference: Nagalla, S., Paladugu, N., Das, D. (2024). "Optimization of
Bio-based Polymer Composites Using Grey Relational Analysis (GRA) Method."
REST Journal on Advances in Mechanical Engineering, 3(3), 19–31.
"""

import numpy as np
from pyDecision.algorithm import gra_method

formulations = [
    'Flax/PLA-30%',
    'Hemp/PBS-25%',
    'Jute/PBAT-35%',
    'Sisal/PLA-20%',
    'Kenaf/PHB-30%',
    'Bamboo/PBS-40%',
]

# Rows = alternatives, columns = criteria
# [tensile(MPa), flexural(GPa), impact(kJ/m²), water_abs(%), biodeg(%), cost(USD/kg)]
dataset = np.array([
    [42.5, 3.8, 18.2, 3.1, 82, 4.20],   # Flax/PLA-30%
    [38.0, 3.2, 22.5, 2.8, 78, 3.85],   # Hemp/PBS-25%
    [35.5, 2.9, 20.1, 4.5, 88, 3.50],   # Jute/PBAT-35%
    [44.0, 4.1, 15.8, 2.2, 75, 4.60],   # Sisal/PLA-20%
    [40.2, 3.5, 24.0, 3.8, 91, 5.10],   # Kenaf/PHB-30%
    [36.8, 3.1, 26.5, 5.2, 85, 3.20],   # Bamboo/PBS-40%
])

weights = np.array([0.20, 0.15, 0.20, 0.15, 0.20, 0.10])

# 'max' = higher is better, 'min' = lower is better
criterion_type = ['max', 'max', 'max', 'min', 'max', 'min']

print('Bio-based Polymer Composite Selection – GRA')
print('=' * 46)
print(f'{"Formulation":<18} {"GRA Grade":>10} {"Rank":>6}')
print('-' * 36)

grades = gra_method(dataset, criterion_type, weights, graph=False, verbose=False)
ranked = sorted(enumerate(grades), key=lambda x: x[1], reverse=True)

for rank, (idx, grade) in enumerate(ranked, start=1):
    print(f'{formulations[idx]:<18} {grade:>10.4f} {rank:>6}')

print()
best = ranked[0]
print(f'Recommended: {formulations[best[0]]} (GRA grade = {best[1]:.4f})')
