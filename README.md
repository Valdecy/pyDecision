# pyDecision

## Introduction

A python library with the following MCDA methods: **AHP** (Analytic Hierarchy Process); **Fuzzy AHP**; **ARAS** (Additive Ratio ASsessment); **Fuzzy ARAS**; **Borda**; **BWM** (Best-Worst Method); **CILOS** (Criterion Impact LOSs); **CoCoSo** (COmbined COmpromise SOlution); **CODAS** (Combinative Distance-based Assessment); **Copeland**; **COPRAS** (Complex PRoportional Assessment); **Fuzzy COPRAS**; **CRADIS** (Compromise Ranking of Alternatives from Distance to Ideal Solution); **CRITIC** (CRiteria Importance Through Intercriteria Correlation); **DEMATEL** (DEcision MAking Trial and Evaluation Laboratory); **Fuzzy DEMATEL**; **EDAS** (Evaluation based on Distance from Average Solution); **Fuzzy EDAS**; **Entropy**; **ELECTRE** (I, I_s, I_v, II, III, IV, Tri-B); **GRA** (Grey Relational Analysis); **IDOCRIW** (Integrated Determination of Objective CRIteria Weights); **MABAC** (Multi-Attributive Border Approximation area Comparison); **MACBETH** (Measuring Attractiveness by a Categorical Based Evaluation TecHnique); **MAIRCA** (Multi-Attributive Ideal-Real Comparative Analysis); **MARCOS** (Measurement of Alternatives and Ranking according to COmpromise Solution); **MAUT** (Multi-attribute Utility Theory); **MEREC** (MEthod based on the Removal Effects of Criteria); **MOORA** (Multi-Objective Optimization on the basis of Ratio Analysis); **Fuzzy MOORA**; **MOOSRA** (Multi-Objective Optimisation on the Basis of Simple Ratio Analysis);  **MULTIMOORA** (Multi-Objective Optimization on the basis of Ratio Analisys Multiplicative Form); **OCRA** (Operational Competitiveness RAting); **Fuzzy OCRA** ; **ORESTE** (Organisation Rangement Et SynThèsE de données relationnelles); **PROMETHEE** (I, II, III, IV, V, VI, Gaia); **SAW** (Simple Additive Weighting); **SMART** (Simple Multi-Attribute Rating Technique); **TODIM** (TOmada de Decisao Interativa e Multicriterio - Interactive and Multicriteria Decision Making); **PIV** (Proximity Indexed Value); **PSI** (Preference Selection Index); **Regime**; **ROV** (Range Of Value); **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution); **Fuzzy TOPSIS**; **VIKOR** (VIseKriterijumska Optimizacija I Kompromisno Resenje); **Fuzzy VIKOR**; **WINGS** (Weighted Influence Non-linear Gauge System); **WSM** (Weighted Sum Model); **WPM** (Weighted Product Model); **WASPAS** (Weighted Aggregates Sum Product Assessment).

pyDecision offers an array of features including the **comparison of ranking alternatives** and **comparison of criterion weights** from various methods. The library is also fully integrated with **chatGPT**, elevating result interpretation through AI. Additionally, pyDecision provides the flexibility to import results from custom methods or those not yet implemented in the library for swift comparison.

## Usage

1. Install
```bash
pip install pyDecision
```

2. Import

```py3

# Import AHP
from pyDecision.algorithm import ahp_method

# Parameters
weight_derivation = 'geometric' # 'mean' or 'geometric'

# Dataset
dataset = np.array([
  #g1     g2     g3     g4     g5     g6     g7                  
  [1  ,   1/3,   1/5,   1  ,   1/4,   1/2,   3  ],   #g1
  [3  ,   1  ,   1/2,   2  ,   1/3,   3  ,   3  ],   #g2
  [5  ,   2  ,   1  ,   4  ,   5  ,   6  ,   5  ],   #g3
  [1  ,   1/2,   1/4,   1  ,   1/4,   1  ,   2  ],   #g4
  [4  ,   3  ,   1/5,   4  ,   1  ,   3  ,   2  ],   #g5
  [2  ,   1/3,   1/6,   1  ,   1/3,   1  ,   1/3],   #g6
  [1/3,   1/3,   1/5,   1/2,   1/2,   3  ,   1  ]    #g7
])

# Call AHP Function
weights, rc = ahp_method(dataset, wd = weight_derivation)

# Weigths
for i in range(0, weights.shape[0]):
  print('w(g'+str(i+1)+'): ', round(weights[i], 3))
  
# Consistency Ratio
print('RC: ' + str(round(rc, 2)))
if (rc > 0.10):
  print('The solution is inconsistent, the pairwise comparisons must be reviewed')
else:
  print('The solution is consistent')

```

3. Try it in **Colab**:

- AHP ([ Colab Demo ](https://colab.research.google.com/drive/1qwFQs5xkTZ8K-Ul_wWcCtPjLH0QooU9g?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1016/0377-2217(90)90057-I))
- Fuzzy AHP ([ Colab Demo ](https://colab.research.google.com/drive/1RtEMOLGL5wtmheMRZv8emcO5wbjYVBCo?usp=sharing)) ( [ Paper ](https://arxiv.org/ftp/arxiv/papers/1311/1311.2886.pdf))
- ARAS ([ Colab Demo ](https://colab.research.google.com/drive/1rwQgXjvC3E6pRhOs7CkcCV8Vw2bXEPLy?usp=sharing)) ( [ Paper ](https://www.tandfonline.com/doi/pdf/10.3846/tede.2010.10#:~:text=According%20to%20the%20ARAS%20method,criteria%20considered%20in%20a%20project.))
- Fuzzy ARAS ([ Colab Demo ](https://colab.research.google.com/drive/1kZDkEWsw0d0nFhDQQk8azZXRod7RnfZr?usp=sharing)) ( [ Paper ](https://doi.org/10.1155/2021/8545379))
- Borda ([ Colab Demo ](https://colab.research.google.com/drive/1t5RVtG7_yXK-nPxM0MVd4U01qfTQYW4k?usp=sharing)) ( [ Paper ](http://gerardgreco.free.fr/IMG/pdf/MA_c_moire-Borda-1781.pdf))
- BWM ([ Colab Demo ](https://colab.research.google.com/drive/1XkacTmtSBvZmx_5K9cfz8t1Ao5j-D-bZ?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.omega.2014.11.009))
- CILOS ([ Colab Demo ](https://colab.research.google.com/drive/1RnSqO_VEPyvXAMHdneloYvA0TzPx55kw?usp=sharing)) ( [ Paper ](https://www.worldscientific.com/doi/10.1142/S0219622016500036))
- CoCoSo ([ Colab Demo ](https://colab.research.google.com/drive/1U8a3NZzQaxDkJdUT3uKIeeoqFtT_3Mnx?usp=sharing)) ( [ Paper ](https://www.emerald.com/insight/content/doi/10.1108/MD-05-2017-0458/full/html))
- CODAS ([ Colab Demo ](https://colab.research.google.com/drive/1hm7__urqFeBHM6nVQJcBzGPF72DFuoLr?usp=sharing)) ( [ Paper ](https://EconPapers.repec.org/RePEc:cys:ecocyb:v:50:y:2016:i:3:p:25-44))
- Copeland ([ Colab Demo ](https://colab.research.google.com/drive/1ObP3AkQAzoCxT6et5Qkyk1trlER7mcdH?usp=sharing)) ( [ Paper ](https://link.springer.com/article/10.1007/bf01212012))
- COPRAS ([ Colab Demo ](https://colab.research.google.com/drive/1TZJtSjXqwYEwuL7-wfLcPQ8ZBtDq3lth?usp=sharing)) ( [ Paper ](https://doi.org/10.3846/20294913.2012.762953))
- Fuzzy COPRAS ([ Colab Demo ](https://colab.research.google.com/drive/1AIGgxBkmcA6YHKx06VeYcGf2EV8dPffW?usp=sharing)) ( [ Paper ](https://doi.org/10.1007/s00500-021-05762-w))
- CRADIS ([ Colab Demo ](https://colab.research.google.com/drive/1p7AQmPIOsZFxaypqMsiRIWW8mIvDtoLi?usp=sharing)) ( [ Paper ](https://doi.org/10.1007%2Fs10668-021-01902-2))
- CRITIC ([ Colab Demo ](https://colab.research.google.com/drive/1D5SaBHa1-Eo_KYSXHkFjsHYu29M21l_F?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/0305-0548(94)00059-H))
- DEMATEL ([ Colab Demo ](https://colab.research.google.com/drive/1T04qEft9uwTyQx--gADN6V_vUrT21Xo6?usp=sharing)) ( [ Paper ](https://doi.org/10.1155/2018/3696457))
- Fuzzy DEMATEL ([ Colab Demo ](https://colab.research.google.com/drive/15e9dMDROr3cxjbWRXg3_t4TScuQtQDpR?usp=sharing)) ( [ Paper ](https://www.mdpi.com/2071-1050/9/11/2083))
- EDAS ([ Colab Demo ](https://colab.research.google.com/drive/1xsMdwH-IH-zvOW-1kv6ztQnKGt7p5JnY?usp=sharing)) ( [ Paper ](https://doi.org/10.15388/Informatica.2015.57))
- Fuzzy EDAS ([ Colab Demo ](https://colab.research.google.com/drive/1kw2LwztNAU9Asjj6BvBmvk11wvk8R3V6?usp=sharing)) ( [ Paper ](https://doi.org/10.1007/978-981-32-9072-3_63))
- Entropy ([ Colab Demo ](https://colab.research.google.com/drive/1LOCef2KFxoV2qUEQRi4DqfzrgnMgtwT9?usp=sharing)) ( [ Paper ](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf))
- ELECTRE I     ([ Colab Demo ](https://colab.research.google.com/drive/1KFqRPBRyv-fxiu2B1y7VNkP5pCCbILF1?usp=sharing)) ( [ Paper ](https://github.com/Valdecy/Datasets/blob/master/MCDA/E01.pdf))
- ELECTRE I_s   ([ Colab Demo ](https://colab.research.google.com/drive/1ngxsQPh2QULjd1_AifFofbukq5zIOePd?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1007/978-1-4757-5057-7_3))
- ELECTRE I_v   ([ Colab Demo ](https://colab.research.google.com/drive/1moonq95gqXqmbRe2KvgqbN2IfowJ12C-?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1007/978-1-4757-5057-7_3))
- ELECTRE II    ([ Colab Demo ](https://colab.research.google.com/drive/1UeAjICH6_tjVr3O9H-fC65HHYMVZgTKc?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1007/978-1-4757-5057-7_3))
- ELECTRE III   ([ Colab Demo ](https://colab.research.google.com/drive/1smeD5ZoPgBnAAUyooAXSrkxHgqZPmUC9?usp=sharing)) ( [ Paper ](https://github.com/Valdecy/Datasets/raw/master/MCDA/E03.pdf))
- ELECTRE IV    ([ Colab Demo ](https://colab.research.google.com/drive/178x062yC-Es6lstEiFaFprbMsTJZwnC-?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1007/978-1-4757-5057-7_3))
- ELECTRE Tri-B ([ Colab Demo ](https://colab.research.google.com/drive/1hu0fJcxdBAiEDrVngmKQfpINpjTF-osE?usp=sharing)) ( [ Paper ](https://drive.google.com/file/d/1oWOI_sX3EEYdRbavoBTT7vUmPII1yPgE/view?usp=sharing))
- GRA ([ Colab Demo ](https://colab.research.google.com/drive/1aMMI0Cuo5kpzTDefqEwJhf0wWpBOP_JL?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/S0167-6911(82)80025-X))
- IDOCRIW ([ Colab Demo ](https://colab.research.google.com/drive/1zt8uPFZGcHaSnpiT7tDnrDjvs0pK_7vS?usp=sharing)) ( [ Paper ](https://link.springer.com/chapter/10.1007/978-3-030-15009-9_19))
- MABAC ([ Colab Demo ](https://colab.research.google.com/drive/1BMqO-HnBXdcOZfZoULpx1H4MLPoUGucJ?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.eswa.2014.11.057))
- MACBETH ([ Colab Demo ](https://colab.research.google.com/drive/1GqM9uPgbaWCGyj4l-XjkoifY2JJoVyf2?usp=sharing)) ( [ Paper ](https://link.springer.com/chapter/10.1007/978-3-030-15009-9_16))
- MAIRCA ([ Colab Demo ](https://colab.research.google.com/drive/1gfqgrBAFGVygwm1j3lTjfy5wTsLgT_j5?usp=sharing)) ( [ Paper ](https://www.tandfonline.com/doi/full/10.1080/1331677X.2018.1506706))
- MARCOS ([ Colab Demo ](https://colab.research.google.com/drive/13MI2Qrakm5VzHN3r5O2RqggCzQwRxCs-?usp=sharing)) ( [ Paper ](https://www.sciencedirect.com/science/article/abs/pii/S0360835219307004))
- MAUT ([ Colab Demo ](https://colab.research.google.com/drive/1qm3ARgQm68GUK2irGiCB-B49vnVHazB7?usp=sharing)) ( [ Paper ](https://apps.dtic.mil/sti/pdfs/AD0770576.pdf))
- MEREC ([ Colab Demo ](https://colab.research.google.com/drive/1XE3AIzS84w-gw_1MEtV7xvkU1Gj_tRPd?usp=sharing)) ( [ Paper ](https://www.mdpi.com/2073-8994/13/4/525))
- MOORA ([ Colab Demo ](https://colab.research.google.com/drive/1FpKl0QAdwGgCVvLYsRHvMWhz7yOp17B5?usp=sharing)) ( [ Paper ](https://www.researchgate.net/publication/228345226_The_MOORA_method_and_its_application_to_privatization_in_a_transition_economy))
- Fuzzy MOORA ([ Colab Demo ](https://colab.research.google.com/drive/1ydHzGeA8WBVY5Gyu8K7Oq6kofQ5XbK3P?usp=sharing)) ( [ Paper ](https://pdfs.semanticscholar.org/6d33/ca3f14c9ed44d23742fd4e9cf94cebcaf148.pdf))
- MOOSRA ([ Colab Demo ](https://colab.research.google.com/drive/1KYyA4f3OsipPA5e63Ja4A0OGmHvNY6dj?usp=sharing)) ( [ Paper ](https://ijret.org/volumes/2014v03/i15/IJRET20140315105.pdf))
- MULTIMOORA ([ Colab Demo ](https://colab.research.google.com/drive/1JAT8qqHPNoFfMV6a-CzF6BgRwtcUF3-e?usp=sharing)) ( [ Paper ](https://journals.vilniustech.lt/index.php/TEDE/article/view/5832/5078))
- OCRA ([ Colab Demo ](https://colab.research.google.com/drive/1yQ41lOdjhiANtD1SOXoxA7gVim7A4X4P?usp=sharing)) ( [ Paper ](https://www.researchgate.net/publication/272362515_Selection_of_non-conventional_machining_processes_using_the_OCRA_method))
- Fuzzy OCRA ([ Colab Demo ](https://colab.research.google.com/drive/1SniY4RLsR6jR9SnI3AR9k0wGlBWH6Pm8?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.5755/j01.ee.30.5.20546))
- ORESTE ([ Colab Demo ](https://colab.research.google.com/drive/1USVCt6KJHJK9NXaknY8wTA4L0d4r2kWw?usp=sharing)) ( [ Paper ](https://link.springer.com/chapter/10.1007/978-3-030-15009-9_3))
- PROMETHEE I    ([ Colab Demo ](https://colab.research.google.com/drive/1WsagC7-Y_5X-Xl90pMz8YwUkKfxf2vol?usp=sharing)) ( [ Paper ](https://www.cin.ufpe.br/~if703/aulas/promethee.pdf))
- PROMETHEE II   ([ Colab Demo ](https://colab.research.google.com/drive/143TUtTBy9y6gW0kMVAfhANBhuw1bKvBB?usp=sharing)) ( [ Paper ](https://www.cin.ufpe.br/~if703/aulas/promethee.pdf))
- PROMETHEE III  ([ Colab Demo ](https://colab.research.google.com/drive/11DBaEBBT8B-B3poXubvZ41HELOHok0Rz?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1007/978-3-030-15009-9_5
))
- PROMETHEE IV   ([ Colab Demo ](https://colab.research.google.com/drive/1X2evE6pIf4F7qiKjt1fSU2PqT-NaA5sJ?usp=sharing)) ( [ Paper ](http://dx.doi.org/10.1007/978-3-319-11949-6_14))
- PROMETHEE V    ([ Colab Demo ](https://colab.research.google.com/drive/1IaZCCtq5m8vBBxrBLMCp6xB5U2j8ZNRc?usp=sharing)) ( [ Paper ](https://www.cin.ufpe.br/~if703/aulas/promethee.pdf))
- PROMETHEE VI   ([ Colab Demo ](https://colab.research.google.com/drive/14QdhifGitj4GK-QijRr1vj_dmGU2Pfh4?usp=sharing)) ( [ Paper ](https://www.cin.ufpe.br/~if703/aulas/promethee.pdf))
- PROMETHEE Gaia ([ Colab Demo ](https://colab.research.google.com/drive/1lj7IRKXcuRjrpoBp_KmQn_3sI3P_Qxju?usp=sharing)) ( [ Paper ](https://www.cin.ufpe.br/~if703/aulas/promethee.pdf))
- PIV ([ Colab Demo ](https://colab.research.google.com/drive/1PwJoBqYn1O2s22MqC9euP89Uyv4sedS0?usp=sharing)) ( [ Paper ](https://www.sciencedirect.com/science/article/abs/pii/S0360835218301360))
- PSI ([ Colab Demo ](https://colab.research.google.com/drive/1u9tN8cYl2mx6KK6yLW2oz6fuVoy8xcCI?usp=sharing)) ( [ Paper ](https://www.sciencedirect.com/science/article/abs/pii/S0261306909006396))
- Regime ([ Colab Demo ](https://colab.research.google.com/drive/1jcAcjAS92rxvE2urhc6HPixvzJ60HqEg?usp=sharing)) ( [ Paper ](https://link.springer.com/chapter/10.1007/978-3-030-15009-9_2))
- ROV ([ Colab Demo ](https://colab.research.google.com/drive/1sQAPCem0pcS29uf6-n4TpncXMXNx9JDh?usp=sharing)) ( [ Paper ](https://doi.org/10.5267/j.dsl.2015.12.001))
- SAW ([ Colab Demo ](https://colab.research.google.com/drive/1R4cIsu0jBP9-6zwww_bNxEEnVGrhnS2d?usp=sharing)) ( [ Paper ](https://media.neliti.com/media/publications/326766-simple-additive-weighting-saw-method-in-f8f093e8.pdf))
- SMART ([ Colab Demo ](https://colab.research.google.com/drive/1K93HXHBR_v2da95Hh_CB6AmTCqta-k3D?usp=sharing)) ( [ Paper ](https://doi.org/10.1007/978-1-4612-3982-6_4))
- TODIM ([ Colab Demo ](https://colab.research.google.com/drive/1EQqhhBQHHb8HT0TfuuVeFA2kwezsQYT1?usp=sharing)) ( [ Paper ](https://www.sciencedirect.com/science/article/abs/pii/S0377221707010740))
- TOPSIS ([ Colab Demo ](https://colab.research.google.com/drive/1s87DC5_oa9GvgVe98oAP1UIhduac09CB?usp=sharing)) ( [ Paper ](https://doi.org/10.1057/jors.1987.44))
- Fuzzy TOPSIS ([ Colab Demo ](https://colab.research.google.com/drive/1eKx7AOYrnG-kZcsBt28rMEtCrUO-j3J-?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.procs.2016.07.088))
- VIKOR ([ Colab Demo ](https://colab.research.google.com/drive/1egZiTNvI2eE-tyJ2m85MM6B3-qhiSjPG?usp=sharing)) ( [ Paper ](https://doi.org/10.1007/978-981-33-4745-8_8))
- Fuzzy VIKOR ([ Colab Demo ](https://colab.research.google.com/drive/1anfCnU2TSrW-Z5vMkS_qXFrYZ0ciQE53?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.eswa.2011.04.097))
- WINGS ([ Colab Demo ](https://colab.research.google.com/drive/1li1_cPxwEM3NOZ4hbI8RROXyOmXeoWew?usp=sharing)) ( [ Paper ](https://doi.org/10.1016/j.ejor.2013.02.007))
- WSM, WPM, WASPAS ([ Colab Demo ](https://colab.research.google.com/drive/1HbLwXI4HkrmI-lsNzDtBOlCiwxfJltHi?usp=sharing)) ( [ Paper ](https://doi.org/10.1007/978-981-33-4745-8_15))

4. Compare Methods:
- Compare Ranks & Ask chatGPT ([ Colab Demo ](https://colab.research.google.com/drive/1RfLNEJjaHjtn3Lb2cfEDqS-iblaC4GQZ?usp=sharing))
- Compare Fuzzy Ranks & Ask chatGPT ([ Colab Demo ](https://colab.research.google.com/drive/1pRO-E9xnk6DYEj_0DaEHUrUiCeIjmnXx?usp=sharing))
- Compare Weigths & Ask chatGPT ([ Colab Demo ](https://colab.research.google.com/drive/169hTJxP2APHrDA1h0fD1YEeu9s29wu0T?usp=sharing))

5. Advanced MCDA Methods:

- [3MOAHP](https://github.com/Valdecy/Method_3MOAHP) - Inconsistency Reduction Technique for AHP and Fuzzy-AHP Methods
- [pyMissingAHP](https://github.com/Valdecy/pyMissingAHP) - A Method to Infer AHP Missing Pairwise Comparisons
- [ELECTRE-Tree](https://github.com/Valdecy/ELECTRE-Tree) - Algorithm to infer the ELECTRE Tri-B method parameters
- [Ranking-Trees](https://github.com/Valdecy/Ranking-Trees) - Algorithm to infer the ELECTRE II, III, IV and PROMETHEE I, II, III, IV method parameters

# Acknowledgement 

This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Sabir Mohammedi Taieb (23.JANUARY.2023) - https://sabir97.github.io/ - Université Abdelhamid Ibn Badis Mostaganem (Algeria)
