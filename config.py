"""
Phononic reservoir parameters: Qi et al. APL 117, 183503 (2020) Eqs. (5)-(6), Fig. 3.
RC framework: Shaabani Shishavan et al., Phys. Rev. Research 7, L042008 (2025).

Fig. 3 (normalised): Delta1=4, kappa=-9, Delta2=Delta1/2+kappa, c21=1, f=20.
"""

import math

# Coupled-mode parameters (Qi 2020 Fig. 3)
DELTA1 = +4.0
KAPPA = -9.0
DELTA2 = DELTA1 / 2 + KAPPA
GAMMA21 = 1.0
F_AVG = 20.0
DELTA_F = 0.0

# Physical scale: omega1/(2*pi) = 3.86 MHz, Q1 = 1000; tau = gamma1 * t_physical (~82.4 us)
OMEGA1_HZ = 3.86e6
Q1 = 1000
GAMMA1_RAD_S = 2 * math.pi * OMEGA1_HZ / (2 * Q1)  # ~12126 rad/s

# RC discretisation
SYMBOL_DURATION_TAU = 5.0
N_VIRTUAL = 512
N_FFT_FEATURES = 512
PRECONDITION_SYMBOLS = 50
WARMUP_SYMBOLS = 200
N_SYMBOLS = 4000
TRAIN_FRACTION = 0.75
N_TEST_SYMBOLS = 1000
REG_LAMBDA = 1e-3
REGRESSION = "ridge"  # "ridge" or "lasso"

# Mackey-Glass benchmark
MG_TAU = 17
MG_BETA = 0.2
MG_GAMMA_MG = 0.1
MG_N_EXP = 10
MG_DT = 1.0

# Lorenz benchmark
LZ_SIGMA = 10.0
LZ_RHO = 28.0
LZ_BETA_LZ = 8.0 / 3.0
LZ_DT = 0.01
LZ_SUBSAMPLE = 10

# Rossler benchmark
RS_A = 0.2
RS_B = 0.2
RS_C = 5.7
RS_DT = 0.01
RS_SUBSAMPLE = 50

# ODE solver
SOLVER_METHOD = "RK45"
SOLVER_RTOL = 1e-8
SOLVER_ATOL = 1e-10
