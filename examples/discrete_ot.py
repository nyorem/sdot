#! /usr/bin/env python3

# Usage: discrete_ot.py source.off target.[cloud, off] nX nY

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from sdot.core.common import vprint, random_in_triangulation, clean_triangulation, eprint, \
     plot_tri, plot_2d_tri, is_2d, timeit
from sdot.core.constants import DEBUG, PROFILE
from sdot.backend import Density
import sdot.core.inout as io
from sdot.optimal_transport import init_optimal_transport_3, optimal_transport_3
from sdot.optimal_transport.discrete import sinkhorn, cost_matrix, sinkhorn_to_kantorovich

assert len(sys.argv) in [4, 5], "Usage {} source.off target.[cloud, off] [nX] [nY]".format(sys.argv[0])

source_filename = sys.argv[1]
assert os.path.isfile(source_filename), source_filename
target_filename = sys.argv[2]
assert os.path.isfile(target_filename), target_filename
target_ext = os.path.splitext(target_filename)[1]

# OT
eps = 1e-10
maxit = 500
# Discrete
reg = 0.001
maxit_discrete = 2000
# Init
# eps_init = 1e-2

if target_ext in [".cloud", ".xyz", ".txt"]:
    assert len(sys.argv) == 4
    Y = np.loadtxt(target_filename)
    nX = int(sys.argv[3])
    nY = len(Y)
elif target_ext in [".off", ".noff", ".coff"]:
    assert len(sys.argv) == 5, "Must give the number of Diracs in the source and target"
    nX = int(sys.argv[3])
    nY = int(sys.argv[4])
    X_Y, T_Y = io.read_off(target_filename, ignore_prefix=True)
    Y, _ = random_in_triangulation(X_Y, T_Y, nY)
else:
    raise RuntimeError("Extension {} not supported".format(target_ext))

# Source
X, T_X = io.read_off(source_filename, ignore_prefix=True)
mu = lambda x: 1.0

# Target
## Uniform
nu = lambda y: 1.0
## Non-uniform
# def linear_target(x, a, b, m, M):
#     alpha = (b - a) / (M - m)
#     beta = a - alpha * m
#     return alpha * x + beta
# axis = 0
# m, M = np.min(Y[:, axis]), np.max(Y[:, axis])
# # min_nu = 0.1 # fandisk 1k
# min_nu = 0.3 # torus 1k
# nu = lambda y: linear_target(y[axis], min_nu, 1, m, M)

nu_Y = np.apply_along_axis(nu, 1, Y)

# Optimal transport setup
# Discretize source
mu_semi = Density(X, T_X)
X, _ = random_in_triangulation(X, T_X, nX)
mu_X = np.apply_along_axis(mu, 1, X)

# Cost matrix
M = cost_matrix(X, Y, metric="sqeuclidean")

## Initial weights
# vprint("Initial weights")
# # psi0 = None
# psi0 = init_optimal_transport_3(mu, Y, nu_Y, method="local", eps_init=eps_init)
# # psi0 = np.loadtxt("/tmp/psi0.txt")

## Normalization (sum to 1)
nu_Y /= nu_Y.sum()
mu_X /= mu_X.sum()

## Optimal transport
vprint("Optimal transport")
# Discrete
with timeit("Discrete"):
    _, log = sinkhorn(mu_X, nu_Y, M, reg=reg, verbose=True, eps=eps, log=True, maxit=maxit_discrete)
    phi, psi = sinkhorn_to_kantorovich(log["u"], log["v"], reg)
# Semi-discrete
with timeit("Semi-discrete"):
    nu_Y *= mu_semi.mass() / nu_Y.sum()
    psi_semi = optimal_transport_3(mu_semi, Y, nu_Y, psi0=None, eps=eps, method="newton", maxit=maxit)

# Results
mu_semi.export_laguerre_cells(Y, psi=psi, basename="/tmp/final_laguerre")

mu_semi.kantorovich(Y, nu_Y, psi)
A, C, T = mu_semi.res["A"], mu_semi.res["C"], mu_semi.res["T"]
T = clean_triangulation(C, T)

mu_semi.kantorovich(Y, nu_Y, psi_semi)
A_semi, C_semi, T_semi = mu_semi.res["A"], mu_semi.res["C"], mu_semi.res["T"]
T_semi = clean_triangulation(C_semi, T_semi)

print("err={}".format(np.linalg.norm(A - nu_Y)))
print("err_semi={}".format(np.linalg.norm(A_semi - nu_Y)))

if is_2d(X):
    from sdot.optimal_transport import conforming_centroids_2
    conforming_C = conforming_centroids_2(mu_semi, C, Y, psi)
    conforming_C_semi = conforming_centroids_2(mu_semi, C_semi, Y, psi_semi)
    plot_2d_tri(conforming_C, T, c="b")
    plot_2d_tri(conforming_C_semi, T_semi, c="r")
else:
    from sdot.optimal_transport import conforming_centroids_3
    conforming_C = conforming_centroids_3(mu_semi, C, Y, psi)
    conforming_C_semi = conforming_centroids_3(mu_semi, C_semi, Y, psi_semi)
    plot_tri(conforming_C, T, c="b")
    plot_tri(conforming_C_semi, T_semi, c="r")
plt.show()
