#! /usr/bin/env python3

# This example computes the optimal transport between a source measure that is uniform on a triangulated surface
# and a discrete target measure that is uniform on a point cloud.
# The triangulated surface is represented by an OFF file 'source.off'.
# The point cloud can be either given directly as a cloud/xyz/txt file or can be randomly sampled on a
# triangulated surface 'target.off', the number of samples being 'nY'.

# Usage:
# - ot.py source.off target.[cloud, xyz, txt]
# - ot.py source.off target.off nY
# Examples:
# - python ot.py examples/assets/sphere.off examples/assets/hemisphere.off 1000
# - python ot.py examples/assets/sphere.off examples/assets/clouds/sphere_1k.cloud

# Python imports
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

# Imports from the 'sdot' library
from sdot.core.common import random_in_triangulation, clean_triangulation, \
     plot_tri, plot_2d_tri, is_2d # for sampling and displaying triangulations
from sdot.backend import Density
import sdot.core.inout as io # for reading OFF files
from sdot.optimal_transport import optimal_transport_3, init_optimal_transport_3 # for solving OT problems

assert len(sys.argv) in [3, 4], "Usage: {} source.off target.[cloud, xyz, txt, off]".format(sys.argv[0])

source_filename = sys.argv[1]
assert os.path.isfile(source_filename), "{} does not exist!".format(source_filename)
target_filename = sys.argv[2]
assert os.path.isfile(target_filename), "{} does not exist!".format(target_filename)
target_ext = os.path.splitext(target_filename)[1]

# Parameters for optimal transport
eps = 1e-10 # numerical error in the Newton's algorithm
maxit = 500 # maximum number of iterations in the Newton's algorithm
# Parameters for initializing optimal transport
eps_init = 1e-2

# Loading source and target files
X, T_X = io.read_off(source_filename, ignore_prefix=True)

if target_ext in [".cloud", ".xyz", ".txt"]:
    assert len(sys.argv) == 3, "Usage: {} source.off target.[cloud, xyz, txt]".format(sys.argv[0])
    Y = np.loadtxt(target_filename)
elif target_ext in [".off", ".noff", ".coff"]:
    assert len(sys.argv) == 4, "Usage: {} source.off target.off N".format(sys.argv[0])
    nY = int(sys.argv[3])
    X_Y, T_Y = io.read_off(target_filename, ignore_prefix=True)
    Y, _ = random_in_triangulation(X_Y, T_Y, nY)
else:
    raise RuntimeError("Extension {} not supported".format(target_ext))

# Source
mu = lambda x: 1.0

# Target
## Uniform density
nu = lambda y: 1.0
## Non-uniform density: linear from a to b
# def linear_target(x, a, b, m, M):
#     alpha = (b - a) / (M - m)
#     beta = a - alpha * m
#     return alpha * x + beta
# axis = 0
# m, M = np.min(Y[:, axis]), np.max(Y[:, axis])
# nu = lambda y: linear_target(y[axis], 0.3, 1, m, M)

nu_Y = np.apply_along_axis(nu, 1, Y)

# Optimal transport setup
## Source
mu_X = np.apply_along_axis(mu, 1, X)
mu = Density(X, T_X, V=mu_X)

## Target
nu_Y *= mu.mass() / nu_Y.sum()

## Find initial weights
print("Initial weights computation")
# psi0 = None # Uncomment if no initialization is needed
psi0 = init_optimal_transport_3(mu, Y, nu_Y, method="local", eps_init=eps_init)

## Optimal transport
start_ot = time.time()
print("Optimal transport")
psi = optimal_transport_3(mu, Y, nu_Y, psi0=psi0, eps=eps, method="newton", maxit=maxit)
end_ot = time.time()
print("Running time = {}s".format(end_ot - start_ot))

# Results
# We can export the Laguerre cells for (Y, psi) as a mesh '/tmp/final_laguerre.obj' that can be opened
# with Meshlab for instance
mu.export_laguerre_cells(Y, psi=psi, basename="/tmp/final_laguerre")

# Now, we plot the dual triangulation (C, T) where C are the centroids of the Laguerre cells
C, T = mu.res["C"], mu.res["T"]
T = clean_triangulation(C, T)

# conforming_centroids_* means that we project the centroids on the boundary
# of the triangulated surface
if is_2d(X):
    from sdot.optimal_transport import conforming_centroids_2
    conforming_C = conforming_centroids_2(mu, C, Y, psi)
    plot_2d_tri(conforming_C, T)
else:
    from sdot.optimal_transport import conforming_centroids_3
    conforming_C = conforming_centroids_3(mu, C, Y, psi)
    plot_tri(conforming_C, T)
plt.show()
