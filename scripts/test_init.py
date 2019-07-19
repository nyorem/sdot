#! /usr/bin/env python3

# Usage: python examples/misc/test_init.py source.off target

# Examples:
# - python examples/misc/test_init.py examples/assets_link/plane.off examples/assets_link/clouds/left_1k.txt (WITH source transform, uncomment lines 72-75)
# - python examples/misc/test_init.py examples/assets_link/ring.off examples/assets_link/clouds/circle_1000.txt
# - python examples/misc/test_init.py examples/assets_link/two_circles.off examples/assets_link/clouds/circle_small_1000.txt
# - python examples/misc/test_init.py examples/assets_link/two_planes.off examples/assets_link/clouds/circle_small_1000.txt

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from sdot.core.constants import CUBE_MESH
from sdot.core.common import plot_cloud, vprint, random_in_triangulation, clean_triangulation, count_zeros, eprint, \
     plot_tri, add_noise, plot_2d_cloud, plot_2d_tri, transform_box, plot_2d_hull, is_2d
from sdot.core.constants import DEBUG, PROFILE, VERBOSE
from sdot.backend import GeogramBackend, NNRVDBackend
# Density_2 = GeogramBackend().available_settings["Plane"]
Density_2 = NNRVDBackend().available_settings["Plane"]
Density_3 = GeogramBackend().available_settings["3D"]
import sdot.core.inout as io
from sdot.optimal_transport import optimal_transport_3, Results, init_optimal_transport_3, rescale_set
from sdot.optimal_transport.initialization import diam, diam2
import sdot.optimal_transport.interp as interp

assert len(sys.argv) in [3, 4], "Usage: {} source.off target.[cloud, off] [nY]".format(sys.argv[0])

def check_laguerre_cells(mu, Y, nu, psi):
    # mu.kantorovich(Y, nu, psi)
    A = mu.res["A"]
    return count_zeros(A), A

source_filename = sys.argv[1]
assert os.path.isfile(source_filename), "{} does not exist!".format(source_filename)
target_filename = sys.argv[2]
assert os.path.isfile(target_filename), "{} does not exist!".format(target_filename)
target_ext = os.path.splitext(target_filename)[1]

tests_init = { "local"  : True,
               "rescale": True,
               "interp" : True
             }
ncalls = { "local"  : None,
           "rescale": None,
           "interp" : None
         }

if target_ext in [".cloud", ".xyz", ".txt"]:
    assert len(sys.argv) == 3
    Y = np.loadtxt(target_filename)
    nY = len(Y)
elif target_ext in [".off", ".noff", ".coff"]:
    assert len(sys.argv) == 4, "Must give the number of Diracs"
    nY = int(sys.argv[3])
    X_Y, T_Y = io.read_off(target_filename, ignore_prefix=True)
    Y, _ = random_in_triangulation(X_Y, T_Y, nY)
else:
    raise RuntimeError("Not supported extension: {}".format(target_ext))

eps_ot = 1e-8
eps_init = 1e-2 # local
scale_big_source = 1.0 # interp
maxit_interp = 30
save_results = True
fontsize = 25

# Source
X, T_X = io.read_off(source_filename, ignore_prefix=True)

## Smaller source
# box_origin = [ -1.0, 1.0, -1.0, 1.0 ]
# box_source = [ -1.0 , 0.0, -1.0, 1.0] # TODO: change this
# transform_box_function = transform_box(box_origin, box_source)
# X = np.apply_along_axis(transform_box_function, 1, X)

if X.shape[1] == 2:
    XX = np.zeros((len(X), 3))
    XX[:, 0:2] = X
    X = XX

mu = Density_2(X, T_X)

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

nu = np.apply_along_axis(nu, 1, Y)

fig = plt.figure()
if is_2d(X):
    ax = plt.gca()
    ax.set_aspect("equal")
    plot_2d_cloud(Y, fig=fig, cmap="b", s=10, labels=["target"])
    # plot_2d_hull(X[:, 0:2], fig=fig, c="r")
    plot_2d_tri(X, T_X, fig=fig, c="r", label="source")
else:
    ax = plt.gca(projection="3d")
    plot_cloud(Y, ax=ax, fig=fig, cmap="b",labels=["target"], s=15)
    plot_tri(X, T_X, fig=fig, color="r", label="source")
# plt.legend()
plt.show() # TODO

## Local
if tests_init["local"]:
    start_local = time.time()
    psi0_local = init_optimal_transport_3(mu, Y, nu, method="local",
                                            eps_init=eps_init, display_empty=False,
                                            save_results=save_results)
    end_local = time.time()
    vprint("Initialization local: {}s\n".format(end_local - start_local))
    ncalls["local"] = mu.kantorovich.calls.calls
    mu.kantorovich.calls.calls = 0

    zz, A = check_laguerre_cells(mu, Y, nu, psi0_local)
    assert zz == 0, "local: {} empty cells".format(zz)

## Rescale
if tests_init["rescale"]:
    # Center X at the origin
    XX = np.copy(X)
    # bary_X = np.mean(XX, axis=0)
    # XX -= bary_X

    YY = np.copy(Y)

    lam, t = rescale_set(XX, YY, method="bary")
    Z = YY * lam + t
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    plot_2d_cloud(YY, fig=fig, cmap="b", s=10)
    plot_2d_cloud(Z, fig=fig, cmap="g", s=10)
    plot_2d_tri(X, T_X, fig=fig, c="r")
    # plot_2d_hull(X[:, 0:2], fig=fig, c="r")
    # plt.legend()
    plt.show()

    start_rescale = time.time()
    psi0_rescale, phi0_rescale = init_optimal_transport_3(mu, YY, nu, method="rescale",
                                                          save_results=save_results)
    end_rescale = time.time()
    vprint("Initialization rescale: {}s\n".format(end_rescale - start_rescale))

    zz, A = check_laguerre_cells(mu, Y, nu, psi0_rescale)
    assert zz == 0, "rescale: {} empty cells".format(zz)

    optimal_transport_3(mu, YY, nu, psi0=psi0_rescale, verbose=True, eps=eps_ot)

    # mu.kantorovich(Z, nu, phi0_rescale)[0]
    # mu.kantorovich(YY, nu, psi0_rescale)[0]

# Interp
# TODO: handle 3D domains
if tests_init["interp"]:
    start_interp = time.time()
    if is_2d(Y):
        vprint("2D")
        psi0_interp = init_optimal_transport_3(mu, Y, nu, method="interp",
                                               scale_big_source=None,
                                               maxit_interp=maxit_interp,
                                               eps=eps_ot, verbose=VERBOSE,
                                               save_results=save_results)
    else:
        vprint("3D")
        # Create 3D uniform density
        X_leb, T_leb = io.read_off(os.path.join(CUBE_MESH))
        scale_leb = 1.1 * max(diam(X), diam(Y))
        X_leb *= scale_leb
        leb = Density_3(X_leb, T_leb)

        psi0_interp = interp.optimal_transport_surface(leb, mu, Y, nu,
                                                       maxit_interp=maxit_interp, psi0=None,
                                                       update_method=None,
                                                       save_results=save_results,
                                                       adaptive_t=False,
                                                       eps=eps_ot, verbose=VERBOSE)
    end_interp = time.time()

    vprint("Initialization interp: {}s\n".format(end_interp - start_interp))
    ncalls["interp"] = mu.kantorovich.calls.calls
    mu.kantorovich.calls.calls = 0

    zz, A = check_laguerre_cells(mu, Y, nu, psi0_interp)
    assert zz == 0, "interp: {} empty cells".format(zz)

# Plot results
if not save_results:
    sys.exit(0)

results = { "local": None, "rescale": None, "interp": None }
if os.path.exists("results/init/results_init_local.txt"):
    results["local"] = np.loadtxt("results/init/results_init_local.txt")
if os.path.exists("results/init/results_init_rescale.txt"):
    results["rescale"] = np.loadtxt("results/init/results_init_rescale.txt")
if os.path.exists("results/init/results_init_interp.txt"):
    results["interp"] = np.loadtxt("results/init/results_init_interp.txt")

colors = { "local"  : "r",
           "rescale": "b",
           "interp" : "g"
         }

# to_plot = tests_init
to_plot = { "local"  : True,
            "rescale": False,
            "interp" : False
          }
nplots = 0
for key in to_plot.keys():
    if not to_plot[key]:
        continue
    nplots += 1
    val = results[key]
    col = colors[key]
    if len(val.shape) == 1:
        val = val[np.newaxis, :]
    plt.plot(val[:, 0], val[:, 1], c=col, label=key)
    plt.xlabel("iterations", fontsize=fontsize)
    plt.ylabel("empty cells", fontsize=fontsize)

if nplots >= 2:
    plt.legend()

plt.show()
