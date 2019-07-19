#! /usr/bin/env python3

from sdot.core.constants import PLANE_SIMPLE_MESH
from sdot.core.common import timeit
from sdot.backend import NNRVDBackend, GeogramBackend, LaguerreBackend
import sdot.core.inout as io
import numpy as np
import os

nnrvd_backend = NNRVDBackend()
geogram_backend = GeogramBackend()
laguerre_backend = LaguerreBackend()

# Source
## Planar
source_filename  = os.path.join(PLANE_SIMPLE_MESH)
X, T_X, = io.read_off(source_filename, ignore_prefix=True)
V, V_geogram = np.ones(len(X)), lambda x: 1.0

## NNRVD
mu_nnrvd_class   = nnrvd_backend.available_settings["Plane"]
mu_nnrvd   = mu_nnrvd_class(X, T_X, V=V)

## geogram
mu_geogram_class = geogram_backend.available_settings["Plane"]
mu_geogram = mu_geogram_class(X, T_X, V=V_geogram)

## laguerre
mu_laguerre_class = laguerre_backend.available_settings["Plane"]
mu_laguerre = mu_laguerre_class(X, T_X, V=V)

# Tests
eps = 1e-8
eps_laguerre = 1e-6
eps_laguerre_DA = 0.5 * 1e-1

def test_kantorovich_simple():
    N = 1000
    from sdot.core.common import random_inside_cube
    Y = random_inside_cube(N, a=2)
    Y[:, 2] = 0
    nu = np.repeat(1.0, N)
    psi = None

    with timeit("nnrvd"):
        A_nnrvd, DA_nnrvd = mu_nnrvd.kantorovich(Y, nu, psi=psi)[0:2]
    with timeit("geogram"):
        A_geogram, DA_geogram = mu_geogram.kantorovich(Y, nu, psi=psi)[0:2]
    with timeit("laguerre"):
        A_laguerre, DA_laguerre = mu_laguerre.kantorovich(Y, nu, psi=psi)[0:2]

    assert(np.linalg.norm(A_nnrvd - A_geogram) < eps)
    assert(np.linalg.norm(A_nnrvd - A_laguerre) < eps_laguerre)
    assert(np.linalg.norm(A_geogram - A_laguerre) < eps_laguerre)

    assert(np.linalg.norm((DA_nnrvd - DA_geogram).todense()) < eps)
    assert(np.linalg.norm((DA_nnrvd - DA_laguerre).todense()) < eps_laguerre_DA)
    assert(np.linalg.norm((DA_geogram - DA_laguerre).todense()) < eps_laguerre_DA)
