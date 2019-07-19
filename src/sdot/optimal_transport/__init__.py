import math
import numpy as np
import scipy as sp
from sdot.core.common import vprint, eprint
from sdot.core.constants import VERBOSE, PROFILE
if PROFILE:
    import time

# Solve DA X = A
# where ker(DA) = Vect(1, ..., 1), DA sparse
def solve_graph_laplacian(DA, A):
    import scipy.sparse
    import scipy.sparse.linalg

    if PROFILE:
        start = time.time()

    N = len(A)
    DAs = DA[0:(N-1), 0:(N-1)]
    As = A[0:N-1]
    ds = sp.sparse.linalg.spsolve(DAs, As)
    d = np.hstack((ds, [0]))

    if PROFILE:
        end = time.time()
        # eprint("Linear system: {}".format(end - start))

    return d

class Results:
    A           = 0
    DA          = 1
    DISTANCE    = 2
    C           = 3
    T           = 4

# Solve optimal transport between mu and (Y, nu) using a damped Newton's method
def optimal_transport_3_newton(mu, Y, nu, psi0=None, eps=1e-8, maxit=150, verbose=VERBOSE):
    N = len(Y)
    if psi0 is None:
        psi = np.zeros(N)
    else:
        assert len(psi0) == N
        psi = psi0

    assert (math.fabs(mu.mass() - nu.sum()) < 1e-2), "nu and mu must have the same mass: mu={} and nu={}".format(mu.mass(), nu.sum())

    mu.kantorovich(Y, nu, psi)
    A, DA = mu.res["A"], mu.res["DA"]
    if PROFILE and "f" in mu.res.keys():
        f = mu.res["f"]
        # eprint("Kantorovich: {}".format(f))
    g = A - nu
    eps0 = min(min(A), min(nu)) / 2

    zz0 = len(np.where(A == 0)[0])
    assert eps0 > 0, "{} empty Laguerre cells".format(zz0)

    it = 0

    # mu.export_laguerre_cells(Y, psi, basename="/tmp/ot" + str(it))
    # export_it = 0

    while (np.linalg.norm(g) > eps and it <= maxit):
        d = solve_graph_laplacian(DA, -g)

        alpha = 1
        psi0 = psi
        n0 = np.linalg.norm(g)

        while True:
            # eprint("psi = {}".format(psi))

            psi = psi0 + alpha * d

            mu.kantorovich(Y, nu, psi)
            A, DA = mu.res["A"], mu.res["DA"]
            # if PROFILE:
            #     f = mu.res["f"]
            #     eprint("Kantorovich: {}".format(f))
            g = A - nu

            # export_it += 1
            # mu.export_laguerre_cells(Y, psi, basename="/tmp/ot" + str(export_it))

            if (min(A) >= eps0 and np.linalg.norm(g) <= (1 - 0.5 * alpha) * n0):
                if PROFILE and "f" in mu.res.keys():
                    f = mu.res["f"]
                    # eprint("Kantorovich: {}".format(f))
                break
            alpha *= .5

        if PROFILE:
            eprint("it %d: |g|=%.10g, t=%g" % (it, np.linalg.norm(g), alpha))
        else:
            vprint("it %d: |g|=%.10g, t=%g" % (it, np.linalg.norm(g), alpha), verbose=verbose)

        it += 1

    if PROFILE:
        eprint("Number of iterations: {}".format(it))

    return psi

# Solve optimal transport between mu and (Y, nu) using a BFGS algorithm
# Need the climin library
def optimal_transport_3_bfgs(mu, Y, nu, psi0=None, eps=1e-8, maxit=500, verbose=VERBOSE):
    try:
        from climin.linesearch import LineSearch
    except ImportError:
        print("The BFGS solver is only available when the climin package is installed")
        import sys
        sys.exit(1)

    # Line search for the BFGS method
    class DampedNewtonLineSearch(LineSearch):
        def __init__(self, wrt, f, fprime, areas, eps0):
            self.f = f
            self.fprime = fprime
            self.areas = areas

            self.eps0 = eps0
            self.psi0 = wrt
            self.psi = self.psi0
            self.n0 = np.linalg.norm(fprime(self.psi0))
            self.alpha = 1.0

        # called like that: step_length = self.line_search.search(direction, None, args, kwargs)
        def search(self, direction, initialization=None, args=None, kwargs=None):
            self.alpha = 1.0

            while True:
                self.psi = self.psi0 + self.alpha * direction
                A = self.areas(self.psi)
                self.grad = self.fprime(self.psi)

                if (min(A) >= self.eps0 and np.linalg.norm(self.grad) <= (1 - 0.5 * self.alpha) * self.n0):
                    return self.alpha

                self.alpha *= .5

    N = len(Y)
    if psi0 is None:
        psi = np.zeros(N)
    else:
        assert len(psi0) == N
        psi = psi0

    def kantorovich(psi):
        mu.kantorovich(Y, nu, psi)
        f = mu.res["f"]
        if PROFILE:
            eprint("Kantorovich: {}".format(f))
        return -f

    def kantorovich_prime(psi):
        mu.kantorovich(Y, nu, psi)
        g = mu.res["A"] - nu
        if PROFILE:
            eprint("junk 0.0: |g|={}".format(np.linalg.norm(g)))
        return -g

    def areas(psi):
        # return -kantorovich_prime(psi) + nu
        mu.kantorovich(Y, nu, psi)
        return mu.res["A"]

    # line search
    from climin.bfgs import Bfgs
    mu.kantorovich(Y, nu, psi)
    A = mu.res["A"]
    eps0 = min(min(A), min(nu)) / 2
    line_search = DampedNewtonLineSearch(psi, kantorovich, kantorovich_prime, areas, eps0)

    # stop criterion
    # from climin.stops import AfterNIterations, NotBetterThanAfter
    # converged = NotBetterThanAfter(eps, 10, key="gradient_diff")

    # bfgs
    bfgs = Bfgs(psi, kantorovich, kantorovich_prime, line_search=line_search)
    bfgs.logfunc = print

    it = 0
    for info in bfgs:
        if it >= maxit or np.linalg.norm(line_search.grad) < eps:
            break
        it += 1

    if PROFILE:
        eprint("Number of iterations: {}".format(it))

    return line_search.psi

# Solve the optimal transport problem between:
# - mu: Density_2(X, T, V) a piecewise affine density defined on a triangulation
#         (X, T) of a surface embedded in \Rsp^3
# - nu: a discrete measure defined on the point cloud Y
def optimal_transport_3(mu, Y, nu, method="newton", **kwargs):
    method = method.lower()
    available_methods = ["newton", "bfgs"]
    assert method in available_methods

    if method == "newton":
        return optimal_transport_3_newton(mu, Y, nu, **kwargs)
    else:
        # method == "bfgs"
        return optimal_transport_3_bfgs(mu, Y, nu, **kwargs)

# Boundary edges of a triangulation
def find_boundary_edges(T):
    edges = []
    for t in T:
        ia, ib, ic = t
        edges.append((ia, ib))
        edges.append((ib, ic))
        edges.append((ic, ia))
    occ = dict()
    for e in edges:
        i, j = e
        vij = occ.get((i, j), 0)
        vji = occ.get((j, i), 0)
        occ[(i, j)] = vij + 1
        occ[(j, i)] = vji + 1
    bs = []
    for (k, v) in occ.items():
        if v == 1:
            s, t = k
            bs.append([s, t])
    bs = np.array(bs)
    return bs

# Conforming centroids i.e. projected on the boundary of the domain
# Y = 2D points
def conforming_centroids_2(mu, C, P, w, cfg=None, scale_source=1.0):
    from sdot.cgal_utils import conforming_lloyd_2

    if cfg is None:
        source_poly = np.array([[-1.0, -1.0],  # bl
                                [+1.0, -1.0],  # br
                                [+1.0, +1.0],  # ur
                                [-1.0, +1.0]]) # ul
        if isinstance(scale_source, tuple):
            source_poly[:, 0] *= scale_source[0]
            source_poly[:, 1] *= scale_source[1]
        else:
            source_poly *= scale_source
    else:
        if cfg.far_field:
            xmin, xmax, ymin, ymax = cfg.directional_source_box
            source_poly  = np.array([[xmin, ymin],  # bl
                                     [xmax, ymin],  # br
                                     [xmax, ymax],  # ur
                                     [xmin, ymax]]) # ul
        else:
            source_poly = np.array([[-1.0, -1.0],  # bl
                                    [+1.0, -1.0],  # br
                                    [+1.0, +1.0],  # ur
                                    [-1.0, +1.0]]) # ul

    new_C = np.zeros_like(C)
    CC = conforming_lloyd_2(C, P, -w, source_poly) # -w because of CGAL
    new_C[:, 0:2] = CC

    return new_C

# Y = 3D points
def conforming_centroids_3(mu, C, P, w):
    from sdot.cgal_utils import conforming_lloyd_3
    source_X = mu.vertices
    source_E = find_boundary_edges(mu.triangles)
    CC = conforming_lloyd_3(C, P, -w, source_X, source_E) # -w because of CGAL
    return CC

# Re-export initialization methods
from .initialization import *
