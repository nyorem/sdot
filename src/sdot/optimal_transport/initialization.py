import sdot.core.inout as io
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.spatial
from . import optimal_transport_3, Results
from sdot.core.common import plot_cloud, plot_2d_cloud, plot_2d_tri, plot_tri, vprint, count_zeros, \
    plot_2d_hull, is_2d
from sdot.core.constants import VERBOSE

# Plot diracs correspondig to empty Laguerre cells
def plot_empty_cells(mu, Y, ind, title=""):
    fig = plt.figure()

    if is_2d(Y):
        plot_2d_cloud(Y[ind], fig=fig)
        plot_2d_tri(mu.vertices, mu.triangles, fig=fig, c="r")
    else:
        plot_cloud(Y[ind])
        plot_tri(mu.vertices, mu.triangles, fig=fig, c="r")

    plt.title("{} empty cells{}".format(len(ind), title))

# Maximal L2 distance between points
def diam(X):
    assert len(X.shape) in [1, 2]
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    dists = sp.spatial.distance.cdist(X, X, metric="euclidean")
    return np.max(dists)

# Maximal diameter in every direction
def diam2(X):
    m = -1
    for i in range(X.shape[1]):
        m = max(m, diam(X[:, i]))
    return 0.5 * m

# Bounding box of a point set
def bbox(X):
    d = X.shape[1]

    if d == 2:
        return np.array([[ np.min(X[:, 0]), np.max(X[:, 0])],
                         [ np.min(X[:, 1]), np.max(X[:, 1])]])

    elif d == 3:
        return np.array([[ np.min(X[:, 0]), np.max(X[:, 0])],
                         [ np.min(X[:, 1]), np.max(X[:, 1])],
                         [ np.min(X[:, 2]), np.max(X[:, 2])]])

# Volume of a bounding box (cube)
def volume_cube(bbox):
    # 2D point set
    two = 3
    if (len(bbox) == 2) or (bbox[2, 0] == bbox[2, 1]):
        two = 2

    return np.prod(bbox[:two, 1] - bbox[:two, 0])

# TODO: does not work (and can not work) with any domain
# Find lam and t such that lam * Y + t fits inside X
def rescale_set(X, Y, method="bbox"):
    methodl = method.lower()
    assert methodl in ["bbox", "bary"]

    if methodl == "bbox":
        # bary_X = np.mean(X, axis=0)
        # assert np.allclose(bary_X, [0, 0, 0]), "X must be centered at the origin"

        # 1. recenter Y on X
        bbox_X = bbox(X)
        bbox_Y = bbox(Y)

        t = np.mean(bbox_X, axis=1) - np.mean(bbox_Y, axis=1)

        # 2. rescale Y on X
        vol_X = volume_cube(bbox_X)
        vol_Y = volume_cube(bbox_Y)
        t_Y = np.mean(bbox_Y, axis=1)

        lam = vol_X / vol_Y
        # t = lam * t + t_Y * (1 - lam)
    elif methodl == "bary":
        bary_X = np.mean(X, axis=0)
        bary_Y = np.mean(Y, axis=0)

        t = bary_X - bary_Y

        # diam_X = diam(X)
        # diam_Y = diam(Y)
        diam_X = diam2(X)
        diam_Y = diam2(Y)

        lam = 0.5 * diam_X / diam_Y

    return lam, t

# Extend mu on (X, T) by setting mu = 0 on X
# (X, T) must contain (mu.vertices, mu.triangles)
def extend_density_2(mu, domain):
    from .density import Density_2

    if isinstance(domain, tuple):
        X, T = domain
    elif isinstance(domain, str):
        X, T = io.read_off(domain, ignore_prefix=True)
    elif isinstance(domain, float) or isinstance(domain, int):
        scale_big_source = domain
        X = np.array([[-scale_big_source, -scale_big_source, 0.0],
                      [ scale_big_source, -scale_big_source, 0.0],
                      [ scale_big_source,  scale_big_source, 0.0],
                      [-scale_big_source,  scale_big_source, 0.0]])

    # Remove already existing points
    del_X = []
    for i in range(len(X)):
        p = X[i]
        inside = np.any(np.sum(mu.vertices == p, axis=1) == len(p))
        if inside:
            del_X.append(i)
    big_X = np.delete(X, del_X, axis=0)
    new_X = np.concatenate((mu.vertices, big_X))

    # Retriangulate
    new_T = sp.spatial.Delaunay(new_X[:, 0:2]).simplices
    # new_T = T

    # Extended density is zero on the new points
    new_V = np.zeros(len(big_X))
    new_V = np.concatenate((mu.values, new_V))

    extended_mu = Density_2(new_X, new_T, V=new_V,
                            transform=mu.transform, cache=mu.cache)

    return extended_mu

# Create a linearly interpolated measure between Lebesgue on a bigger domain
# and mu (extended to be zero on the bigger domain).
def create_interpolation(mu, scale_big_source=1.0):
    from sdot.core.density import Interpolated_density
    from sdot.backend import GeogramBackend, NNRVDBackend
    from sdot.core.constants import PLANE_MESH, CUBE_MESH

    Density_2 = NNRVDBackend().available_settings["Plane"]
    Density_3 = GeogramBackend().available_settings["3D"]

    dim = mu._dim
    assert dim in [2, 3]

    big_support_filename = PLANE_MESH
    make_density = Density_2
    if dim == 3:
        big_support_filename = CUBE_MESH
        make_density = Density_3
    X, T = io.read_off(big_support_filename, ignore_prefix=True)
    X *= scale_big_source

    leb = make_density(X, T)

    interp = Interpolated_density([leb, mu])

    return interp

def _init_optimal_transport_3_local(mu, Y, nu, **kwargs):
    from ..aabb_tree import TriangleSoup

    eps = kwargs.get("eps_init")
    display = kwargs.get("display_empty", False)
    print_step = kwargs.get("print_step", 50)
    save_results = kwargs.get("save_results", False)
    verbose = kwargs.get("verbose", VERBOSE)

    tree = TriangleSoup(mu.vertices, mu.triangles)
    psi0 = tree.squared_distances(Y)

    if eps is None:
        return psi0

    A = mu.kantorovich(Y, nu, psi0)[Results.A]
    ind = np.where(A == 0)[0]

    it = 0
    results = []
    epss = []
    while len(ind) != 0:
        eps_step = eps

        if (it % print_step == 0):
            if save_results:
                results.append([it, len(ind)])
            if display:
                plot_empty_cells(mu, Y, ind, title=" / it = {}".format(it+1))
                plt.show()

        while True:
            # vprint("Trying with eps={}".format(eps_step))

            psi = psi0.copy()
            psi[ind] -= eps_step
            A = mu.kantorovich(Y, nu, psi)[Results.A]
            ind_new = np.where(A == 0)[0]

            # TODO: equality between indices
            if len(ind_new) > len(ind):
                # vprint("eps too large")
                eps_step /= 2
            else:
                # vprint("eps is ok")
                break

        epss.append(eps_step)
        if verbose:
            print("empty_cells={}, eps={}".format(len(ind), eps_step))

        psi0[ind] -= eps_step
        A = mu.kantorovich(Y, nu, psi0)[Results.A]
        ind = np.where(A == 0)[0]

        it += 1

    # if verbose:
    #     print("mean eps = {}".format(np.mean(np.array(epss))))

    if save_results:
        filename = "results/init/results_init_local.txt"
        results.append([it, 0])
        results = np.array(results)
        np.savetxt(filename, results)

    return psi0

def rescale_transform_weights(lam, t, Y, psi):
    norms = np.square(np.linalg.norm(Y, axis=1))
    phi = psi / lam + 2 * np.dot(Y, t) + (lam - 1) * norms
    return phi

def _init_optimal_transport_3_rescale(mu, Y, nu,
                                      psi0=None, save_results=False,
                                      **kwargs):
    X = mu.vertices

    if save_results:
        A = mu.kantorovich(Y, nu, psi0)[Results.A]
        zz = count_zeros(A)
        results = []
        results.append([0, zz])

    # 1) Translate and scale point cloud such that it fits inside X
    lam, t = rescale_set(X, Y, method="bbox")
    Z = lam * Y + t

    # 2) Rescale target measure
    # psi0 = _init_optimal_transport_3_local(mu, Z, nu)
    if psi0 is None:
        psi0 = np.zeros(len(Z))
    A0 = mu.kantorovich(Z, nu, psi0)[Results.A]
    zz0 = count_zeros(A0)
    assert zz0 == 0, "{} empty cells".format(zz0)
    nu *= A0.sum() / nu.sum()

    if save_results:
        results.append([1, zz0])

    # 3) Solve optimal transport
    psi = optimal_transport_3(mu, Z, nu)

    # 4) Transform weights
    phi0 = rescale_transform_weights(lam, t, Y, psi)

    # Check that the weights work after transformation
    A = mu.kantorovich(Y, nu, phi0)[Results.A]
    zz = count_zeros(A)
    assert zz == 0, "{} empty cells".format(zz)

    if save_results:
        results.append([2, zz])
        results = np.array(results)
        filename = "results/init/results_init_rescale.txt"
        np.savetxt(filename, results)

    return phi0, psi

def _init_optimal_transport_3_interp(mu, Y, nu,
                                     scale_big_source=None, maxit_interp=30, psi0=None,
                                     save_results=False,
                                     **kwargs):
    eps = kwargs.get("eps", 1e-8)

    if psi0 is None:
        psi = np.zeros(len(Y))
    else:
        psi = psi0

    nu /= nu.sum()
    t_min = 0.5 * (np.min(nu) - eps)
    maxit_interp = maxit_interp or math.ceil(-math.log2(t_min))
    assert maxit_interp >= 1
    vprint("t_min={}, maxit_interp={}".format(t_min, maxit_interp))

    # Interpolated density
    if scale_big_source is None:
        scale_big_source = max(diam2(mu.vertices), diam2(Y))
    mu_t = create_interpolation(mu, scale_big_source=scale_big_source)
    t = 1.0
    mu_t.weights = [t, 1 - t]

    if save_results:
        results = []

    for it in range(maxit_interp):
        vprint("it_interp {}: t={}".format(it + 1, t))

        A = mu.kantorovich(Y, nu, psi)[0]
        if save_results:
            mu.export_laguerre_cells(Y, psi, basename="/tmp/init_interp" + str(it))

        ind_zz = np.where(A == 0)[0]
        zz = len(ind_zz)

        if zz == 0:
            break

        if save_results:
            vprint("{} empty cells".format(zz))
            results.append([it, zz])

        # Normalize
        nu *= mu_t.mass() / nu.sum()

        # Solve OT
        psi = optimal_transport_3(mu_t, Y, nu, psi0=psi, **kwargs)

        t /= 2
        mu_t.weights = [t, 1 - t]

    if save_results:
        A = mu.kantorovich(Y, nu, psi)[0]
        mu.res = mu.res

        zz = count_zeros(A)
        vprint("{} empty cells".format(zz))
        results.append([it, zz])
        results = np.array(results)
        filename = "results/init/results_init_interp.txt"
        np.savetxt(filename, results)

    return psi

def init_optimal_transport_3(mu, Y, nu, method="local", **kwargs):
    # We first check if we really need to find good initial weights
    A = mu.kantorovich(Y, nu, psi=None)[Results.A]
    if count_zeros(A) == 0:
        vprint("All the Laguerre cells have positive mass, no need to initialize")
        return None

    methodl = method.lower()
    available_methods = { "interp": _init_optimal_transport_3_interp,
                          "local": _init_optimal_transport_3_local,
                          "rescale": _init_optimal_transport_3_rescale,
                        }

    assert methodl in available_methods.keys(), "Unkown method {}".format(method)

    init_func = available_methods.get(method)
    return init_func(mu, Y, nu, **kwargs)
