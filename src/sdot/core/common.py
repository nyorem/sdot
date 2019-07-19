from itertools import cycle
import math
import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sdot.core.constants import VERBOSE, DEBUG

# Test if a point set is 2D (ie.e with the z)
def is_2d(Y):
    return Y.shape[1] == 2 or np.allclose(Y[:, 2], Y[0, 2])

# print if the VERBOSE constant is defined
def vprint(*objects, verbose=VERBOSE, **kwargs):
    if verbose:
        print(*objects, **kwargs)

# print if the DEBUG constant is defined
def dprint(*args, **kwargs):
    vprint(*args, verbose=DEBUG, **kwargs)

# print to stderr
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

# Project array on the unit sphere
def normalize_array(X):
    N = np.linalg.norm(X, axis=1).reshape(-1, 1)
    return X / N

# Number of zeros in an array
def count_zeros(A):
    return len(np.where(A == 0)[0])

# Add noise to an array
def add_noise(X, eps=1e-6):
    return X + eps * np.random.rand(*X.shape)

# Plot the convex hull of a 2D point set
def plot_2d_hull(X, fig=None, **kwargs):
    hull = sp.spatial.ConvexHull(X)
    fig = fig or plt.figure()
    for simplex in hull.simplices:
        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], **kwargs)

# Plot a 2D triangulation
def plot_2d_tri (X, T, fig=None, **kwargs):
    x, y = X[:, 0], X[:, 1]
    fig = fig or plt.figure()
    plt.triplot(x, y, triangles=T, **kwargs)

# Plot a function over a 2D triangulation
def plot_2d_tri_func(X, T, V, fig=None, kind="linear", N=200, colorbar=True):
    import scipy.interpolate

    x, y = X[:, 0], X[:, 1]
    fig = fig or plt.figure()
    # plt.triplot(x, y, triangles=T)
    f = sp.interpolate.interp2d(x, y, V, kind=kind, fill_value=0)
    xs, ys = np.linspace(min(x), max(x), num=N), np.linspace(min(y), max(y), num=N)
    zs = f(xs, ys)
    plt.pcolor(xs, ys, zs, cmap="Greys_r")
    if colorbar:
        plt.colorbar()
    plt.axis("off")

# Plot 2D point clouds
def plot_2d_cloud(clouds, fig=None, cmap=None, colorbar=False, labels=None, **kwargs):
    colors = cycle("bgrcmk")
    fig = fig or plt.figure()
    if not isinstance(clouds, list):
        clouds = [clouds]
    if labels is not None:
        assert len(labels) == len(clouds)
    for i in range(len(clouds)):
        pts = clouds[i]
        if labels is not None:
            label = labels[i]
        else:
            label = None
        if cmap is not None:
            plt.scatter(pts[:,0], pts[:,1], c=cmap, label=label, **kwargs)
        else:
            plt.scatter(pts[:,0], pts[:,1], c=next(colors), label=label, **kwargs)
    if (cmap is not None) and colorbar:
        plt.colorbar()

# Plot a 3D triangulation
def plot_tri(X, T, fig=None, **kwargs):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    fig = fig or plt.figure()
    ax = fig.gca(projection='3d')
    if "c" in kwargs.keys():
        kwargs["color"] = kwargs.pop("c")
    ax.plot_trisurf(x, y, z, triangles=T, **kwargs)

# Plot 3D point clouds
def plot_cloud(clouds, ax=None, fig=None, cmap=None, no_grid=False, colorbar=False, labels=None, **kwargs):
    colors = cycle("bgrcmk")
    if ax is None:
        fig = fig or plt.figure()
        ax = fig.gca(projection="3d")
    else:
        assert fig is not None
    if no_grid:
        ax._axis3don = False
    if not isinstance(clouds, list):
        clouds = [clouds]
    if labels is not None:
        assert len(labels) == len(clouds)
    for i in range(len(clouds)):
        pts = clouds[i]
        if labels is not None:
            label = labels[i]
        else:
            label = None
        if cmap is not None:
            scat = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=cmap, label=label, **kwargs)
        else:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=next(colors), label=label, **kwargs)
    if (cmap is not None) and colorbar:
        fig.colorbar(scat)

# Random points inside a circle
def random_inside_circle(origin=[0, 0, 0], r=1.0, N=100):
    theta = np.random.uniform(low=0, high=2*math.pi, size=N)
    length = np.sqrt(np.random.uniform(low=0, high=1.0, size=N))
    origin = np.array(origin)
    xs = origin[0] + r * length * np.cos(theta)
    ys = origin[1] + r * length * np.sin(theta)
    zs = np.repeat(origin[2], N)
    return np.vstack((xs, ys, zs)).T

# Random points inside a sphere
def random_on_sphere(N, r=1.0, origin=[0, 0, 0]):
    origin = np.array(origin)
    Y = np.random.normal(size=(N, 3))
    Y = normalize_array(Y)
    Y = r
    Y += origin
    return Y

# Random point on a square
def random_on_square(N, center=[0, 0, 0], a=1.0):
    Ns = N // 4
    side = 0.5 * a
    def get_random():
        return np.random.uniform(low=-side, high=side, size=Ns)
    m = np.repeat(-side, Ns)
    M = -m
    zs = np.repeat(0.0, Ns)
    xmax = np.vstack((M, get_random(), zs)).T
    xmin = np.vstack((m, get_random(), zs)).T
    ymax = np.vstack((get_random(), m, zs)).T
    ymin = np.vstack((get_random(), M, zs)).T
    Y = np.concatenate((xmax, xmin, ymax, ymin))
    Y = Y + center
    return Y

# Random points inside a cube
def random_inside_cube(N, a=1.0, center=[0, 0, 0]):
    if isinstance(a, float) or isinstance(a, int):
        a = [a, a, a]
    if len(a) == 2:
        a = [a[0], a[1], 0]
    if len(center) == 2:
        center = [center[0], center[1], 0]
    side = [0.5*a[0], 0.5*a[1], 0.5*a[2]]

    xs = np.random.uniform(low=-side[0], high=side[0], size=N)
    ys = np.random.uniform(low=-side[1], high=side[1], size=N)
    zs = np.random.uniform(low=-side[2], high=side[2], size=N)

    Y = np.vstack((xs, ys, zs))
    return center + Y.T

# Random points inside a (ND) triangle
# Reference: http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
def random_in_triangle (a, b, c):
    def rand01 ():
        return np.random.uniform()

    r1 = math.sqrt(rand01())
    r2 = rand01()

    return (1 - r1) * a + (r1 * (1 - r2)) * b + (r2 * r1) * c

# (Unsigned) area of a (ND) triangle (a, b, c).
def area_triangle (a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

# Barycentric coordinates of point 'p' with respect to triangle [a, b, c]
def barycentric(a, b, c, p):
    # dim = len(a)
    # A = np.array([a, b, c]).T
    # one = np.ones(dim)
    # A = np.vstack((A, one))
    # pp = np.ones(dim+1)
    # pp[:dim] = p
    # coords = np.linalg.lstsq(A, pp, rcond=-1)
    # return coords[0]

    A = np.array([a, b, c]).T
    coords = np.linalg.lstsq(A, p, rcond=-1)
    return coords[0]

# Test if the values of a numpy array are all between 'a' and 'b'
def between(X, a=0.0, b=1.0):
    return len(np.where((X >= a) & (X <= b))[0]) == len(X)

# Test if 'p' is on the plane defined by [a, b, c]
def on_plane(a, b, c, p, eps=1e-8):
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    p = np.asarray(p)
    n = np.cross(b - a, c - a)
    return np.inner(n, p - a) <= eps

# Test if p inside triangle [a, b, c]
def inside_triangle(a, b, c, p):
    if not on_plane(a, b, c, p):
        return False

    coords = barycentric(a, b, c, p)
    return between(coords, a=0.0, b=1.0)

# Integrate a function over a triangle (a, b, c)
# The function must be at most affine.
# Equivalent (for callable) to integrate_quadrature(a; b, c, V, order=1)
def integrate_centroid(a, b, c, V):
    if callable(V):
        val = V(a + b + c) / 3
    else:
        va, vb, vc = V
        val = (va + vb + vc) / 3

    return area_triangle(a, b, c) * val

# Integrate a function over a triangle (a, b, c) using quadrature formulas
def integrate_quadrature (a, b, c, f, order=1):
    def barycentric_eval(alpha, beta, gamma):
        p = alpha * a + beta * b + gamma * c
        return f(p)

    A = area_triangle(a, b, c)

    s = 0
    if order == 1:
        s += 1.0 / 2.0 * barycentric_eval(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    elif order == 2:
        s += 1.0 / 6.0 * barycentric_eval(1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0)
        s += 1.0 / 6.0 * barycentric_eval(4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
        s += 1.0 / 6.0 * barycentric_eval(1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
    elif order == 3:
        s += -27.0 / 96.0 * barycentric_eval(1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0)
        s += 25.0  / 96.0 * barycentric_eval(1.0 / 5.0, 1.0 / 5.0, 3.0 / 5.0)
        s += 25.0  / 96.0 * barycentric_eval(1.0 / 5.0, 3.0 / 5.0, 1.0 / 5.0)
        s += 25.0  / 96.0 * barycentric_eval(3.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0)
    else:
        raise ValueError("Unsupported order %d" % order)

    return 2 * A * s

# Sample N points inside a triangulation defined by (X, T)
# The density can be non uniforn if V is None.
# If V is not None, it represents the values of the density
# at the vertices of the triangulation
# or is a callable function invoked at the vertices of the triangulation.
def random_in_triangulation (X, T, N=1000, V=None):
    dim = X.shape[1]
    assert dim >= 2 and dim <= 3

    def compute_area (ti):
        t = ti[0:3]
        a, b, c = X[t]
        if V is None:
            area = area_triangle(a, b, c)
        else:
            if callable(V):
                area = integrate_quadrature(a, b, c, V, order=2)
            else:
                area = integrate_centroid(a, b, c, V[ti[3]])
        return area

    I = np.arange(len(T)).reshape((-1, 1)) # indices of triangles
    areas = np.apply_along_axis(compute_area, 1, np.hstack((T, I)))
    weights = areas / areas.sum()

    points = []
    tindices = []
    ns = np.random.choice(len(T), N, p=weights)

    def compute_point (n):
        t = T[n]
        a, b, c = X[t]
        p = random_in_triangle(a, b, c)
        if dim == 2:
            x, y = p
            return x, y, n
        elif dim == 3:
            x, y, z = p
            return x, y, z, n

    if dim == 2:
        compute_point_vec = np.vectorize(compute_point, otypes=[np.float, np.float, np.int])
        px, py, tindices = compute_point_vec(ns)
        points = np.vstack((px, py)).T
    elif dim == 3:
        compute_point_vec = np.vectorize(compute_point, otypes=[np.float, np.float, np.float, np.int])
        px, py, pz, tindices = compute_point_vec(ns)
        points = np.vstack((px, py, pz)).T

    return points, tindices

# Remove duplicate triangles in a triangulation and orient the normals consistently
def clean_triangulation(X, T, inside=True):
    T_sorted = np.sort(T, axis=1)
    new_T = np.vstack([tuple(row) for row in T_sorted])

    # Flip normals
    o = np.mean(X, axis=0)

    for t in new_T:
        pa, pb, pc = X[t[0]], X[t[1]], X[t[2]]
        nt = np.cross(pb - pa, pc - pa)
        nt = nt / np.linalg.norm(nt)

        u = pa - o
        if not inside:
            u = -u

        if np.inner(nt, u) >= 0:
            t[1], t[2] = t[2], t[1]

    return new_T

# Return the points in X which are the closest to Y
# reference = X, queries = Y
def closest_from(X, Y, low_memory=False):
    if low_memory:
        tree = sp.spatial.KDTree(X)
        return tree.query(Y)[1]
    else:
        return sp.spatial.distance.cdist(Y, X).argmin(axis=1)

# Affine transformation from box1 to box2
def transform_box(box1, box2):
    xma, xMa, yma, yMa = box1
    xmb, xMb, ymb, yMb = box2

    # Avoid division by zero
    assert xma != xMa
    assert yma != yMa

    a = (xmb - xMb) / (xma - xMa)
    b = 0
    c = 0
    d = (ymb - yMb) / (yma - yMa)
    e = xmb - a * xma
    f = ymb - d * yma

    M = np.array([[a, b], [c, d]])
    v = np.array([e, f])

    def f(p):
        p = p[:2]
        return M.dot(p) + v
    return f

# Bilinear interpolation on a 2D numpy array
# source: http://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# Mesh subdivision
def subdivide(X, T, N=None, weights=None, normal_at=None):
    if weights is None:
        weights = np.repeat(1.0, 3)
    weights /= weights.sum()

    assert weights.shape[0] == 3

    XX = X
    new_X = []
    ix = len(X)
    NN = N
    TT = []

    # TODO: vectorize
    for t in T:
        ia, ib, ic = t
        a, b, c = X[ia], X[ib], X[ic]

        x = weights[0] * a + weights[1] * b + weights[2] * c
        new_X.append(x)

        TT.append([ia, ib, ix])
        TT.append([ib, ic, ix])
        TT.append([ic, ia, ix])

        # Linear interpolation for the normals (if no interpolator is given)
        if (N is not None) and (normal_at is None):
            na, nb, nc = N[ia], N[ib], N[ic]
            n = weights[0] * na + weights[1] * nb + weights[2] * nc
            n /= np.linalg.norm(n)
            NN = np.append(NN, [n], axis=0)

        ix = ix + 1

    # Add new vertices
    new_X = np.array(new_X)
    XX = np.append(XX, new_X, axis=0)

    # Vectorize normal computation (since it can be quite long)
    if normal_at is not None:
        new_N = normal_at(new_X)
        NN = np.append(NN, new_N, axis=0)

    TT = np.array(TT)

    if N is None:
        return XX, TT
    else:
        return XX, TT, NN

# Concatenate a list of meshes (X, T)
def concat_meshes(meshes):
    X = []
    T = []

    l = 0
    for mesh in meshes:
        Xi, Ti = mesh
        X.append(Xi)
        Ti += l
        T.append(Ti)
        l += len(Xi)

    X = np.asarray(X)
    X = np.concatenate(X, axis=0)

    T = np.array(T)
    T = np.concatenate(T, axis=0)

    return X, T

# Finite differences for a vector function 'f'
def finite_differences(f, psi, eps=1e-8):
    N = len(psi)
    DA = np.zeros((N , N))

    for i in range(N):
        ppsi = psi.copy()
        ppsi[i] += eps

        mpsi = psi.copy()
        mpsi[i] -= eps

        Ap = f(ppsi)
        # A = f(psi)
        Am = f(mpsi)

        DA[:, i] = (Ap - Am) / (2 * eps)
        # DA[:, i] = (Ap - A) / eps

    return DA

# flatten a list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# Prepend zeros to a number to have 'zeros' digits
def int_to_str(i, zeros=2):
    str_i = str(i)
    if len(str_i) < zeros:
        str_i = "0" * (zeros - len(str_i)) + str_i
    return str_i

# Time a portion of code
import time
class timeit(object):
    def __init__(self, timer_name=None):
        self.timer_name = timer_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.running_time = self.end - self.start
        if self.timer_name is None:
            print("Running time = {}s".format(self.running_time))
        else:
            print("Running time of {} = {}s".format(self.timer_name, self.running_time))
# Count the number of times a function is called
class CallInfo(object):
    def __init__(self):
        self.raw_data = []
        self.calls = 0
    def append(self, caller):
        self.raw_data.append({ "filename": caller.filename,
                               "lineno"  : caller.lineno,
                               "function": caller.function})
        self.calls += 1
    def stats(self):
        d = {}
        for data in self.raw_data:
            k = "{}:{}".format(data["filename"], data["lineno"])
            if k not in d.keys():
                d[k] = 0
            d[k] += 1
        return d
from functools import wraps
from inspect import getframeinfo, stack
def counted(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        wrapped.calls.append(caller)
        return f(*args, **kwargs)
    wrapped.calls = CallInfo()
    return wrapped
