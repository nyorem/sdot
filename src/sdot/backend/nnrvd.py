from . import Backend

from sdot.core.constants import DEBUG, VERBOSE, PROFILE
from sdot.core.common import counted
import sdot.backend_nnrvd as nnrvd
import numpy as np

# Source density supported on a triangulated surface
class Density_2(nnrvd.Density_2):
    def __init__(self, X, T, V=None, cache=True):
        self._dim = 2

        self.vertices = X
        self.triangles = T
        if V is None:
            V = np.repeat(1.0, len(self.vertices))
        self.values = V
        self.values /= self.values.max()
        self.cache = cache
        if self.cache:
            # W_2^2  / Kantorovich potential / Areas / Derivatives / Centroids / Triangulation
            self.res = {} # store results whenever self.kantorovich is called
        super().__init__(X, T, V)

    # TODO: remove loop
    def mass(self):
        from sdot.core.common import integrate_centroid
        A = 0.0

        for t in self.triangles:
            ia, ib, ic = t[0], t[1], t[2]
            a, b, c = self.vertices[ia], self.vertices[ib], self.vertices[ic]
            va, vb, vc = self.values[ia], self.values[ib], self.values[ic]
            V = np.array([va, vb, vc])
            A += integrate_centroid(a, b, c, V)

        return A

    def set_values(self, values):
        super().set_values(values)

        self.values = values

    # Kantorovich potential / Areas / Derivatives / Centroids / Triangulation (dual)
    @counted
    def kantorovich(self, Y, nu, psi=None):
        if PROFILE:
            start = time.time()

        if psi is None:
            psi = np.zeros(len(Y))

        dist, A, DA, C, T, TT, _ = nnrvd.monge_ampere(self, Y, psi)
        f = dist + np.dot(psi, A - nu)

        if self.cache:
            self.res = { 'dist': dist, 'f': f, 'A': A, 'DA': DA, 'C': C, 'T': T, 'TT': TT }

        if PROFILE:
            from sdot.core.common import eprint
            end = time.time()
            eprint("Computation of G: {}".format(end - start))

        return A, DA, f, C, T

    # Export the Laguerre diagram to a obj mesh
    # will export /tmp/mesh.obj and /tmp/mesh.mtl
    # How to see colors in MeshLab? 'Face' attribute in 'Color'
    def export_laguerre_cells(self, Y, psi=None, basename="/tmp/mesh"):
    # def export_laguerre_cells(self, Y, nu, psi=None, basename="/tmp/mesh"):
        if psi is None:
            psi = np.zeros(len(Y))
        nnrvd.export_laguerre_cells(self, Y, psi, basename)

    def inside_support(self, p):
        from sdot.core.common import inside_triangle
        for it in range(len(self.triangles)):
            t = self.triangles[it]
            a, b, c = self.vertices[t]
            if inside_triangle(a, b, c, p):
                return it
        return None

    def plot(self, fig=None, as_img=False, colorbar=False):
        import matplotlib.pyplot as plt
        fig = fig or plt.figure()

        if as_img:
            # Only work with square densities
            root = math.sqrt(self.values.size)
            assert int(root + 0.5) ** 2 == self.values.size, "Plot as image only works with square densities"
            n = int(root)

            img = self.values.reshape(n, n)
            plt.imshow(img, interpolation="none")
        else:
            from sdot.core.common import plot_2d_tri_func, plot_2d_cloud, plot_tri, plot_cloud, is_2d

            if is_2d(self.vertices):
                plot_2d_cloud(self.vertices, cmap=self.values, colorbar=colorbar, fig=fig)
                # plot_2d_tri_func(self.vertices, self.triangles, self.values, fig=fig)
            else:
                plot_cloud(self.vertices, cmap=self.values, colorbar=colorbar, fig=fig)
                # plot_tri(self.vertices, self.triangles fig=fig)

    @classmethod
    def from_function(cls, X, T, mu, remove_black_areas=False, **kwargs):
        V = np.apply_along_axis(mu, 1, X)
        if remove_black_areas:
            black_areas = np.where(V == 0)[0]
            V[black_areas] = 1 # between 0 and 255
        return cls(X, T, V=V, **kwargs)

    @classmethod
    def from_texture(cls, X, T, img, uv_map, **kwargs):
        from sdot.core.common import bilinear_interpolate
        w, h = img.shape
        def mu(p):
            u, v = uv_map(p)
            # u and v are between 0 and 1
            val = bilinear_interpolate(img, u * w, v *h)
            return val
        return cls.from_function(X, T, mu, remove_black_areas=True, **kwargs)

    @classmethod
    def directional(cls, X, T, dir=[0, 0, 1], **kwargs):
        dir = np.array(dir)
        def mu(p):
            u = p / np.linalg.norm(p)
            return max(np.dot(u, dir), 0.0)
        return cls.from_function(X, T, mu, **kwargs)

    @classmethod
    def from_grid(cls, X, V, **kwargs):
        import matplotlib.tri as tri
        T = tri.Triangulation(X[:, 0], X[:, 1])
        T = T.triangles
        return cls(X, T, V, **kwargs)

    @classmethod
    def from_ies(cls, fname, dims, size=None, bbox=None, attenuation=1.0, **kwargs):
        from .density import ies_to_function
        if bbox is None:
            bbox = [ -1.0, 1.0, -1.0, 1.0 ]
        if size is None:
            size = np.linalg.norm([bbox[0], bbox[2]])
        mu = ies_to_function(fname, attenuation=attenuation, size=size)

        # Uniform grid
        if isinstance(dims, tuple):
            nx, ny = dims
            nx, ny = int(nx), int(ny)
        elif isinstance(dims, int):
            nx, ny = dims, dims
        else:
            raise RuntimeError("dims={} parameter is not a tuple nor an int".format(dims))

        xs, ys = np.meshgrid(np.linspace(bbox[0], bbox[1], nx),
                             np.linspace(bbox[2], bbox[3], ny))
        xs = xs.flatten()
        ys = ys.flatten()
        X = np.vstack((xs, ys)).T

        V = np.apply_along_axis(mu, 1, X)

        return cls.from_grid(X, V, **kwargs)

    # The values file has one value per line and per vertex
    # in the SAME order as in the 'X' array.
    @classmethod
    def from_file(cls, X, T, V_fname, **kwargs):
        V = np.loadtxt(V_fname)
        return cls(X, T, V=V, **kwargs)

    # TODO: check
    # - generalize for N densities
    # - generalize for mixed (2D / 3D) densities
    # Hyp: \supp(\mu) \subseteq \supp(\nu)
    # Return mu_t : t ->  Density_2 such that mu_t(t) = t * nu + (1 - t) mu
    # Support = \supp(\nu)
    @classmethod
    def from_convex_combination(cls, mu, nu):
        # Pre compute values of mu on the biggest support (here the support of 'nu')
        mu_values = np.zeros_like(nu.values)

        for i in range(len(nu.vertices)):
            p = nu.vertices[i]
            it = mu.inside_support(p)
            mu_val = 0.0
            if it is not None:
                mu_val = mu.eval(it, p)
            mu_values[i] = mu_val

        def mu_t(t):
            V_t = t * nu.values + (1 - t) * mu_values
            return cls(nu.vertices, nu.triangles, V=V_t, cache=mu.cache)

        return mu_t

class NNRVDBackend(Backend):
    name = "nnrvd"

    available_settings = { "Plane"   : Density_2,
                           "Surface" : Density_2,
                           "3D"      : None
                          }
