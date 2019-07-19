import sdot.backend_geogram as geogram

from . import Backend
from sdot.core.common import counted
import numpy as np

geogram.geogram_init(False)

class Density_Plane(geogram.Density_1):
    def __init__(self, X, T, V=None, cache=True,):
        from sdot.core.common import is_2d
        assert is_2d(X), "This class can only be used for measures supported on 2D domains"

        self.vertices = X
        self.triangles = T
        if V is not None:
            if isinstance(V, np.ndarray):
                # TODO
                raise NotImplementedError
            else:
                assert callable(V)
                self.values_func = V
        else:
            self.values_func = lambda x: 1.0

        self.cache = cache
        if self.cache:
            self.res = {}

        super().__init__(self.vertices, self.triangles, self.values_func)

    @counted
    def kantorovich(self, Y, nu, psi=None):
        from sdot.core.common import is_2d
        assert is_2d(Y), "This class can only be used for measures supported on 2D domains"

        if psi is None:
            psi = np.zeros(len(Y))

        A, DA = super().kantorovich(Y, nu, psi)

        if self.cache:
            self.res = { "A": A, "DA": DA }

        return A, DA

    def export_laguerre_cells(self, Y, nu, psi, fname):
        self.kantorovich(Y, nu, psi=psi)
        self.save_diagram(fname)

    def centroids(self, Y, psi=None):
        from sdot.core.common import is_2d
        assert is_2d(Y), "This class can only be used for measures supported on 2D domains"

        if psi is None:
            psi = np.zeros(len(Y))

        C = super().centroids(Y, psi)

        if self.cache:
            self.res["C"] = C

        return C

    def plot(self, fig=None, **kwargs):
        from sdot.core.common import plot_2d_tri
        fig = fig or plt.figure()
        plot_2d_tri(self.vertices, self.triangles, fig=fig, **kwargs)

    def optimal_transport(self, Y, nu, psi0=None, eps=1e-8, verbose=False, maxit=100):
        from sdot.core.common import is_2d
        assert is_2d(Y), "This class can only be used for measures supported on 2D domains"

        if psi0 is None:
            psi0 = np.zeros(len(Y))

        psi = super().optimal_transport(Y, nu, psi0, eps, verbose, maxit)

        return psi

class Density_Surface(geogram.Density_2):
    def __init__(self, X, T, V=None, cache=True):
        self.vertices = X
        self.triangles = T
        if V is not None:
            assert callable(V)
            self.values_func = V
        else:
            self.values_func = lambda x: 1.0

        self.cache = cache
        if self.cache:
            self.res = {}

        super().__init__(self.vertices, self.triangles, self.values_func)

    @counted
    def kantorovich(self, Y, nu, psi=None):
        if psi is None:
            psi = np.zeros(len(Y))

        A, DA = super().kantorovich(Y, nu, psi)

        if self.cache:
            self.res = { "A": A, "DA": DA }

        return A, DA

    def export_laguerre_cells(self, Y, nu, psi, fname):
        self.kantorovich(Y, nu, psi=psi)
        self.save_diagram(fname)

    def centroids(self, Y, psi=None):
        if psi is None:
            psi = np.zeros(len(Y))

        C = super().centroids(Y, psi)

        if self.cache:
            self.res["C"] = C

        return C

    def plot(self, fig=None, **kwargs):
        from sdot.core.common import plot_tri
        fig = fig or plt.figure()
        plot_tri(self.vertices, self.triangles, fig=fig, **kwargs)

    def optimal_transport(self, Y, nu, psi0=None, eps=1e-8, verbose=False, maxit=100):
        if psi0 is None:
            psi0 = np.zeros(len(Y))

        psi = super().optimal_transport(Y, nu, psi0, eps, verbose, maxit)

        return psi

class Density_3(geogram.Density_3):
    def __init__(self, X, T, V=None, cache=True):
        self.vertices = X
        self.triangles = T
        if V is not None:
            assert callable(V)
            self.values_func = V
        else:
            self.values_func = lambda x: 1.0

        self.cache = cache
        if self.cache:
            self.res = {}

        super().__init__(self.vertices, self.triangles, self.values_func)

    @counted
    def kantorovich(self, Y, nu, psi=None):
        if psi is None:
            psi = np.zeros(len(Y))

        A, DA = super().kantorovich(Y, nu, psi)

        if self.cache:
            self.res = { "A": A, "DA": DA }

        return A, DA

    # fname must be a tet format (tet, tet6)
    def export_laguerre_cells(self, Y, nu, psi, fname):
        ext = os.path.splitext(fname)[1]
        assert ext in [".tet", ".tet6"], "Extension of the output file must be .tet or .tet6"

        self.kantorovich(Y, nu, psi=psi)
        self.save_diagram(fname)

    def centroids(self, Y, psi=None):
        if psi is None:
            psi = np.zeros(len(Y))

        C = super().centroids(Y, psi)

        if self.cache:
            self.res["C"] = C

        return C

    def plot(self, fig=None, **kwargs):
        from sdot.core.common import plot_tri
        fig = fig or plt.figure()
        plot_tri(self.vertices, self.triangles, fig=fig, **kwargs)

    def optimal_transport(self, Y, nu, psi0=None, eps=1e-8, verbose=False, maxit=100):
        if psi0 is None:
            psi0 = np.zeros(len(Y))

        psi = super().optimal_transport(Y, nu, psi0, eps, verbose, maxit)

        return psi

class GeogramBackend(Backend):
    name = "geogram"

    available_settings = { "Plane"   : Density_Plane,
                           "Surface" : Density_Surface,
                           "3D"      : Density_3
                         }
