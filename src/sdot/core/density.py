import numpy as np
import scipy as sp
import scipy.misc # TODO: replace with imageio
from .constants import DEBUG, VERBOSE

############
## Source ##
############

# Transform to apply before computing Kantorovich potential and such
class Transform(object):
    is_directional = False
    is_point    = False

    def prepare_input_directions(self, Z):
        return Z

    def get_initial_weights(self, Y):
        psi0 = np.zeros(Y.shape[0])
        return psi0

    def transform_points(self, Y, psi=None):
        if psi is None:
            psi = np.zeros(len(Y))
        return Y, psi

    def transform_derivatives(self, DA):
        return DA

    def build_component(self, C, Y, psi):
        raise NotImplementedError()

# Convex combination of source measures:
# \sum_{i=1}^N * w_i mu_i where \sum_i w_i = 1
class Interpolated_density(object):
    def __init__(self, densities, weights=None):
        assert len(densities) >= 2

        self.densities = densities
        N = len(densities)

        if weights is None:
            self._weights = np.repeat(1 / N, N)
        else:
            assert len(weights) == N
            self._weights = np.array(weights)
        self._weights /= self._weights.sum()

        self.cache = any([ density.cache for density in densities ])

        if self.cache:
            self.res = {}

    def get_weights(self):
        return self._weights

    def set_weights(self, w):
        assert len(self._weights) == len(w)
        w = np.array(w)
        self._weights = w / w.sum()

    weights = property(get_weights, set_weights)

    def kantorovich(self, Y, nu, psi=None):
        N = len(Y)

        A = np.zeros(N)
        DA = sp.sparse.csc_matrix((N, N))
        C = np.zeros_like(Y)
        have_centroids = False

        for i in range(len(self.densities)):
            # if math.fabs(self._weights[i]) < 1e-8:
            #     continue

            density = self.densities[i]
            density.kantorovich(Y, nu, psi)

            A += self._weights[i] * density.res["A"]
            DA += self._weights[i] * density.res["DA"]

            if "C" in density.res.keys():
                have_centroids = True
                C += self._weights[i] * density.res["C"]

        if not have_centroids:
            C = None

        if self.cache:
            self.res = { "A" : A, "DA": DA, "C": C, "T": None }

        return A, DA

    def mass(self):
        m = 0
        for i in range(len(self.densities)):
            density = self.densities[i]
            m += self._weights[i] * density.mass()
        return m
