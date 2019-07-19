from . import Backend

import sdot.backend_laguerre as laguerre
from sdot.core.common import counted
import numpy as np

class Density_2(laguerre.MA):
    def __init__(self, X, T, V=None, cache=True):
        self.vertices = X
        self.triangles = T
        if V is None:
            self.values = np.ones_like(T)
        elif (len(V.shape) == 1) or (len(V.shape) == 2 and V.shape[1] == 1):
            self.values = np.zeros_like(T)
            for it, t in enumerate(T):
                self.values[it] = V[t]
        else:
            self.values = V
        self.cache = cache

        super().__init__(X, T)
        self.set_values(self.values)

        if self.cache:
            self.res = {}

    def set_values(self, V):
        super().with_density(V)

    @counted
    def kantorovich(self, Y, nu, psi=None):
        N = len(Y)
        if psi is None:
            psi = np.zeros(N)

        Y = np.concatenate((Y, self.__bbox))
        psibbox = np.zeros(len(self.__bbox))
        psi = np.concatenate((psi, psibbox))

        # TODO; why?
        from sdot.core.common import add_noise
        psi = add_noise(psi, 1e-8)
        res = super().compute(Y, psi)

        A, DA, C, T = res[0], res[1], res[2], res[3]
        C = C[:N]

        # TODO: why?
        DA = -DA

        if self.cache:
            self.res = { "A": A, "DA": DA, "C": C, "T": T }

        return A, DA, C, T

    def centroids(self, Y, psi):
        _, _, C, _ = self.kantorovich(Y, None, psi)
        return C

    __bbox_side = 10
    __bbox = np.array([[+__bbox_side, +__bbox_side, +__bbox_side],
                       [+__bbox_side, +__bbox_side, -__bbox_side],
                       [+__bbox_side, -__bbox_side, +__bbox_side],
                       [+__bbox_side, -__bbox_side, -__bbox_side],
                       [-__bbox_side, +__bbox_side, +__bbox_side],
                       [-__bbox_side, +__bbox_side, -__bbox_side],
                       [-__bbox_side, -__bbox_side, +__bbox_side],
                       [-__bbox_side, -__bbox_side, -__bbox_side]])

class LaguerreBackend(Backend):
    name = "laguerre"

    available_settings = { "Plane"   : Density_2,
                           "Surface" : Density_2,
                           "3D":       None
                          }
