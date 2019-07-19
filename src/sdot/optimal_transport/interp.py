import numpy as np
import time

from sdot.core.common import vprint, count_zeros
from . import optimal_transport_3

# mu = 3D / nu = density on surface, (Y, dst) = target
def optimal_transport_surface(mu, nu, Y, dst, eps=1e-8, maxit=500, maxit_interp=20, verbose=False,
                              psi0=None, update_method=None, save_results=False,
                              adaptive_t=False):
    from sdot.core.density import Interpolated_density

    # for debugging purposes
    if save_results:
        from sdot.core.common import int_to_str
        zeros = len(str(maxit_interp))

    if update_method is None:
        update_method = lambda t, maxit_interp: t / 2

    # Normalization of the source and target measures
    nu_values = np.repeat(1.0 / nu.mass(), len(nu.vertices))
    nu.set_values(nu_values)

    mu_mass = mu.mass()
    mu_values = lambda x: 1 / mu_mass
    mu.set_values(mu_values)

    dst /= dst.sum()

    # Interpolation measure
    interp = Interpolated_density([mu, nu], weights=[1.0, 0.0])

    # Do the interpolation
    start = time.time()

    psi = psi0
    t = 1.0
    zz = len(Y)
    interp.weights = [t, 1 - t]
    it = 0
    while zz != 0:
        vprint("it_interp {}: t={}".format(it + 1, t))

        dst *= interp.mass() / dst.sum()

        psi = optimal_transport_3(interp, Y, dst, psi0=psi, eps=eps, maxit=maxit)

        if adaptive_t:
            interp.kantorovich(Y, dst, psi=psi)
            # G_t  = interp / G_mu = interp.densities[1] / G_lambda = interp.densities[0]
            diff = interp.densities[1].res["A"] - interp.densities[0].res["A"]
            DG = interp.res["DA"]
            from . import solve_graph_laplacian
            delta_psi = solve_graph_laplacian(DG, diff)

            s = min(0.1, t)
            while True:
                print("s={}".format(s))
                psi_tilde = psi - s * delta_psi
                interp.weigths = [ t - s, 1 - (t - s) ]
                A_tilde = interp.kantorovich(Y, dst, psi=psi_tilde)[0]
                print("min(A)={}".format(np.min(A_tilde)))
                if np.min(A_tilde) > 0:
                    psi = psi_tilde
                    t -= s
                    break
                s /= 2
        else:
            t = update_method(t, maxit_interp)

        if it > 0:
            A = nu.kantorovich(Y, dst, psi=psi)[0]
            if save_results:
                nu.export_laguerre_cells(Y, psi, "/tmp/interp3d_nu" + int_to_str(it, zeros=zeros))
            zz = count_zeros(A)
            vprint("{} empty cells".format(zz))

            if zz == 0:
                break

        if maxit_interp is not None and it >= maxit_interp - 1:
            break

        interp.weights = [t, 1 - t]
        it += 1

    # it += 1
    # dst *= nu.mass() / dst.sum()
    # psi = optimal_transport_3(nu, Y, dst, psi0=psi, eps=eps, maxit=maxit)
    # if save_results:
    #     nu.export_laguerre_cells(Y, psi, "/tmp/interp3d_nu" + int_to_str(it, zeros=zeros))

    end = time.time()

    if verbose:
        print("Running time = {}s".format(end - start))

    return psi
