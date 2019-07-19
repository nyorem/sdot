#! /usr/bin/env python3

import numpy as np
import os
import scipy as sp
import sys

from sdot.core.common import plot_2d_tri, plot_2d_cloud, plot_tri, plot_cloud, \
    count_zeros, random_in_triangulation, transform_box, vprint, timeit, is_2d
import sdot.core.inout as io
from sdot.optimal_transport.discrete import sinkhorn, sinkhorn_stabilized, \
    sinkhorn_epsilon_scaling, cost_matrix, sinkhorn_to_kantorovich
from sdot.optimal_transport.discrete.initialization import init_sinkhorn, \
    sinkhorn_rescale_transform_weights

# np.set_printoptions(suppress=True)

verbose = True
compute_plan = False

# Parameters
eps = 1e-8
maxit = None
maxit_interp = None
reg = 1e-1
with_torch = os.environ.get("WITH_TORCH") == "1"
if with_torch:
    from sdot.optimal_transport.discrete import torch_cost_matrix, torch_sinkhorn, torch_available
    assert torch_available
    cost_matrix = torch_cost_matrix

sinkhorn_method = "normal"
# sinkhorn_method = "stabilized"
# sinkhorn_method = "scaling"

def launch_sinkhorn(mu_flat, nu_flat, M, plan=False, verbose=False, **kwargs):
    methods = { "normal"    : torch_sinkhorn if with_torch else sinkhorn,
                "stabilized": sinkhorn_stabilized,
                "scaling"   : sinkhorn_epsilon_scaling
              }

    launcher = methods[sinkhorn_method]

    return launcher(mu_flat, nu_flat, M, verbose=verbose, reg=reg, log=True, plan=plan,
                    eps=eps, maxit=maxit, **kwargs)

discrete_init = { "normal"  : True,
                  "local"   : True,
                  "rescale" : True,
                  "interp"  : True,
                 }

def sinkhorn_print_costs(gamma, M, reg, mu, nu, u, v):
    assert gamma is not None

    from sdot.optimal_transport.discrete import sinkhorn_primal_cost, sinkhorn_dual_cost

    alpha, beta = sinkhorn_to_kantorovich(u, v, reg)

    primal = sinkhorn_primal_cost(gamma, M, reg)
    dual   = sinkhorn_dual_cost(gamma, mu, nu, alpha, beta, reg)

    print("primal={}, dual={}".format(primal, dual))

def sinkhorn_display_plan(gamma, X, Y, fig=None, tol=0.6):
    fig = fig or plt.figure()
    ax = plt.gca()

    plot_2d_cloud(X, cmap="r", fig=fig)
    plot_2d_cloud(Y, cmap="b", fig=fig)

    if gamma is not None:
        gamma /= gamma.max()

        from matplotlib import collections  as mc

        lin, col = np.nonzero(gamma > tol)
        src, dst = X[lin], Y[col]
        src = src[:, 0:2]
        dst = dst[:, 0:2]

        ps = []
        colors = []
        for i in range(len(src)):
            ps.append([src[i], dst[i]])
            colors.append([0, 1, 0, 1])

        lc = mc.LineCollection(ps, colors=colors)
        ax.add_collection(lc)

    plt.show()

def sinkhorn_display_interp(gamma, X, Y, step=20, fig=None, tol=0.2, dim=2):
    if gamma is None:
        return

    assert dim in [2, 3]

    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D

    fig = fig or plt.figure()

    gamma /= gamma.max()
    lin, col = np.nonzero(gamma > tol)
    src, dst = X[lin], Y[col]
    vprint("{} points".format(len(src)))

    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")

    src = src[:, 0:dim]
    dst = dst[:, 0:dim]

    if dim == 2:
        sc = plt.scatter(dst[:, 0], dst[:, 1], c="g")
        title = plt.title("")
    elif dim == 3:
        sc = ax.scatter(dst[:, 0], dst[:, 1], dst[:, 2], c="g")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        title = ax.set_title("")

    xmin, xmax = np.min(src[:, 0]), np.max(src[:, 0])
    ymin, ymax = np.min(src[:, 1]), np.max(src[:, 1])
    alpha = 0.1 * max(xmax - xmin, ymax - ymin)
    plt.xlim(xmin - alpha, xmax + alpha)
    plt.ylim(ymin - alpha, ymax + alpha)

    def animate(i):
        t = i / step
        ps = t * src + (1 - t) * dst
        if dim == 2:
            sc.set_offsets(ps)
        elif dim == 3:
            sc._offsets3d = (tuple(ps[:, 0]), tuple(ps[:, 1]), ps[:, 2])
        title.set_text("t={}".format(t))
        return sc,

    ani = animation.FuncAnimation(fig, animate, np.arange(step+1),
                                  interval=150, repeat=True)

    plt.show()

if __name__ == "__main__":
    vprint("Using sinkhorn_method={}, reg={}, torch={}".format(sinkhorn_method, reg, with_torch))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    assert(len(sys.argv) in [1, 3, 4]), "Usage: {} source.off target.[off, cloud] [nY]".format(sys.argv[0])

    running_times = {}
    if len(sys.argv) in [3, 4]:
        # Load source as off file and discretize it / target as a cloud
        source_filename, target_filename = sys.argv[1:3]
        source_ext = os.path.splitext(source_filename)[1]

        # Target
        target_ext = os.path.splitext(target_filename)[1]
        if target_ext in [".xyz", ".txt", ".cloud"]:
            P_nu = np.loadtxt(target_filename)
        elif target_ext == ".off":
            assert len(sys.argv) == 4
            nY = int(sys.argv[3])

            X_Y, T_Y = io.read_off(target_filename, ignore_prefix=True)
            P_nu, _ = random_in_triangulation(X_Y, T_Y, nY)

        nu_flat = np.repeat(1.0, len(P_nu))
        nu = None
        N = len(P_nu) # N is the total number of points

        # Source
        if source_ext == ".off":
            X, T_X = io.read_off(source_filename, ignore_prefix=True)

            ## Smaller source
            box_origin = [ -1.0, 1.0, -1.0, 1.0 ]
            # box_source = [ -1.0 , 0.0, -1.0, 1.0] # TODO: change this
            # transform_box_function = transform_box(box_origin, box_source)
            # X[:, 0:2] = np.apply_along_axis(transform_box_function, 1, X)

            P_mu, _ = random_in_triangulation(X, T_X, N)
        elif source_ext in [".xyz", ".txt", ".cloud"]:
            P_mu = np.loadtxt(source_filename)

        assert P_mu.shape[1] == P_nu.shape[1] # same dimension

        mu_flat = np.repeat(1.0, len(P_mu))
        mu = None

        # Save source cloud
        from decimal import Decimal
        results_fname = "{},reg={:.0e}".format(os.path.splitext(os.path.basename(source_filename))[0],
                                               Decimal(reg))
        results_fname = "results/discrete/{}/".format(results_fname)
        np.savetxt(os.path.join(results_fname, "source.xyz"), P_mu)
        np.savetxt(os.path.join(results_fname, "target.xyz"), P_nu)
    else:
        # 2 gaussians
        # height of a gaussian centered at mu with variance sigma
        def gaussian(X, Y, mu=[0.0, 0.0], sigma=1.0):
            return np.exp(-((X-mu[0])**2+(Y-mu[1])**2)/2.0*sigma**2)

        # Discrete
        N = 30 # N^2 is the total number of points

        # Source
        side_mu = 4
        bbox = [0, side_mu, -side_mu, side_mu]
        X_mu, Y_mu = np.meshgrid(np.linspace(bbox[0], bbox[1], num=N),
                                 np.linspace(bbox[2], bbox[3], num=N))
        P_mu = np.vstack([X_mu.ravel(), Y_mu.ravel()]).T
        mu = gaussian(X_mu, Y_mu, mu=[0.5 * side_mu, 0], sigma=1.0)

        # Target
        side_nu = 4
        bbox = [-side_nu, 0, -side_nu, side_nu]
        X_nu, Y_nu = np.meshgrid(np.linspace(bbox[0], bbox[1], num=N),
                                 np.linspace(bbox[2], bbox[3], num=N))
        P_nu = np.vstack([X_nu.ravel(), Y_nu.ravel()]).T
        nu = gaussian(X_nu, Y_nu, mu=[-0.5 * side_nu, 0], sigma=1.0)

        mu_flat = mu.flatten()
        nu_flat = nu.flatten()

    X_discrete = P_mu
    Y_discrete = P_nu
    M = cost_matrix(X_discrete, Y_discrete)

    mu_flat /= mu_flat.sum()
    nu_flat /= nu_flat.sum()

    fig = plt.figure()
    if len(sys.argv) == 1:
        ax = fig.add_subplot("111", projection="3d")
        ax.plot_surface(X_mu, Y_mu, mu, color="r")
        ax.plot_surface(X_nu, Y_nu, nu, color="b")

        # import matplotlib as mpl
        # legend1 = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker='o')
        # legend2 = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker='o')
        # ax.legend([legend1, legend2], ["Source", 'Target'], numpoints = 1)
    elif len(sys.argv) in [3, 4]:
        if is_2d(P_mu):
            plot_2d_cloud(P_mu, fig=fig, cmap="r", labels=["source"])
            plot_2d_cloud(P_nu, fig=fig, cmap="b", labels=["target"])
        else:
            plot_cloud(P_mu, fig=fig, cmap="r", labels=["source"])
            plot_cloud(P_nu, fig=fig, cmap="b", labels=["target"])
    ax = plt.gca()
    ax.set_aspect("equal")
    # plt.legend()
    # plt.show()

    # plt.imshow(M)
    # plt.show()

    # Normal
    if discrete_init["normal"]:
        with timeit("NORMAL") as timer:
            gamma, log = launch_sinkhorn(mu_flat, nu_flat, M,  u0=None, v0=None, plan=compute_plan,
                                         verbose=verbose)
            # gamma /= gamma.max()

            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(gamma, interpolation="bilinear")
            # plt.subplot(122)
            # plt.imshow(M, interpolation="bilinear")
            # plt.show()

            u, v = log["u"], log["v"]
            if compute_plan:
                sinkhorn_print_costs(gamma, M, reg, mu_flat, nu_flat, u, v)
            # sinkhorn_display_plan(gamma, P_mu, P_nu)

            # P_mu = np.hstack((P_mu, mu_flat.reshape(-1, 1)))
            # P_nu = np.hstack((P_nu, nu_flat.reshape(-1, 1)))
            # sinkhorn_display_interp(gamma, P_mu, P_nu, dim=2)
        running_times["normal"] = timer.running_time

    # local
    if discrete_init["local"]:
        with timeit("LOCAL") as timer:
            u0, v0 = init_sinkhorn(X_discrete, mu, Y_discrete, nu, reg, method="local", plan=False)
            gamma, log = launch_sinkhorn(mu_flat, nu_flat, M, u0=u0, v0=v0, plan=compute_plan, verbose=verbose)
            # gamma /= gamma.max()

            u, v = log["u"], log["v"]
            if compute_plan:
                sinkhorn_print_costs(gamma, M, reg, mu_flat, nu_flat, u, v)
        running_times["local"] = timer.running_time

    # Rescale
    if discrete_init["rescale"]:
        with timeit("RESCALE") as timer:
            lam, t = init_sinkhorn(X_discrete, mu_flat, Y_discrete, nu_flat, reg, method="rescale",
                                   plan=False)
            Z_discrete = lam * Y_discrete + t

            M_rescale = cost_matrix(X_discrete, Z_discrete)

            # fig = plt.figure()
            # plot_2d_cloud(X_discrete, cmap="b", fig=fig, labels=["source"])
            # plot_2d_cloud(Y_discrete, cmap="r", fig=fig, labels=["target"])
            # plot_2d_cloud(Z_discrete, cmap="g", fig=fig, labels=["rescaled"])
            # plt.legend()
            # plt.show()

            print("BEFORE RESCALING: OT between X and Y with no initial weights")
            gamma, log = launch_sinkhorn(mu_flat, nu_flat, M, u0=None, v0=None, verbose=verbose)

            # Compute OT between X and Z (rescaled)
            gamma, log = launch_sinkhorn(mu_flat, nu_flat, M_rescale, u0=None, v0=None, verbose=False)
            # gamma /= gamma.max()

            u_rescale, v_rescale = log["u"], log["v"]
            u_tilde = sinkhorn_rescale_transform_weights(lam, t, Y_discrete, u_rescale, reg)
            v_tilde = sinkhorn_rescale_transform_weights(lam, t, Y_discrete, v_rescale, reg)

            # Initialize OT between X and Y (non rescaled) with previous weights
            print("AFTER RESCALING: OT between X and Y with new intial weights")
            with timeit("rescale_final") as timer_final:
                gamma, log = launch_sinkhorn(mu_flat, nu_flat, M, u0=u_tilde, v0=v_tilde, verbose=verbose,
                                             plan=compute_plan)
                # gamma /= gamma.max()
            running_times["rescale_final"] = timer_final.running_time

            u, v = log["u"], log["v"]
            if compute_plan:
                sinkhorn_print_costs(gamma, M, reg, mu_flat, nu_flat, u, v)
        running_times["rescale"] = timer.running_time

    # Interp
    if discrete_init["interp"]:
        with timeit("INTERP") as timer:
            u, v = init_sinkhorn(X_discrete, mu_flat, Y_discrete, nu_flat, reg, method="interp",
                                 maxit=maxit, eps=eps, maxit_interp=maxit_interp, plan=False)

            # Solve real OT with the values for u and v prevously computed
            print("AFTER INTERP")
            with timeit("interp_final") as timer_final:
                gamma, log = launch_sinkhorn(mu_flat, nu_flat, M, u0=u, v0=v, verbose=verbose, plan=compute_plan)
                # gamma /= gamma.max()
            running_times["interp_final"] = timer_final.running_time

            # mu_flat = mu_flat[:len(u)]
            # nu_flat = nu_flat[:len(v)]

            u, v = log["u"], log["v"]
            if compute_plan:
                sinkhorn_print_costs(gamma, M, reg, mu_flat, nu_flat, u, v)
            # sinkhorn_display_interp(gamma, P_mu, P_nu)
        running_times["interp"] = timer.running_time

        # gamma, log = sinkhorn_stabilized(mu, nu, M, verbose=True, reg=reg, log=True)
        # gamma, log = sinkhorn_epsilon_scaling(mu, nu, M, verbose=True, reg=reg, log=True)

        # Cost / Transport matrix
        # fig = plt.figure(1)
        # plt.subplot(211)
        # plt.title("Cost / Transport matrix")
        # plt.imshow(M)
        # plt.subplot(212)
        # plt.imshow(gamma)
        # plt.show()

    print(running_times)
