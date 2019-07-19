#! /usr/bin/env python3

# Usage: icp.py mesh [source.[xyz, txt, cloud, off, noff, coff]]
# Output:
# - initial.xyz is the point set to be registered if the second argument is not a cloud
# - icp_euclidean.xyz is the registered point set using Euclidean ICP
# - icp_ot.xyz is the registered point set using OT-ICP
# Examples:
# - python examples/misc/icp.py examples/assets/sphere.off
# - python examples/misc/icp.py examples/assets/torus.off
# - python examples/misc/icp.py examples/assets/icosahedron.off

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import trimesh

from sdot.core.constants import CUBE_MESH
from sdot.core.common import vprint, plot_cloud, add_noise, random_in_triangulation, is_2d, plot_2d_cloud, \
    plot_2d_tri, plot_tri
from sdot.core.constants import VERBOSE
from sdot.backend import GeogramBackend, NNRVDBackend
Density_2 = NNRVDBackend().available_settings["Surface"]
Density_3 = GeogramBackend().available_settings["3D"]
import sdot.core.inout as io
from sdot.optimal_transport import optimal_transport_3, Results
from sdot.optimal_transport.initialization import init_optimal_transport_3, diam2, diam
import sdot.optimal_transport.interp as interp

# ICP parameters
maxit_icp = 20
tolerance_icp = 1e-6
init_interp = False  # whether to find initial weights with interp or local perturbation

# OT parameters
eps = 1e-8
maxit = 500
eps_init = 1e-1 # local perturbation
maxit_interp = 20 # interp

# Interpolation parameters (if used)
np.set_printoptions(suppress=True)
psi = None
X_cube, T_cube = io.read_off(os.path.join(CUBE_MESH), ignore_prefix=True)
update_method = None
# update_method = lambda t, maxit_interp: t - 1 / maxit_interp if t > 0.5 else t / 2

# Best rigid transformation (translation and rotation)
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    # Scale
    # s = diam2(A) / diam2(B)
    # s = volume_cube(bbox(A)) / volume_cube(bbox(B))
    # print("s={}".format(s))

    return T, R, t

# input: src = cloud / dst = mesh (centered at the origin)
# output: distances between nearest neighbors / indices of nearest neighbors
def nearest_neighbor_mesh(src, dst, ma=None, verbose=False):
    if ma is None:
        return dst.nearest.vertex(src) # trimesh + L2 norm
    else:
        global psi

        # Target measure
        Y = src
        nu = np.repeat(1.0, len(Y))

        # Optimal transport
        # Interp
        if init_interp:
            global X_cube
            scale = 2 * 1.1 * max(diam(ma.vertices), diam(Y))
            vprint("scale={}".format(scale))
            X_cube *= scale
            cube = Density_3(X_cube, T_cube)
            cube.set_values(lambda x: 1.0)

            psi = interp.optimal_transport_surface(cube, ma, Y, nu, psi0=None,
                                                   eps=eps, maxit=maxit, maxit_interp=maxit_interp,
                                                   adaptive_t=False,
                                                   verbose=verbose,
                                                   update_method=update_method)
            # del cube
        else:
            # Local perturbation
            psi0 = init_optimal_transport_3(ma, Y, nu, method="local", eps_init=eps_init)
            nu = mu.mass() * nu / nu.sum()
            psi = optimal_transport_3(ma, Y, nu, psi0=psi0, eps=eps, verbose=verbose)

        # Centroids
        C = ma.kantorovich(Y, nu, psi=psi)[Results.C]

        return C

# Registration between a mesh (target) and a point cloud (source)
# input: A = cloud and a  mesh
# output: T s.t. ||T*A - mesh|| is minimized
def icp_mesh(A, mesh, init_pose=None, max_iterations=20, tolerance=1e-6, verbose=False, ma=None):
    '''
    The Iterative Closest Point method
    Input:
        A: Nx3 numpy array of source 3D points
        B: Nx3 numpy array of destination 3D point
        init_pose: 4x4 homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation
        distances: Euclidean distances (errors) of the nearest neighbor
    '''

    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((4,A.shape[0]))
    src[0:3,:] = np.copy(A.T)

    # Only used in the non OT setting
    B = mesh.vertices
    dst = np.ones((4,B.shape[0]))
    dst[0:3,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        if ma is None:
            distances, indices = nearest_neighbor_mesh(src[0:3,:].T, mesh, verbose=verbose)
        else:
            neighbors = nearest_neighbor_mesh(src[0:3,:].T, mesh, ma=ma, verbose=verbose)
            distances = np.linalg.norm(neighbors - src[0:3, :].T, axis=1)

        # compute the transformation between the current source and nearest destination points
        if ma is None:
            T,_,_ = best_fit_transform(src[0:3,:].T, dst[0:3,indices].T)
        else:
            T,_,_ = best_fit_transform(src[0:3,:].T, neighbors)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if verbose:
            vprint("Iteration {}, error = {}".format(i + 1, mean_error))
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation: cloud -> new_cloud
    T,_,_ = best_fit_transform(A, src[0:3,:].T)

    return T, distances

# source: https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
# returns a 3x3 matrix
def rotation_matrix(axis, theta):
    from scipy.linalg import expm
    return expm(np.cross(np.eye(3), axis/np.linalg.norm(axis)*theta))

# returns a 4x4 matrix (homogeneous matrix)
def rotation_matrix_homogeneous(axis, theta):
    M = rotation_matrix(axis, theta)
    MM = np.zeros((4, 4))
    MM[0:3, 0:3] = M
    MM[3, 3] = 1
    return MM

# Apply an homogeneous transformation T on a point cloud.
def transform_3d_cloud(T, cloud):
    tmp_cloud = np.ones((4, cloud.shape[0]))
    tmp_cloud[0:3, :] = cloud.T
    new_cloud = np.dot(T, tmp_cloud)[0:3, :].T
    return new_cloud

def test_icp(mesh, cloud, ma=None, verbose=False, max_iterations=20, tolerance=1e-6):
    final_T, _ = icp_mesh(cloud, mesh, max_iterations=max_iterations, tolerance=tolerance, verbose=verbose, ma=ma)
    new_cloud = transform_3d_cloud(final_T, cloud)

    if ma is None:
        distances, _ = nearest_neighbor_mesh(new_cloud, mesh, verbose=verbose)
    else:
        neighbors = nearest_neighbor_mesh(new_cloud, mesh, ma=ma, verbose=verbose)
        distances = np.linalg.norm(neighbors - new_cloud, axis=1)

    final_error = np.mean(distances)

    return final_T, new_cloud, final_error

def synchronize_rotation(fig, ax1, ax2):
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
        else:
            return
        fig.canvas.draw_idle()
    return fig.canvas.mpl_connect('motion_notify_event', on_move)

if __name__ == "__main__":
    assert len(sys.argv) in [2, 3], "Usage: {} target_mesh.off source".format(sys.argv[0])

    # Target mesh
    off_file = sys.argv[1]
    assert os.path.isfile(off_file), "{} does not exist!".format(off_file)
    target_mesh_basename = os.path.splitext(os.path.basename(off_file))[0]

    # Load and center it at the origin
    mesh = trimesh.load_mesh(off_file)
    mesh.vertices -= np.mean(mesh.vertices, axis=0)

    # Object to register (cloud or mesh) on the target mesh
    # If it is a mesh, it is approximated by randomly sampling N points on it
    N = 1000
    source_filename = sys.argv[2] if len(sys.argv) >= 3 else None
    if source_filename is not None:
        assert os.path.isfile(source_filename), "{} does not exist!".format(source_filename)
    source_ext = os.path.splitext(source_filename)[1] if source_filename is not None else None

    if source_filename is None:
        source_filename = "default.xyz"
        # cloud = np.copy(mesh.vertices)
        cloud, _ = random_in_triangulation(mesh.vertices, mesh.faces, N)

        M = diam2(cloud)

        # Default perturbation parameters : translation, rotation and noise
        noise_intensity = 0.5 * M
        vprint("Noise intensity = {}".format(noise_intensity))
        cloud = add_noise(cloud, noise_intensity) # noise
        cloud += 1.0 * np.array([0, 0, 1]) # translation
        cloud = transform_3d_cloud(rotation_matrix_homogeneous([0, 1, 0], math.pi / 2),
                                   cloud) # rotation

        vprint("Saving initial point cloud to results/icp/initial_{}.xyz".format(target_mesh_basename))
        np.savetxt("results/icp/initial_{}.xyz".format(target_mesh_basename), cloud)
    elif source_ext in [".xyz", ".txt", ".cloud"]:
        assert os.path.isfile(source_filename), "{} does not exist".format(source_filename)
        cloud = np.loadtxt(source_filename)

        if is_2d(cloud):
            z = np.repeat(0, len(cloud))
            cloud = np.hstack((cloud, z.reshape(-1, 1)))
    elif source_ext in [".off", ".noff", ".coff"]:
        X, T = io.read_off(source_filename, ignore_prefix=True)
        cloud, _ = random_in_triangulation(X, T, N)
    else:
        raise RuntimeError("Format {} not supported".format(source_ext[1:]))

    vprint("ICP between {} and {}\n".format(off_file, source_filename))

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plot_tri(mesh.vertices, mesh.faces, fig=fig)
    plot_cloud(cloud, fig=fig)
    # plt.show()

    # Matching between mesh (target) and cloud (source)
    ## Quadratic (L2 nearest neighbor search)
    vprint("Euclidean")
    start_euclidean = time.time()
    final_T, new_cloud, final_error = test_icp(mesh, cloud,
                                               verbose=VERBOSE,
                                               max_iterations=maxit_icp, tolerance=tolerance_icp)
    end_euclidean = time.time()

    ## OT
    vprint("OT")
    ### Source density
    # mu = lambda p: 1.0
    # mu_X = np.apply_along_axis(mu, 1, mesh.vertices)
    mu = Density_2(mesh.vertices, mesh.faces)
    start_ot = time.time()
    final_T_OT, new_cloud_OT, final_error_OT = test_icp(mesh, cloud, ma=mu,
                                                        verbose=VERBOSE,
                                                        max_iterations=maxit_icp, tolerance=tolerance_icp)
    end_ot = time.time()

    # Statistics
    vprint("Euclidean setting:")
    vprint("Running time = {}s".format(end_euclidean - start_euclidean))
    vprint("Final mean error = {}".format(final_error))
    vprint("Transformation =\n {}\n".format(final_T))
    vprint("Saving Euclidean ICP to results/icp/icp_euclidean_{}.xyz".format(target_mesh_basename))
    np.savetxt("results/icp/icp_euclidean_{}.xyz".format(target_mesh_basename), new_cloud)

    vprint("OT setting:")
    vprint("Running time = {}s".format(end_ot - start_ot))
    vprint("Final mean error = {}".format(final_error_OT))
    vprint("Transformation =\n {}\n".format(final_T_OT))
    vprint("Saving OT ICP to results/icp/icp_ot_{}.xyz".format(target_mesh_basename))
    np.savetxt("results/icp/icp_ot_{}.xyz".format(target_mesh_basename), new_cloud_OT)

    # plot_cloud: first = blue, second = green, third = red
    plot = plot_2d_cloud if is_2d(cloud) else plot_cloud

    # Euclidean
    fig_euclidean = plt.figure()
    fig_euclidean.suptitle("Euclidean")

    ax1 = fig_euclidean.add_subplot(1, 2, 1, projection="3d")
    ax1.set_title("vertices and new cloud")
    # ax1.set_aspect("equal")
    plot([mesh.vertices, new_cloud], fig=fig_euclidean, ax=ax1, labels=["target vertices", "icp"])
    ax1.legend()

    ax2 = fig_euclidean.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("old and new cloud")
    # ax2.set_aspect("equal")
    plot([cloud, new_cloud], fig=fig_euclidean, ax=ax2, labels=["initial", "final"])
    ax2.legend()

    synchronize_rotation(fig_euclidean, ax1, ax2)

    # OT
    fig_ot = plt.figure()
    fig_ot.suptitle("OT")

    ax1 = fig_ot.add_subplot(1, 2, 1, projection="3d")
    ax1.set_title("vertices and new cloud")
    # ax1.set_aspect("equal")
    plot([mesh.vertices, new_cloud_OT], ax=ax1, fig=fig_ot, labels=["target vertices", "icp"])
    ax1.legend()

    ax2 = fig_ot.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title("old and new cloud")
    # ax2.set_aspect("equal")
    plot([cloud, new_cloud_OT], ax=ax2, fig=fig_ot, labels=["initial", "final"])
    ax2.legend()

    synchronize_rotation(fig_ot, ax1, ax2)

    # plt.show()
