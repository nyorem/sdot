#! /usr/bin/env python

# Usage: bluenoise.py off_file number_points [output_suffix]
# Examples:
# - python3 examples/bluenoise.py assets/hemisphere.off 1000

# Python imports
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy as sp
import scipy.misc
import sys
import time

# Imports from the 'sdot' library
from sdot.backend import Density
from sdot.core.constants import DEBUG, PROFILE, VERBOSE
from sdot.core.common import vprint, random_in_triangulation, count_zeros, plot_cloud, eprint, \
    is_2d, plot_2d_cloud
import sdot.core.inout as io
from sdot.optimal_transport import optimal_transport_3, init_optimal_transport_3, Results

def blue_noise_sampling(mu, N,
                        iter_smoothing=3, iter_blue_noise=5,
                        eps=1e-8,
                        output_suffix=None, verbose=False):
    eps_init = 1e-1

    # Target
    Y, _ = random_in_triangulation(mu.vertices, mu.triangles, N)
    nu = np.repeat(1.0, N)

    if output_suffix is not None:
        np.savetxt("initial_" + output_suffix + ".xyz", Y)

    ## Initial weights
    psi0 = init_optimal_transport_3(mu, Y, nu, method="local", eps_init=eps_init, verbose=verbose)

    ## Normalization of the source measure
    A = mu.kantorovich(Y, nu, psi0)[0]
    nu = A.sum() * nu / nu.sum()

    # Smoothen point cloud
    vprint("Smoothen initial point cloud...")
    for i in range(1, iter_smoothing + 1):
        vprint("LLOYD it = %d" % i)

        Z = mu.kantorovich(Y, nu, psi=psi0)[Results.C]
        Y = Z

    vprint("Smoothing done")

    # Bluenoise sampling
    vprint("Bluenoise sampling...")

    # psi0 = tree.squared_distances(Y)
    psi0 = init_optimal_transport_3(mu, Y, nu, method="local", eps_init=eps_init, verbose=verbose)
    for i in range(1, iter_blue_noise + 1):
        vprint("BLUENOISE it = %d" % i)

        psi = optimal_transport_3(mu, Y, nu, psi0=psi0, eps=eps, verbose=verbose)
        Z = mu.kantorovich(Y, nu, psi)[Results.C]
        Y = Z

    vprint("Blue noise sampling done")

    return Y

# Load an image
# channel = None => greyscale
def load_image(path, w, h, channel=None, flip=False, flip_up=False, verbose=VERBOSE):
    if channel is None:
        # Greyscale
        V = imageio.imread(path, as_gray=True)
    else:
        V = imageio.imread(path)
        channel_i = channel_to_int(channel)
        assert channel_i != -1
        if channel != "RGB" and len(V.shape) == 3:
            V = V[:, :, channel_i]

    # V.shape = (height, width)
    resized_image = np.array(Image.fromarray(V).resize((h, w)))
    V = np.fliplr(V) if flip else V
    V = np.flipud(V) if flip_up else V

    if DEBUG:
        if channel is None:
            target_fname = "target.png"
        else:
            target_fname = "target_" + channel + ".png"
        imageio.imwrite(target_fname, V)

    return V

# Create a target measure from an image
def target_from_image(path, dims=None, channel=None, size_target=1, height_target=-1,
                      flip=False, flip_up=False,
                      origin=None, verbose=VERBOSE):
    if isinstance(dims, tuple):
        w, h = dims
    else:
        w, h = dims, dims
    ratio =  w / h

    # Values
    V = load_image(path, w, h, channel=channel, flip=flip, flip_up=flip_up, verbose=verbose)
    V = V.flatten()
    V = np.asarray(V, dtype=float)

    # Directions
    bbox = [-size_target * ratio, size_target * ratio, -size_target, size_target]
    xs, ys = np.meshgrid(np.linspace(bbox[0], bbox[1], w),
                         np.linspace(bbox[2], bbox[3], h))

    X = xs.flatten()
    Y = ys.flatten()
    Z = np.repeat(height_target, w * h)
    grid_points = np.vstack((X, Y, Z)).T

    # translate
    if origin is not None:
        grid_points += origin

    return grid_points, V

assert len(sys.argv) == 4, "Usage: {} source.off N output_suffix".format(sys.argv[0])

# Source
off_file = sys.argv[1]
X_source, T_source = io.read_off(off_file, ignore_prefix=True)
# Target
nY = int(sys.argv[2])
output_suffix = sys.argv[3]

# Source density
## Examples
## Values from an array
# mu_values = np.repeat(1.0, len(X_source))
# mu = Density(X_source, T_source, mu_values)

### Values from a function
# mu = Density.from_function(X_source, T_source, lambda p: 1.0) # uniform
# mu = Density.from_function(X_source, T_source, lambda p: math.exp(-3 * abs(p[1]))) # custom

### From a texture
# filename = "examples/assets/textures/checkerboard.png"
filename = "examples/assets/textures/shinobu.jpg"
tex = imageio.imread(filename, as_gray=True)

if "plane" in off_file:
    # dims = tex.shape
    dims = (tex.shape[0] // 4, tex.shape[1] // 4)
    nY = dims[0] * dims[1]
    print(nY)
    X_source, tex = target_from_image(filename, dims=dims, height_target=0,
                                         flip=False, flip_up=True)
    T_source = sp.spatial.Delaunay(X_source[:, 0:2]).simplices

    tex = 255 - tex
    # black_areas = np.where(tex == 0)
    # tex[black_areas] = 10
    mu = Density(X_source, T_source, V=tex)
elif "sphere" in off_file:
    #### UV-mapping on the sphere (u and v are between 0 and 1)

    def uv_map_sphere(p):
        d = p / np.linalg.norm(p)
        u = 0.5 + math.atan2(d[0], d[2]) / (2 * math.pi)
        v = 0.5 - math.asin(d[1]) / math.pi
        return u, v

    mu = Density.from_texture(X_source, T_source, tex, uv_map_sphere)
else:
    vprint("UV-mapping not defined for off_file={}".format(off_file))
    sys.exit(1)

#### Directional
# mu = Density.directional(X_source, T_source, dir=[0, 0, 1])

### From a file
# mu = Density.from_file(X_source, T_source, "curvature_bunny_simplified.txt", off_file)

if DEBUG:
    mu.plot()
    plt.show()

# Sampling
start = time.time()
Y = blue_noise_sampling(mu, nY, iter_smoothing=5, iter_blue_noise=10,
                        eps=1e-8, output_suffix=output_suffix, verbose=VERBOSE)
end = time.time()

if PROFILE:
    eprint("GLOBAL Running time: %f" % (end - start))
else:
    vprint("GLOBAL running time = %f seconds" % (end - start))

# Results
np.savetxt("final_" + output_suffix + ".xyz", Y)
