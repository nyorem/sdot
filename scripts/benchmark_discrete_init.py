import sys
import subprocess
import os
import matplotlib.pyplot as plt
import pickle
from sdot.core.constants import HEMISPHERE_MESH, SPHERE_MESH

source_mesh = HEMISPHERE_MESH
target_mesh = SPHERE_MESH
fontsize = 25

def parse_output(out):
    import ast
    return ast.literal_eval(out.decode("utf-8").split("\n")[-2])

if __name__ == "__main__":
    running_times = {}
    n_targets = [ 1000, 5000, 10000 ]

    for n_target in n_targets:
        print("n_target={}".format(n_target))

        command = "python examples/misc/test_discrete_init.py {} {} {}".format(
            os.path.join(source_mesh),
            os.path.join(target_mesh),
            n_target)

        out = subprocess.check_output(command, shell=True)
        current_times = parse_output(out)

        for method, t in current_times.items():
            running_times.setdefault(method, []).append(t)

    pickle.dump(running_times, open("/tmp/discrete.pkl", "wb"))
    # running_times = pickle.load(open("/tmp/discrete.pkl", "rb"))

    print(running_times)

    to_plot = ["normal", "local", "rescale_final", "interp_final"]
    c1 = "#1f77b4"
    c2 = "#ff7f0e"
    c3 = "#2ca02c"
    c4 = "#d62728"
    colors = { "normal"       : c1,
               "local"        : c2,
               "rescale_final": c3,
               "interp_final" : c4
              }
    fig = plt.figure()
    for method, times in running_times.items():
        if method not in to_plot:
            continue
        plt.plot(n_targets, times, c=colors[method])
    plt.xlabel("Discretization", fontsize=fontsize)
    plt.ylabel("Running time", fontsize=fontsize)
    plt.show()
