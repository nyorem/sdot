import numpy as np

# Read a OFF file
# Faces must be triangles
def read_off(fname, with_normals=False, with_colors=False, ignore_prefix=False):
    assert not (with_normals and with_colors)

    prefix = "OFF"
    if with_normals:
        prefix = "NOFF"
    if with_colors:
        prefix = "COFF"

    with open(fname, "r") as file:
        if not ignore_prefix:
            if prefix != file.readline().strip():
                raise Exception("Not a valid " + prefix + " header")
        else:
            file.readline()

        n_verts, n_faces, n_edges = tuple([int(s) for s in file.readline().strip().split(' ')])

        verts = []
        if with_normals or with_colors:
            extra = []

        i_vert = 0
        while i_vert < n_verts:
            elems = file.readline().strip().split(' ')

            # Skip empty lines
            if elems == ['']:
                continue

            i_vert += 1
            line = [float(s) for s in elems ]
            verts.append([line[0], line[1], line[2]])
            if with_normals:
                extra.append([line[3], line[4], line[5]])
            if with_colors:
                extra.append([line[3], line[4], line[5], line[6]])

        faces = []
        for i_face in range(n_faces):
            elems = file.readline().strip().split(' ')
            # Ignore whitespace end empty lines
            elems = list(filter(bool, elems))
            if len(elems) == 1:
                continue
            face_line = [int(s) for s in elems][1:]
            assert(len(face_line) == 3), "read_off can only read triangulations"
            faces.append(face_line)

    if with_normals or with_colors:
        return np.array(verts), np.array(extra), np.array(faces)
    else:
        return np.array(verts), np.array(faces)

# Read a NOFF file
# Faces must be triangles
def read_noff(fname, **kwargs):
    return read_off(fname, with_normals=True, **kwargs)

# Read a COFF file
# Faces must be triangles
def read_coff(fname, **kwargs):
    return read_off(fname, with_colors=True, **kwargs)

# Write a OFF file
# Faces must be triangles
def write_off(X, T, filename, N=None):
    prefix = "OFF"
    if N is not None:
        prefix = "NOFF"
    with open(filename, "w") as off_file:
        off_file.write(prefix + "\n")
        off_file.write(str(len(X)) + " " + str(len(T)) + " 0\n")
        for i in range(len(X)):
            c = X[i]
            off_file.write(str(c[0]) + " " + str(c[1]) + " " + str(c[2]))
            if N is not None:
                n = N[i]
                off_file.write(" " + str(n[0]) + " " + str(n[1]) + " " + str(n[2]) + "\n")
            else:
                off_file.write("\n")

        for i in range(len(T)):
            simplex = T[i]
            off_file.write("3 " + str(simplex[0]) + " " + str(simplex[1]) + " " + str(simplex[2]) + "\n")

# Write a NOFF file
# Faces must be triangles
def write_noff(X, N, T, filename):
    write_off(X, T, filename, N=N)
