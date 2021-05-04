import numpy as np


def read_file(path):
    file = open(path, "r")
    lines = file.readlines()
    file.close()
    return lines


def write_file(solve, path):
    file = open(path, "w")
    file.write(' '.join([str(elem) for elem in solve]))
    file.close()


def linear(a, b):
    M = np.array(a)
    v = np.array(b)
    res = np.linalg.solve(M, v)
    write_file(res, "files/output_linear.txt")
    return res


def cramer_sol(a, b):
    if (len(a) != len(a[0])) or (len(a) != len(b)):
        return None
    n = len(a)
    x = []
    Delta = np.linalg.det(a)
    for i in range(0, n):
        M = np.array(a).T
        M[i] = b
        x.append(np.linalg.det(M) / Delta)
    write_file(x, "files/output_cramer.txt")
    return x
