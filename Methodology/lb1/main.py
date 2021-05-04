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
    write_file(np.linalg.solve(M, v), "files/output_linear.txt")


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


if __name__ == '__main__':
    matrix = read_file("files/matrix.txt")
    free = read_file("files/free.txt")
    arr_matrix = []
    arr_free = []
    for line in matrix:
        arr_matrix.append(np.float_(line.split(" ")))
    for line in free:
        arr_free.append(float(line))
    linear(arr_matrix, arr_free)
    cramer_sol(arr_matrix, arr_free)
    print("Success")
