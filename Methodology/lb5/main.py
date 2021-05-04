import math
import numpy as np
from scipy.optimize import fsolve, bisect


def f(x):
    return 2 - math.pow(x, 2)


def get_solve(root):
    return fsolve(f, root)


def get_bisect(a, b, xtol=1e-6):
    return bisect(f, a, b, xtol=xtol)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Bisect root = {:f}".format(np.float_(get_bisect(0, 2, xtol=1e-8))))
    print("Fsolve root = {:f}".format(np.float_(get_solve(2))))
    print("Math.sqrt = {:f}".format(math.sqrt(2)))
