import math
import numpy as np
from scipy.optimize import fsolve, bisect


def f(x):
    return 2 - math.pow(x, 2)


def get_solve(root):
    return fsolve(f, root)


def get_bisect(a, b, xtol=1e-6):
    return bisect(f, a, b, xtol=xtol)


def lb5_main():
    return np.float_(get_bisect(0, 2, xtol=1e-8)), np.float_(get_solve(2)), math.sqrt(2)
