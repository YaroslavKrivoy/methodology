from math import pi
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.round_(np.cos(2 * pi * x), 2)


def F(x):
    return np.round_(np.sin(2 * pi * x) / (2 * pi), 2)


def plotquad(f, a, b, path):
    x = np.linspace(-2 * pi, 2 * pi, 500)
    y = f(x)
    plt.plot(x, y)
    plt.savefig(path)

    res, err = quad(f, a, b)


def newton_leibniz(a, b):
    return F(b) - F(a)


# Press the green button in the gutter to run the script.
def lb3_main():
    path = 'static/images/lb3/plot.png'
    a = 0.
    b = 1.
    guad_res, err = quad(f, a, b)
    plotquad(f, a, b, path)
    nl_res = newton_leibniz(a, b)

    return path, guad_res, err, nl_res
