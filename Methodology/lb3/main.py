from math import cos, pi
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def f(x):
    return np.round_(np.cos(2 * pi * x), 2)


def F(x):
    return np.round_(np.sin(2 * pi * x) / (2 * pi), 2)


def plotquad(f, a, b):
    x = np.linspace(-2 * pi, 2 * pi, 500)
    y = f(x)
    plt.plot(x, y)
    plt.show()

    res, err = quad(f, a, b)

    return "The numerical result is {:f} (+-{:g})".format(res, err)


def newton_leibniz(a, b):
    return F(b) - F(a)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    matplotlib.use('TkAgg')
    a = 0.
    b = 1.
    res, err = quad(f, a, b)

    print("The numerical result is {:f} (+-{:g})".format(res, err))
    print(plotquad(f, a, b))
    print("The analytic result is {:f}".format(newton_leibniz(a, b)))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
