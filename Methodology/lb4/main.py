import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.integrate import odeint


def f(y, t):
    return np.negative(np.exp(-t)) * (10 * np.sin(10 * t) + np.cos(10 * t))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    matplotlib.use('TkAgg')

    y0 = 1.
    a = 0.
    b = 10.
    t = np.arange(a, b + 0.01, 0.01)

    y = odeint(f, y0, t)

    plt.plot(t, y)
    plt.show()
