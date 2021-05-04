import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import json
import plotly


def f1(x):
    return np.round_(np.exp(x) * np.sin(4 * x), 2)


def f2(x):
    return np.round_(np.exp((-1.5) * x) * np.sin(4 * x), 2)


def z(x, y):
    return np.round_(np.cos(x + y) * np.sin(x * y))


def con(a, b):
    return np.round_(a + b - np.sqrt(np.power(a, 2) + np.power(b, 2)))


def draw_plot(x, y, subplot, color="black", linestyle="solid", xlabel="x", ylabel="y", legend="", title=""):
    plt.subplot(subplot)
    plt.plot(x, y, color=color, linestyle=linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)


def draw_isosurface(x, y, z):
    fig = go.Figure(data=go.Isosurface(
        x=x,
        y=y,
        z=z,
        value=con(con(1 - np.power(x, 2), 1 - np.power(y, 2)), 1 - np.power(z, 2)),
    ))

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig.write_image('static/images/lb2/iso.png')

    return graph_json


def lb2_main():
    path = 'static/images/lb2/plot.png'
    start = 0
    end = 4
    x_arr = np.arange(start, end + (end - start) / 500, (end - start) / 500)
    start_z = -2
    end_z = 2
    x_z = np.arange(start_z, end_z + (end_z - start_z) / 500, (end_z - start_z) / 500)
    y_z = np.arange(start_z, end_z + (end_z - start_z) / 500, (end_z - start_z) / 500)

    plt.figure(1)
    plt.grid()

    draw_plot(x_arr, f1(x_arr), 221, "red", xlabel="x", ylabel="y", legend="g", title="Damped sine wave")
    draw_plot(x_arr, f2(x_arr), 222, "black", "dashed", "x", "y", "h")
    draw_plot(x_z, z(x_z, y_z), 223, "green", xlabel="x", ylabel="y", legend="z")

    isosurface = draw_isosurface([0, 0, 0, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0])
    plt.savefig(path)
    return path, isosurface
