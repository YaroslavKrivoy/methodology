from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def ada_boost_regressor(X, Y):
    rng = np.random.RandomState(1)
    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)
    regr.fit(X, Y.ravel())
    y_pred = regr.predict(X)
    plt.subplot(221)
    plt.title('Ada boost')
    plt.plot(X, y_pred, color="blue", label="AdaBoost")


def random_forest(X, Y):
    regr = RandomForestRegressor(max_depth=4, random_state=0)
    regr.fit(X, Y.ravel())
    y_pred = regr.predict(X)
    plt.subplot(222)
    plt.title('Random forest')
    plt.plot(X, y_pred, color="green", label="RandomForest")


def extra_tree(X, Y):
    regr = ExtraTreesRegressor(max_depth=4, n_estimators=100, random_state=0)
    regr.fit(X, Y.ravel())
    y_pred = regr.predict(X)
    plt.subplot(223)
    plt.title('Extra tree')
    plt.plot(X, y_pred, color="yellow", label="RandomForest")


def lb7_main():
    path = 'static/images/lb7/plot.png'
    data = pd.read_csv('files/Daily-stats.csv', sep=";")
    X = data.iloc[:, 2].values.reshape(-1, 1)
    Y = data.iloc[:, 3].values.reshape(-1, 1)

    plt.figure(figsize=[15., 8.])
    plt.grid()

    plt.scatter(X, Y, color='red', linestyle="dashed", label="Initial")
    ada_boost_regressor(X, Y)
    random_forest(X, Y)
    extra_tree(X, Y)
    plt.savefig(path)
    return path
