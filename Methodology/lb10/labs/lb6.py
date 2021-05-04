from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def linear(X, Y):
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)
    plt.subplot(221)
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.xlabel("FLIGHTS")
    plt.ylabel("DISTANCE")


def ridge(X, Y):
    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)
    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, Y)
        coefs.append(float(ridge.coef_))

    ax = plt.gca()
    plt.subplot(222)
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')


def lasso(X, Y):
    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)
    lasso = linear_model.Lasso(max_iter=10000, normalize=True)
    coefs = []
    plt.subplot(223)

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(sklearn.preprocessing.scale(X), Y)
        coefs.append(lasso.coef_)

    ax = plt.gca()
    ax.plot(alphas * 2, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights')


def polynomial(X, Y):
    poly_reg = sklearn.preprocessing.PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(X)
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X_poly, Y)
    Y_pred = linear_regressor.predict(X_poly)
    plt.subplot(222)
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')


def lb6_main():
    path = 'static/images/lb6/plot.png'
    data = pd.read_csv('files/Daily-stats.csv', sep=";")
    X = data.iloc[:, 2].values.reshape(-1, 1)
    Y = data.iloc[:, 3].values.reshape(-1, 1)

    plt.figure(figsize=[15., 8.])
    plt.grid()

    linear(X, Y)
    # ridge(X, Y)
    lasso(X, Y)
    polynomial(X, Y)

    plt.savefig(path)

    return path
