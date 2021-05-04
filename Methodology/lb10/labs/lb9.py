import csv as csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def random_forest_classifier(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(n_estimators=500)
    forest = forest.fit(X_train, y_train)
    forest_output = forest.predict(X_test)
    forest_five_hundred = accuracy_score(y_test, forest_output)

    forest = RandomForestClassifier(n_estimators=2000)
    forest = forest.fit(X_train, y_train)
    forest_output = forest.predict(X_test)
    forest_two_thousand = accuracy_score(y_test, forest_output)

    return forest_output, forest_five_hundred, forest_two_thousand


def gradient_boosting_classifier(X_train, y_train, X_test, y_test):
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    gradient_output = clf.predict(X_test)
    return accuracy_score(y_test, gradient_output)


def mlp_classifier(X_train, y_train, X_test, y_test):
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), random_state=1)
    clf.fit(X_train, y_train)
    neural_output = clf.predict(X_test)
    sgd = accuracy_score(y_test, neural_output)

    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=1)
    clf.fit(X_train, y_train)
    neural_output = clf.predict(X_test)
    lbfgs = accuracy_score(y_test, neural_output)

    return sgd, lbfgs


def displayData(X, Y, path):
    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    fig.suptitle("Display randomly images of the training data set")
    for i in range(10):
        for j in range(10):
            ind = np.random.randint(X.shape[0])
            tmp = X[ind, :].reshape(28, 28)
            ax[i, j].set_title("Label: {}".format(Y[ind]))
            ax[i, j].imshow(tmp, cmap='gray_r')  # display it as gray colors.
            plt.setp(ax[i, j].get_xticklabels(), visible=False)
            plt.setp(ax[i, j].get_yticklabels(), visible=False)

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig(path)


def lb9_main():
    path = 'static/images/lb9/plot.png'
    train_df = pd.read_csv("files/train.csv", header=0)
    submit_test_df = pd.read_csv("files/test.csv", header=0)
    train_data = train_df.values
    X_train, X_test, y_train, y_test = train_test_split(train_data[0::, 1::], train_data[0::, 0], test_size=0.2, random_state=0)

    displayData(X_train, y_train, path)

    submit_test_data = submit_test_df.values

    gradient_boosting_accuracy = gradient_boosting_classifier(X_train, y_train, X_test, y_test)
    sgd_accuracy, lbfgs_accuracy = mlp_classifier(X_train, y_train, X_test, y_test)

    output, five_hundred_accuracy, two_thousand_accuracy= random_forest_classifier(X_train, y_train, X_test, y_test)
    predictions_file = open("files/forest_output.csv", "w")
    open_file_object = csv.writer(predictions_file)
    ids = range(output.__len__())
    ids = [x + 1 for x in ids]
    open_file_object.writerow(["ImageId", "Label"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()

    return path, gradient_boosting_accuracy, sgd_accuracy, lbfgs_accuracy, five_hundred_accuracy, two_thousand_accuracy



