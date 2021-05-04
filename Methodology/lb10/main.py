from flask import Flask, render_template
import numpy as np

from labs.lb1 import read_file, linear, cramer_sol
from labs.lb2 import lb2_main
from labs.lb3 import lb3_main
from labs.lb4 import lb4_main
from labs.lb5 import lb5_main
from labs.lb6 import lb6_main
from labs.lb7 import lb7_main
from labs.lb8 import lb8_main
from labs.lb9 import lb9_main

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/lb1')
def lb1():
    matrix = read_file("files/matrix.txt")
    free = read_file("files/free.txt")
    arr_matrix = []
    arr_free = []
    for line in matrix:
        arr_matrix.append(np.float_(line.split(" ")))
    for line in free:
        arr_free.append(float(line))
    linear_r = linear(arr_matrix, arr_free)
    cramer_r = cramer_sol(arr_matrix, arr_free)
    return render_template('lb1.html', linear=linear_r, cramer=cramer_r)


@app.route('/lb2')
def lb2():
    path, isosurface = lb2_main()
    return render_template('lb2.html', url=path, plot=isosurface)


@app.route('/lb3')
def lb3():
    path, guad_res, err, nl_res = lb3_main()
    return render_template('lb3.html', url=path, guad=guad_res, err=err, nl=nl_res)


@app.route('/lb4')
def lb4():
    path = lb4_main()
    return render_template('lb4.html', url=path)


@app.route('/lb5')
def lb5():
    bisect, f_solve, sqrt = lb5_main()
    return render_template('lb5.html', bisect=bisect, f_solve=f_solve, sqrt=sqrt)


@app.route('/lb6')
def lb6():
    path = lb6_main()
    return render_template('lb6.html', url=path)


@app.route('/lb7')
def lb7():
    path = lb7_main()
    return render_template('lb7.html', url=path)


@app.route('/lb8')
def lb8():
    path = lb8_main()
    return render_template('lb8.html', url=path)


@app.route('/lb9')
def lb9():
    path, gradient_boosting_accuracy, sgd_accuracy, lbfgs_accuracy, five_hundred_accuracy, two_thousand_accuracy = lb9_main()
    return render_template('lb9.html', url=path, gba=gradient_boosting_accuracy,
                           sa=sgd_accuracy, la=lbfgs_accuracy, fha=five_hundred_accuracy, tha=two_thousand_accuracy)
