# file: perceptron.py
# -------------------
# Classification algorithm

from numpy import *
import random
import matplotlib.pyplot as plt
import numpy as np
from partitioner import *
import numpy as np


def partitioner_l(data, test_examples):

    m, n = data.shape
    # print(data)
    labels_train = data[:, n-1]

    data_train = data[:, :n-1]
    data_train = data_train.astype(float)
    m, n = data_train.shape

    # data_test = ones((1, n))
    # num_partitions = int(float(test_examples)/100 * m)
    # labels_test = array([1])  # 1 x 1 matrix

    data_test = data_train[:test_examples]
    labels_test = labels_train[:test_examples]
    data_train = data_train[test_examples:]
    labels_train = labels_train[test_examples:]
    # data_test = data_test[1:]
    # labels_test = labels_test[1:]

    return data_test, data_train, labels_test, labels_train


def unit_step(x):
    if x < 0:
        return -1.
    else:
        return 1.


def perceptron (data, labels):
    m, n = data.shape
    # w = np.random.rand(n)
    w = np.zeros(n)
    # eta = 0.2 # learning rate
    count = 1000 # number of learning iterations
    errors_sum = 0
    curr_errors = 0
    pocket = np.zeros(n)
    for i in range(count):
        for j in range(len(data)):
            result = dot(w, data[j])
            expected = labels[j]
            if expected != unit_step(result):
                curr_errors += 1 
                w += expected*data[j]
        if curr_errors < errors_sum or errors_sum==0:
            errors_sum = curr_errors
            pocket = w
        curr_errors = 0
        
    print(pocket)
    print(errors_sum)
    return pocket


def show_line(data, labels):
    plt.figure()
    for i in range(0, len(data)):
        if labels[i] == 1.:
            plt.scatter(data[i, 1], data[i, 2], marker=">")
        else:
            plt.scatter(data[i, 1], data[i, 2], marker="+")
    plt.show()


def check_accuracy(data, labels, weights):
    """ Use weights to classify data points and check the accuracy """
    count = 0
    gs = []
    rs = []
    for x in range(0, len(data)):
        results = dot(data[x], weights)
        guess = unit_step(results)
        gs.append(guess) # append prediction
        rs.append(labels[x]) # append result
        if guess == labels[x]:
            count += 1

    percentage = ((float(count) / len(data)) * 100)

    return percentage
