from perceptron import *


def binary_class (labels):
    """ convert strings to binary classifiers """
    for i in range(len(labels)):
        if labels[i] == "Iris-versicolor":
            # if labels[i] == "Iris-versicolor":
            labels[i] = -1
        else:
            labels[i] = 1
    labels = labels.astype(float)
    return labels


def read_data():
    a = genfromtxt("iris.data.txt", dtype=str, delimiter=',')
    # a = a[:50].vstack(a[100:150])
    # a = np.hstack((a, np.ones(len(a),1)))
    a = np.c_[np.ones(len(a)),a]
    # print(a)
    a = np.vstack((a[50:100], a[100:150]))  # second and third data
    # a = a[:100]  # first and second
    # a = np.vstack((a[:50], a[100:150]))
    # a = np.random.shuffle(np.ndarray(a))
    np.random.shuffle(a)
    # print(a)
    return a


def pla(a):
    data_test, data_train, labels_test, labels_train = partitioner_l(a, 40)
    labels_test = binary_class(labels_test)
    labels_train = binary_class(labels_train)
    w = perceptron(data_train, labels_train)

    p_train = check_accuracy(data_train, labels_train, w)
    p_test = check_accuracy(data_test, labels_test, w)

    print("TRAINING ACCURACY: " + str(p_train))
    print("TESTING ACCURACY: " + str(p_test))

    show_line(data_train, labels_train)

if __name__ == '__main__':
    a = read_data()
    pla(a)