import pandas as pd

import numpy
import time
import matplotlib.pyplot as plt
import seaborn as sns
import decimal
from PIL import Image


from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_svmlight_file


# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def load_data(path):
    with open(path, 'rb') as f:
        data, labels = load_svmlight_file(f, n_features=784)
        return data, labels


def train(data, labels, c):
    svm = LinearSVC(C=c, loss='hinge', penalty='l2')
    start = time.time()
    svm.fit(data, labels)
    print(f'Training time: {time.time() - start}')
    return svm


def plot_data(errors, c_vals=None):
    if c_vals is None:
        c_vals = c_values
    df = pd.DataFrame(data={'C': [*c_vals, *c_vals], 'Error': errors[0] + errors[1], 'Type': ['train']*10 + ['test']*10})

    # Initialize figure and ax
    fig, ax = plt.subplots()
    # Set the scale of the x-and y-axes
    ax.set(xscale="log", yscale="log")
    # Create a regplot
    sns.lineplot("C", "Error", data=df, ax=ax, hue='Type')
    # Show plot
    plt.show()


def load_precomputed_data(path):
    with open(f'../../{path}') as f:
        content = f.readlines()
        content = list(map(float, content))
        return numpy.sign(numpy.array(content))


def plot_precomputed_data():
    # for c in numpy.geomspace(10**-13, 10**-4, num=10):
    train_x, train_y = load_data('train-01-images.svm')
    test_x, test_y = load_data('test-01-images.svm')
    train_y = numpy.array(train_y)
    test_y = numpy.array(test_y)
    train_error = []
    test_error = []
    train_mis_error = []
    test_mis_error = []
    # for c in range(4, 14):
    for c in range(0, 1):
        train_classif = load_precomputed_data(f'predictions_train_{c}')
        test_classif = load_precomputed_data(f'predictions_test_{c}')
        # train_mislabeled_classif = load_precomputed_data(f'predictionsW_train_{c}')
        # test_mislabeled_classif = load_precomputed_data(f'predictionsW_test_{c}')

        diff = train_y - train_classif
        train_error.append(sum(numpy.where(diff == 0, 0, 1))/len(train_y))
        m1 = Image.fromarray(train_x[314].reshape((28,28)).todense()).convert('RGB')
        m1.save('m1.png')
        m1 = Image.fromarray(train_x[2823].reshape((28,28)).todense()).convert('RGB')
        m1.save('m2.png')
        m1 = Image.fromarray(train_x[4495].reshape((28,28)).todense()).convert('RGB')
        m1.save('m3.png')
        exit(1)

        diff = test_y - test_classif
        test_error.append(sum(numpy.where(diff == 0, 0, 1)) / len(test_y))

        diff = train_y - train_mislabeled_classif
        train_mis_error.append(sum(numpy.where(diff == 0, 0, 1)) / len(train_y))

        diff = test_y - test_mislabeled_classif
        test_mis_error.append(sum(numpy.where(diff == 0, 0, 1)) / len(test_y))
    plot_data([train_error, test_error], numpy.geomspace(10**-13, 10**-4, num=10))
    plot_data([train_mis_error, test_mis_error], numpy.geomspace(10**-13, 10**-4, num=10))


def svm_train_and_validate(train_file, test_file, data_type):
    global c_values
    train_x, train_y = load_data(train_file)
    test_x, test_y = load_data(test_file)
    test_y = numpy.array(test_y)
    c_values = [10 ** 5, 10 ** -10, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
    training_errors = []
    testing_errors = []
    for c in c_values:
        model = train(train_x, train_y, c)
        testing_errors.append(1 - model.score(test_x, test_y))
        training_errors.append(1 - model.score(train_x, train_y))

    plot_data(testing_errors)
    path = f'testing_error_{data_type}.png'
    with open(path, 'w') as f:
        plt.savefig(path)

    plot_data(training_errors)
    path = f'training_error_{data_type}.png'
    with open(path, 'w') as f:
        plt.savefig(path)

    print(testing_errors)
    print(training_errors)


if __name__ == '__main__':
    # svm_train_and_validate('train-01-images.svm', 'test-01-images.svm', 'simple')
    # svm_train_and_validate('train-01-images-W.svm', 'test-01-images.svm', 'mislabeled')
    plot_precomputed_data()
