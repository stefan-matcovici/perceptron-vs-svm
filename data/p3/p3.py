import pandas as pd

import numpy
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_svmlight_file


def load_data(path):
    with open(path, 'rb') as f:
        data, labels = load_svmlight_file(f, n_features=784)
        return data, labels


def train(data, labels, c):
    svm = SVC(C=c, kernel='linear', shrinking=False)
    start = time.time()
    svm.fit(data, labels)
    print(f'Training time: {time.time() - start}')
    return svm


def plot_data(errors):
    df = pd.DataFrame(data={'C': c_values, 'Error': errors})
    # Initialize figure and ax
    fig, ax = plt.subplots()
    # Set the scale of the x-and y-axes
    ax.set(xscale="log", yscale="linear")
    # Create a regplot
    sns.regplot("C", "Error", data=df, ax=ax)
    # Show plot
    plt.show()


def svm_train_and_validate(train_file, test_file, data_type):
    global c_values
    train_x, train_y = load_data(train_file)
    test_x, test_y = load_data(test_file)
    test_y = numpy.array(test_y)
    c_values = [10 ** 3, 10 ** -10, 10 ** -9, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5]
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
    svm_train_and_validate('train-01-images.svm', 'test-01-images.svm', 'simple')
    # svm_train_and_validate('train-01-images-W.svm', 'mislabeled')
