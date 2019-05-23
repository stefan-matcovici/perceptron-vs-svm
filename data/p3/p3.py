import pandas as pd

import numpy
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file

from utils import save_plot


def load_data(path):
    with open(path, 'rb') as f:
        return load_svmlight_file(f, n_features=784)


def train(data, labels, c):
    svm = SVC(C=c)
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
    sns.regplot("C", "Error", df, ax=ax, scatter_kws={"s": 100})
    # Show plot
    plt.show()


def svm_train_and_validate(train_file, data_type):
    global c_values
    train_x, train_y = load_data(train_file)
    test_x, test_y = load_data('test-01-images.svm')
    test_y = numpy.array(test_y)
    c_values = [0.01, 0.1, 1, 10, 100]
    training_errors = []
    testing_errors = []
    for c in c_values:
        model = train(train_x, train_y, c)
        predict_y = numpy.array(model.predict(test_x))
        unique, counts = numpy.unique(numpy.array(list(map(lambda t: t == 0, test_y - predict_y))), return_counts=True)
        classification_stats = dict(zip(unique, counts))
        testing_errors.append(classification_stats[False])
        print(f'C:{c}\n', classification_stats)

        predict_y = numpy.array(model.predict(train_x))
        training_unique, training_counts = numpy.unique(numpy.array(list(map(lambda t: t == 0, train_y - predict_y))),
                                                        return_counts=True)
        training_error_stats = dict(zip(training_unique, training_counts))
        training_errors.append(training_error_stats[False])
    plot_data(testing_errors)
    save_plot(f'testing_error_{data_type}')
    plot_data(training_errors)
    save_plot(f'training_error_{data_type}')


if __name__ == '__main__':
    svm_train_and_validate('train-01-images.svm', 'simple')
    svm_train_and_validate('train-01-images-W.svm', 'mislabeled')
