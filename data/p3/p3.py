import numpy
import time

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file


def load_data(path):
    with open(path, 'rb') as f:
        return load_svmlight_file(f, n_features=784)


def train(data, labels):
    svm = SVC()
    start = time.time()
    svm.fit(data, labels)
    print(f'Training time: {time.time()-start}')
    return svm


if __name__ == '__main__':
    training_data, training_labels = load_data('train-01-images.svm')
    test_x, test_y = load_data('test-01-images.svm')
    model = train(training_data, training_labels)
    test_y = numpy.array(test_y)
    predict_y = numpy.array(model.predict(test_x))
    print(test_y - predict_y)
    print(test_y)
    print(predict_y)
    exit(0)