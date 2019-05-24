import numpy as np


class Perceptron:
    """
    Perceptron classifier.
    This model is the baseline for beginning working with artificial neural networks(deep learning).
    This acts very much like a linear regression classifier.

    This model is able to learn simple models based on an array of features and a binary target.

    Example inspired from Python Machine Learning book and adapted.

    Parameters:
        - learning_rate = how much the model should learn (overfitting vs underfitting).
                          This value should be between 1 and 0.
        - epochs = number of iterations over the training data set

    Attributes:
        - weights = the weights array result after fitting the model
        - errors = the number of misclassifications
    """

    def __init__(self, learning_rate=0.01, epochs=10):
        self._errors = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, Y):
        """
        Fits the model for the given targets.

        :param X: array of training data with shape (number_of_samples, number_of_features)
        :param Y: array of targets for the training data with shape (number_of_samples)
        :return: self, the fitted model
        """
        # + 1 because we want to also have a free term (bias) that is not influenced by the training values necessarily.
        self._weights = np.zeros(1 + X.shape[1])

        for _ in range(self.epochs):

            errors = 0
            # We now parse the training data set
            for entry, target in zip(X, Y):
                classification_error = target - self.predict(entry)

                # we compute now with how much we should adjust the weights
                weights_update = self.learning_rate * classification_error

                # Adjust the weights based on the error (+/- 1 or 0) and the training entry
                self._weights[1:] += weights_update * entry
                self._weights[0] += weights_update

                errors += np.where(classification_error == 0, 0, 1)

            self._errors.append(errors)

        return self._weights

    def predict(self, entry):
        # compute the predicted value
        return np.where(np.dot(entry, self._weights[1:]) + self._weights[0] >= 0.0, 1, -1)