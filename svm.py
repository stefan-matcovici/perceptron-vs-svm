import numpy as np
from cvxopt import matrix, solvers
import seaborn as sns


def fit(x, y):
    NUM = x.shape[0]

    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))
    G = matrix(-np.eye(NUM))
    h = matrix(np.zeros(NUM))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    w = np.sum(alphas * y[:, None] * x, axis=0)
    support_vectors = (alphas > 1e-4).reshape(-1)
    b = y[support_vectors] - np.dot(x[support_vectors], w)
    bias = np.mean(b)

    return w, bias, support_vectors
