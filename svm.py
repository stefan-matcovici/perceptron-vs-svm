import numpy as np
from cvxopt import matrix, solvers


def fit(x, y):
    NUM = x.shape[0]

    # compute identity kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)

    # 1 * ||a_i||
    q = matrix(-np.ones((NUM, 1)))
    # -a_i <= h
    G = matrix(-np.eye(NUM))

    # = 0
    h = matrix(np.zeros(NUM))

    # a_i * y_i
    A = matrix(y.reshape(1, -1))

    # = 0
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False

    # solve
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    # go into primal form a_i * y_i * x_i
    w = np.sum(alphas * y[:, None] * x, axis=0)

    # support vectors are the ones that have alpha > 0
    support_vectors = (alphas > 1e-4).reshape(-1)

    #compute b as a mean of all the equations for support vectors
    b = y[support_vectors] - np.dot(x[support_vectors], w)
    bias = np.mean(b)

    return w, bias, support_vectors
