from perceptron import Perceptron
from svm import fit
from utils import *

if __name__ == "__main__":
    x, y = read_p1("a")
    plot_points(x, y)

    w, bias, support_vectors = fit(x, y)
    plot_separator(w, bias)
    plot_svm_margin(w)
    plot_support_vectors(x, support_vectors)

    save_plot("svm-separator-p1-a", "png")
    plt.clf()

    plot_points(x, y)
    perceptron = Perceptron()
    weights = perceptron.fit(x, y)

    bias = weights[0]
    w = weights[1:]

    plot_separator(w, bias)
    plot_perceptron_margin(x, w, perceptron.updates)
    print(f"No support vectors: {sum(np.where(support_vectors == True, 1, 0))}")

    save_plot("perceptron-separator-p1-a", "png")
    plt.clf()

    x, y = read_p1("b")
    plot_points(x, y)

    w, bias, support_vectors = fit(x, y)
    plot_separator(w, bias)
    plot_svm_margin(w)
    plot_support_vectors(x, support_vectors)
    print(f"No support vectors: {sum(np.where(support_vectors == True, 1, 0))}")

    save_plot("svm-separator-p1-b", "png")
    plt.clf()

    plot_points(x, y)
    perceptron = Perceptron()
    weights = perceptron.fit(x, y)

    bias = weights[0]
    w = weights[1:]

    plot_separator(w, bias)
    plot_perceptron_margin(x, w, perceptron.updates)

    save_plot("perceptron-separator-p1-b", "png")
    plt.clf()
