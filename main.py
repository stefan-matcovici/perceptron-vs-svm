from svm import fit
from utils import *

if __name__ == "__main__":
    x, y = read_p1("a")
    plot_points(x, y)

    w, bias, support_vectors = fit(x, y)
    plot_separator(w, bias)
    plot_support_vectors(x, support_vectors)

    save_plot("separator-p1-a", "png")

    plt.clf()

    x, y = read_p1("b")
    plot_points(x, y)

    w, bias, support_vectors = fit(x, y)
    plot_separator(w, bias)
    plot_support_vectors(x, support_vectors)

    save_plot("separator-p1-b", "png")