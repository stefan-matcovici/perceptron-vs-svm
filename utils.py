import os

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings("ignore")


def read_p1(problem_name):
    x = pd.read_csv(os.path.join("data", "p1", f"p1_{problem_name}_X.dat"), sep="   ", header=None, engine='python')
    y = pd.read_csv(os.path.join("data", "p1", f"p1_{problem_name}_y.dat"), header=None)

    y = y.values.reshape((-1,))
    x = x.values

    return x, y


def plot_points(x, y):
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=y,
        palette=sns.color_palette("muted", n_colors=2)
    )


def plot_separator(w, bias):
    slope = -w[0] / w[1]
    intercept = -bias / w[1]

    limits = plt.axes().get_xlim()

    x = np.arange(limits[0], limits[1])
    plt.plot(x, x * slope + intercept, 'k-')

    plt.legend(loc='best')


def plot_support_vectors(x, support_vectors):
    sns.scatterplot(
        x=x[support_vectors, 0],
        y=x[support_vectors, 1],
        marker="P",
        s=100,
        c=['C8'],
        label='support vectors'
    )

    plt.legend(loc='best')


def save_plot(filename, extension="eps"):
    file_path = os.path.join("plots", f"{filename}.{extension}")
    with open(file_path, "w") as f:
        plt.savefig(file_path)
