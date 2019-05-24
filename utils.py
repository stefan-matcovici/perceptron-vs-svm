import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_p1(problem_name):
    df = pd.read_csv(os.path.join("data", "p1", f"p1_{problem_name}_X.dat"), sep="   ", header=None, names=['x1', 'x2'], engine='python')
    y = pd.read_csv(os.path.join("data", "p1", f"p1_{problem_name}_y.dat"), header=None, names=['y'])

    df['y'] = y
    return df


def plot_points(df):
    sns.scatterplot(
        x='x1',
        y='x2',
        data=df,
        hue='y',
        palette=sns.color_palette("muted", n_colors=2)
    )


def save_plot(filename, extension="eps"):
    file_path = os.path.join("plots", f"{filename}.{extension}")
    with open(file_path, 'w') as f:
        plt.savefig(file_path)
