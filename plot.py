import matplotlib.pyplot as plt
import numpy as np

import train
import parameters


def lr_plot(theta0=None, theta1=None, show_line=True):
    df = train.get_df('data.csv')
    if not theta0 or not theta1:
        theta0, theta1 = parameters.get_parameters()
    plt.scatter(df['km'], df['price'])

    x = np.array([0, 250000])
    f = lambda km: theta0 + theta1 * km
    if show_line:
        plt.plot(x, f(x), c='black', label="fit line from linear regression.")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lr_plot(True)
