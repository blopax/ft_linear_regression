import matplotlib.pyplot as plt
import numpy as np

import train
import parameters


def lr_plot(theta0=None, theta1=None, show_line=True):
    df = train.get_df('data.csv')
    if not theta0 or not theta1:
        theta0, theta1 = parameters.get_parameters()
    if df is not None:
        plt.scatter(df['km'], df['price'])

        x = np.array([0, 250000])
        f = lambda km: theta0 + theta1 * km
        if show_line:
            plt.plot(x, f(x), c='black', label="Fit line from linear regression.")

        plt.title("Data points and linear regression")
        plt.legend()
        plt.show()


def cost_plot(cost_list):
    x = np.array(range(1, 1 + len(cost_list)))
    y = np.array(cost_list)
    plt.plot(x, y, label="Cost function.")
    plt.title("Cost function.")
    plt.legend()
    plt.show()
