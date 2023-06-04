import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_all_portfolios(stat_arbs, data_train, data_val, data_test):
    """
    param stat_arbs: list of StatArb objects
    param data_train: pandas dataframe with training data
    param data_val: pandas dataframe with validation data
    param data_test: pandas dataframe with test data
    """
    cs = ["b", "g", "r", "c", "m", "y"]

    for i, stat_arb in enumerate(stat_arbs):
        # s, mu = stat_arb.s, stat_arb.mu

        c = cs[i % len(cs)]

        data_concat = pd.concat([data_train, data_val, data_test], axis=0)
        plt.plot(stat_arb.evaluate(data_concat) - stat_arb.mu, c=c)

        # plt.plot(stat_arb.evaluate(data_train) - stat_arb.mu, c=c)
        # plt.plot(stat_arb.evaluate(data_val) - stat_arb.mu, c=c)
        # plt.plot(stat_arb.evaluate(data_test) - stat_arb.mu, c=c)
    plt.axvline(data_val.index[0], color="k", label="train->val", linewidth=1)
    plt.axvline(data_test.index[0], color="k", label="val->test", linewidth=1)

    plt.plot(
        [data_train.index[0], data_test.index[-1]],
        [1, 1],
        "k--",
        label="constraint",
        linewidth=1,
    )
    plt.plot([data_train.index[0], data_test.index[-1]], [-1, -1], "k--", linewidth=1)


def determine_n_plots(candidate_stat_arbs, data_val, cutoff):
    n_plots = 0
    for i, stat_arb in enumerate(candidate_stat_arbs):
        (p, s, mu, all_assets, holdings) = (
            stat_arb.p,
            stat_arb.s,
            stat_arb.mu,
            stat_arb.all_assets,
            stat_arb.holdings,
        )

        data_new_val = data_val[all_assets]

        if np.abs(data_new_val.values @ s - mu).max() <= cutoff:
            n_plots += 1
    return n_plots


def fill_ax(ax, data_train, data_val, data_test, s, mu, c):
    ax.plot(data_train @ s - mu, c=c, linewidth=10)
    ax.plot(data_val @ s - mu, c=c, linewidth=10)
    ax.plot(data_test @ s - mu, c=c, linewidth=10)

    ax.axvline(data_val.index[0], color="k", label="train->val", linewidth=10)
    ax.axvline(data_test.index[0], color="k", label="val->test", linewidth=10)

    ax.plot(
        [data_train.index[0], data_test.index[-1]],
        [1, 1],
        "k--",
        label="constraint",
        linewidth=10,
    )
    ax.plot([data_train.index[0], data_test.index[-1]], [-1, -1], "k--", linewidth=10)

    n_nonzero = np.sum(np.abs(s) > 1e-5)
    title = "mu=" + str(np.round(mu, 2)) + ", |s|=" + str(n_nonzero)
    ax.set_title(title, fontsize=150)


