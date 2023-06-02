# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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

        plt.plot(stat_arb.evaluate(data_train) - stat_arb.mu, c=c)
        plt.plot(stat_arb.evaluate(data_val) - stat_arb.mu, c=c)
        plt.plot(stat_arb.evaluate(data_test) - stat_arb.mu, c=c)
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


# def get_realized_covs(R):
#     """
#     param R: numpy array where rows are vector of asset returns for t=0,1,...
#         R has shape (T, n) where T is the number of days and n is the number of assets

#     returns: (numpy array) list of r_t*r_t' (matrix multiplication) for all days, i.e,
#         "daily realized covariances"
#     """
#     T = R.shape[0]
#     n = R.shape[1]
#     R = R.reshape(T,n,1)

#     return R @ R.transpose(0,2,1)


# def get_next_ewma(EWMA, y_last, t, beta):
#     """
#     param EWMA: EWMA at time t-1
#     param y_last: observation at time t-1
#     param t: current time step
#     param beta: EWMA exponential forgetting parameter

#     returns: EWMA estimate at time t (note that this does not depend on y_t)
#     """

#     old_weight = (beta-beta**t)/(1-beta**t)
#     new_weight = (1-beta) / (1-beta**t)

#     return old_weight*EWMA + new_weight*y_last

# def get_ewmas(y, T_half):
#     """
#     y: array with measurements for times t=1,2,...,T=len(y)
#     T_half: EWMA half life

#     returns: list of EWMAs for times t=2,3,...,T+1 = len(y)


#     Note: We define EWMA_t as a function of the
#     observations up to time t-1. This means that
#     y = [y_1,y_2,...,y_T] (for some T), while
#     EWMA = [EWMA_2, EWMA_3, ..., EWMA_{T+1}]
#     This way we don't get a "look-ahead bias" in the EWMA
#     """

#     beta = np.exp(-np.log(2)/T_half)
#     EWMA_t = 0
#     EWMAs = []
#     for t in range(1,y.shape[0]+1): # First EWMA is for t=2
#         y_last = y[t-1] # Note zero-indexing
#         EWMA_t = get_next_ewma(EWMA_t, y_last, t, beta)
#         EWMAs.append(EWMA_t)
#     return np.array(EWMAs)
