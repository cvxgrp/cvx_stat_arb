# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from time import time

import cvxpy as cvx
import numpy as np
import pandas as pd

from tests.vector import State as StateVector


class State:
    def __init__(self, prices, num=2, P_max=None):
        self.prices = prices
        self.mu = cvx.Variable((1, num), name="mu", nonneg=True)
        self.s = cvx.Variable((self.shape[1], num), name="s")
        self.P_max = P_max or 10
        # self._grad = np.zeros((num,self.shape[0]))
        self._grad = cvx.Parameter((num, self.shape[0]))

    @property
    def shape(self):
        return self.prices.shape

    @property
    def price_centered(self):
        return self.prices @ self.s - np.ones((self.shape[0], 1)) @ self.mu

    def iteration(self, num=10, **kwargs):
        # self.update_grad(self.price_centered.value.T)

        objective = cvx.Maximize(cvx.trace(self._grad @ self.price_centered))
        constraints = [
            cvx.abs(self.price_centered) <= 1.0,
            self.prices @ cvx.abs(self.s) <= self.P_max,
        ]
        prob = cvx.Problem(objective, constraints)

        for _ in range(num):
            self.update_grad(self.price_centered.value.T)
            prob.solve(**kwargs, ignore_dpp=True)
            print(prob.value)

        return self

    def update_grad(self, pk):
        """
        param pk: Tx1 array of current portfolio evolution

        returns the gradient of g at pk
        """
        pk = np.atleast_2d(pk)

        self._grad.value = np.zeros(self._grad.shape)

        self._grad.value[:, 0] = pk[:, 0] - pk[:, 1]
        self._grad.value[:, -1] = pk[:, -1] - pk[:, -2]
        self._grad.value[:, 1:-1] = 2 * pk[:, 1:-1] - pk[:, :-2] - pk[:, 2:]

        self._grad.value = 2 * self._grad.value


if __name__ == "__main__":
    file = Path(__file__).parent / "resources" / "price.csv"

    # period for training
    n = 500
    num = 20
    # extract prices
    prices = pd.read_csv(file, parse_dates=True, index_col=0, header=0)
    prices = prices.ffill()
    prices = prices.head(n=n)
    prices = prices.values

    mu_init = np.random.rand(1, num)
    s_init = np.random.randn(prices.shape[1], num)

    t = time()

    state = State(prices=prices, num=num, P_max=5)

    state.mu.value = mu_init
    state.s.value = s_init

    state = state.iteration(num=10)

    print(time() - t)

    ## iterate per statarb
    t = time()
    for i_mu, i_s in zip(mu_init.T, s_init.T):
        state = StateVector(prices=prices, P_max=5)

        # init mu and s
        state.mu.value = i_mu[0]
        state.s.value = i_s

        state = state.iteration(num=10)

    print(time() - t)
