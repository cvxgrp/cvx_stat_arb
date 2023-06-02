# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import cvxpy as cvx
import numpy as np
import pandas as pd


class State:
    def __init__(self, prices, spread_max=1, P_max=None):
        n, m = prices.shape
        self.mu = cvx.Variable(name="mu", nonneg=True)
        self.s = cvx.Variable(m, name="s")
        self.P_max = P_max or 10
        self.prices = prices
        self.spread_max = spread_max

    @property
    def shape(self):
        return self.prices.shape

    def iteration(self):
        p_centered = self.prices @ self.s - self.mu
        grad = State._get_grad_g2(p_centered.value)

        objective = cvx.Maximize(grad @ p_centered)

        # TODO: I think constraints are wrong; it should be self.P_bar @ cvx.abs(self.s) not self.prices @ cvx.abs(self.s) ?
        prob = cvx.Problem(
            objective,
            constraints=[
                cvx.abs(p_centered) <= self.spread_max,
                self.prices @ cvx.abs(self.s) <= self.P_max,
            ],
        )
        prob.solve(cvx.ECOS)
        return self

    @staticmethod
    def _get_grad_g2(pk):
        """
        param pk: Tx1 array of current portfolio evolution

        returns the gradient of g at pk
        """
        grad_g = np.zeros(pk.shape)
        grad_g[0] = pk[0] - pk[1]
        grad_g[-1] = pk[-1] - pk[-2]
        grad_g[1:-1] = 2 * pk[1:-1] - pk[:-2] - pk[2:]

        return 2 * grad_g

    def build(self):
        mu = self.mu.value
        s = self.s.value
        # return StatArb(mu, s)


if __name__ == "__main__":
    file = Path(__file__).parent / "resources" / "price.csv"

    # period for training
    n = 50

    # extract prices
    prices = pd.read_csv(file, parse_dates=True, index_col=0, header=0)
    prices = prices.ffill()
    prices = prices.head(n=n)
    prices = prices.values

    results = list()
    state = State(prices=prices, P_max=1000000, spread_max=10000)

    for i in range(1000):
        # init mu and s
        state.mu.value = 0.0
        state.s.value = np.random.randn(state.shape[1])

        # do a single iteration
        # state = state.iteration()

        # do like 20 iterations
        for i in range(20):
            state = state.iteration()

        print(state.s.value)
        print(state.mu.value)

        # x is some sort of statarb
        x = state.build()

        # results = list()
        if x.validate(prices):
            # profit = x.test(some other prices)
            results.append(x)

    # results is a list of validated stat arbs
