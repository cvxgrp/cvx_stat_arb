from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np
import cvxpy as cvx


class State:
    def __init__(self, prices, P_max=None):
        n,m = prices.shape
        self.mu = cvx.Variable(name="mu", nonneg=True)
        self.s = cvx.Variable(m, name="s")
        self.P_max = P_max or 10
        self.prices = prices
        # self._grad = np.zeros(n)
        self._grad = cvx.Parameter(n)

    @property
    def shape(self):
        return self.prices.shape


    @property
    def p_centered(self):
        return self.prices @ self.s - self.mu


    def iteration(self, num=10, **kwargs):

        objective = cvx.Maximize(self._grad @ self.p_centered)
        constraints = [cvx.abs(self.p_centered) <= 1.0, self.prices @ cvx.abs(self.s) <= self.P_max]
        prob = cvx.Problem(objective, constraints)

        for _ in range(num):
            self.update_grad(self.p_centered.value)
            prob.solve(**kwargs)

        return self

    def update_grad(self, pk):
        """
        param pk: Tx1 array of current portfolio evolution

        returns the gradient of g at pk
        """
        #grad_g = np.zeros(pk.shape)
        self._grad.value = np.zeros(self._grad.shape)
        self._grad.value[0] = pk[0]-pk[1]
        self._grad.value[-1] = pk[-1]-pk[-2]
        self._grad.value[1:-1] = 2*pk[1:-1] - pk[:-2] - pk[2:]
        self._grad.value = 2*self._grad.value

        #return 2*self._grad

        #return 2*grad_g


if __name__ == '__main__':
    file = Path(__file__).parent / "resources" / "price.csv"

    # period for training
    n = 50

    # extract prices
    prices = pd.read_csv(file, parse_dates=True, index_col=0, header=0)
    prices = prices.ffill()
    prices = prices.head(n=n)
    prices = prices.values

    state = State(prices=prices, P_max=5)

    # init mu and s
    state.mu.value = 0.0
    state.s.value = np.random.randn(state.shape[1])

    # do a single iteration
    state = state.iteration(num=10, verbose=True)

    logger.info(state.s.value)
    logger.info(state.mu.value)
