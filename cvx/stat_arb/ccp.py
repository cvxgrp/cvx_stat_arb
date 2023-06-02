# -*- coding: utf-8 -*-
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

from cvx.stat_arb.metrics import Metrics
from cvx.stat_arb.portfolio import build_portfolio


def evaluate_solver(prices, P_max, spread_max, solver, seed=1, M=None):
    np.random.seed(seed)

    if M is not None:
        # Find M random assets
        assets = np.random.choice(prices.columns, M, replace=False)
        prices = prices[assets]

    state = _State(prices, P_max=P_max, spread_max=spread_max, solver=solver)
    state.reset()
    state.iterate()
    return state.prob


def construct_stat_arbs(
    prices,
    K=1,
    P_max=None,
    spread_max=1,
    s_init=None,
    mu_init=None,
    seed=None,
    M=None,
    solver="MOSEK",
    verbose=True,
):
    np.random.seed(1)

    all_seeds = list(np.random.choice(range(10 * K), K, replace=False))

    all_args = zip(
        [prices] * K,
        [P_max] * K,
        [spread_max] * K,
        [s_init] * K,
        [mu_init] * K,
        all_seeds,
        [M] * K,
        [solver] * K,
    )

    pool = mp.Pool()
    all_stat_arbs = []
    if verbose:
        iterator = tqdm(
            pool.imap_unordered(construct_stat_arb_helper, all_args), total=K
        )
    else:
        iterator = pool.imap_unordered(construct_stat_arb_helper, all_args)
    for stat_arb in iterator:
        all_stat_arbs.append(stat_arb)
    pool.close()
    pool.join()

    return all_stat_arbs


def construct_stat_arb_helper(args):
    """
    Call this when using of imap_unordered in multiprocessing

    param args: tuple of arguments to pass to construct_stat_arb
    """
    return construct_stat_arb(*args)


def construct_stat_arb(
    prices,
    P_max=None,
    spread_max=1,
    s_init=None,
    mu_init=None,
    seed=None,
    M=None,
    solver="MOSEK",
):
    if seed is not None:
        np.random.seed(seed)

    # Drop nan columns in prices; allows for missing data
    prices = prices.dropna(axis=1)

    if M is not None:
        # Find M random assets
        assets = np.random.choice(prices.columns, M, replace=False)
        prices = prices[assets]

    # Scale prices; remember to scale back later
    P_bar = prices.mean(axis=0).values.reshape(-1, 1)
    prices = prices / P_bar.T

    state = _State(prices, P_max=P_max, spread_max=spread_max, solver=solver)
    if s_init is None or mu_init is None:
        state.reset()
    else:
        state.s.value = s_init
        state.mu.value = mu_init

    obj_old, obj_new = 1, 10
    i = 0
    while np.linalg.norm(obj_new - obj_old) / obj_old > 1e-3 and i < 5:
        state.iterate()
        if state.prob.status == "optimal":
            obj_old = obj_new
            obj_new = state.prob.value
        else:
            print("Solver failed... resetting")
            obj_old, obj_new = 1, 10
        i += 1

    # Second pass
    if P_max is not None:
        s_times_P_bar = np.abs(state.s.value) * state.P_bar
        s_at_P_bar = np.abs(state.s.value).T @ state.P_bar
        non_zero_inds = np.where(np.abs(s_times_P_bar) > 1e-2 * s_at_P_bar)[0]

        # Scale back prices for second pass
        prices = prices * P_bar.T
        prices_new = prices.iloc[:, non_zero_inds]

        s_init = state.s.value[non_zero_inds]
        mu_init = state.mu.value

        return construct_stat_arb(
            prices_new,
            P_max=None,
            spread_max=spread_max,
            s_init=s_init,
            mu_init=mu_init,
            seed=None,
        )

    # Scale s and return stat arb
    state.s.value = state.s.value / P_bar
    stat_arb = state.build()

    return stat_arb


class State_vectorized:
    """
    Helper class for constructing stat arb using the convex-concave procedure\
        in a vectorized manner
    """

    def __init__(self, prices, K, P_max=None, spread_max=1):
        self.T, self.n = prices.shape
        self.K = K
        self.s = cp.Variable((self.n, self.K), name="s")
        self.mu = cp.Variable((1, self.K), name="mu", nonneg=True)
        self.p = cp.Variable((self.T, self.K), name="p")
        self.P_max = P_max  # allow for P_max=0 for second pass
        self.prices = prices
        self.spread_max = spread_max
        self.P_bar = prices.mean(axis=0).values.reshape(-1, 1)
        # self.prices = self.prices / self.P_bar.T

        # Construct linearized convex-concave problem
        self.grad_g = cp.Parameter((self.T, self.K), name="grad_g")

        self.obj = cp.Maximize(cp.trace(self.grad_g.T @ self.p))

        self.cons = [cp.abs(self.p - self.mu) <= self.spread_max]
        self.cons += [self.p == self.prices.values @ self.s]
        if self.P_max is not None:
            self.cons += [cp.abs(self.s).T @ self.P_bar <= self.P_max]

        self.prob = cp.Problem(self.obj, self.cons)

        # Solve once for speedup later
        print(1)
        self.grad_g.value = np.zeros((self.T, self.K))
        self.prob.solve(solver="MOSEK", verbose=False)
        print(2)

        # For debugging
        self.solve_times = []

    @property
    def assets(self):
        return list(self.prices.columns)

    @property
    def shape(self):
        return self.prices.shape

    def reset(self):
        """
        Resets to random feasible point
        """
        self.s = cp.Variable((self.n, self.K), name="s")

        s = np.random.normal(0, 1, (self.n, self.K))
        s_at_P_bar = np.abs(s).T @ self.P_bar  # will be Kx1; s is nxK
        if self.P_max is not None:
            s = s / s_at_P_bar.T * self.P_max  # scale s to be feasible

        mu = np.abs(np.random.normal(0, 1, size=(1, self.K)))

        scale = np.abs(self.prices.values @ s - mu).max()

        s = s / scale
        mu = mu / scale

        self.s.value = s
        self.mu.value = mu

    def iterate(self, solver="MOSEK"):
        """
        Performs one iteration of the convex concave procedure
        """

        # Update p_centered and grad_g
        p = self.prices.values @ self.s.value
        self.grad_g.value = self._get_grad_g(p)

        # Solve problem
        start = time.time()
        self.prob.solve(solver=solver, verbose=False, ignore_dpp=True)
        end = time.time()
        print(end - start)
        self.solve_times.append(end - start)

        return self

    @staticmethod
    def _get_grad_g(pk):
        """
        param pk: TxK array of current portfolio evolutions

        returns the gradients of g at pk
        """
        grad_g = np.zeros(pk.shape)
        grad_g[0, :] = pk[0, :] - pk[1, :]
        grad_g[-1, :] = pk[-1, :] - pk[-2, :]
        grad_g[1:-1, :] = 2 * pk[1:-1, :] - pk[:-2, :] - pk[2:, :]

        return 2 * grad_g

    def build(self):
        assets_dict = dict(zip(self.assets, self.s.value))
        stat_arb = StatArb(assets=assets_dict, mu=self.mu.value)

        return stat_arb


class _State:
    """
    Helper class for constructing stat arb using the convex-concave procedure
    """

    def __init__(self, prices, P_max=None, spread_max=1, solver="MOSEK"):
        self.T, self.n = prices.shape
        self.s = cp.Variable((self.n, 1), name="s")
        self.mu = cp.Variable(name="mu", nonneg=True)
        self.p = cp.Variable((self.T, 1), name="p")
        self.P_max = P_max  # allow for P_max=0 for second pass
        self.prices = prices
        self.spread_max = spread_max
        self.P_bar = prices.mean(axis=0).values.reshape(-1, 1)
        self.solver = solver
        # self.prices = self.prices / self.P_bar.T

        # Construct linearized convex-concave problem
        self.grad_g = cp.Parameter((self.T, 1), name="grad_g")

        self.obj = cp.Maximize(self.grad_g.T @ self.p)

        self.cons = [cp.abs(self.p - self.mu) <= self.spread_max]
        self.cons += [self.p == self.prices.values @ self.s]
        if self.P_max is not None:
            self.cons += [cp.abs(self.s).T @ self.P_bar <= self.P_max]

        self.prob = cp.Problem(self.obj, self.cons)

        # Solve once for speedup later
        self.grad_g.value = np.ones((self.T, 1))
        self.prob.solve(solver=self.solver, verbose=False)

        # For debugging
        self.solve_times = []

    @property
    def assets(self):
        return list(self.prices.columns)

    @property
    def shape(self):
        return self.prices.shape

    def reset(self):
        """
        Resets to random feasible point
        """
        s = np.random.normal(0, 1, (self.n, 1))
        s_at_P_bar = np.abs(s).T @ self.P_bar
        if self.P_max is not None:
            s = s / s_at_P_bar * self.P_max  # scale s to be feasible

        mu = np.abs(np.random.normal(0, 1))

        scale = np.abs(self.prices.values @ s - mu).max()

        s = s / scale
        mu = mu / scale

        self.s.value = s
        self.mu.value = mu

    def iterate(self):
        """
        Performs one iteration of the convex concave procedure
        """

        # Update p_centered and grad_g
        p = self.prices.values @ self.s.value
        self.grad_g.value = self._get_grad_g(p)

        # Solve problem
        start = time.time()
        try:
            self.prob.solve(solver=self.solver, verbose=False)
        except Exception:
            print("Solver failed, resetting...")
            self.reset()

        end = time.time()
        self.solve_times.append(end - start)

        return self

    @staticmethod
    def _get_grad_g(pk):
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
        assets_dict = dict(zip(self.assets, self.s.value))
        stat_arb = StatArb(assets=assets_dict, mu=self.mu.value)

        return stat_arb


@dataclass(frozen=True)
class StatArbGroup:
    """
    Stores a group of stat arb objects
    """

    stat_arbs: list

    def metrics(self, prices: pd.DataFrame, cutoff: float = 1):
        all_profits_daily = []
        for stat_arb in self.stat_arbs:
            m = stat_arb.metrics(prices, cutoff)
            if m is not None:
                all_profits_daily.append(m.daily_profit)
        # do outer concat
        all_profits_daily = pd.concat(all_profits_daily, axis=1)
        profits_daily = all_profits_daily.sum(axis=1)
        profits_daily[0] = np.nan
        m = Metrics(daily_profit=profits_daily)

        return m

    def validate(
        self,
        prices_val: pd.DataFrame,
        prices_train_val: pd.DataFrame,
        cutoff: float = 1,
        SR_cutoff: float = 3,
    ):
        """
        validates stat arbs on validation set (prices_val) and refits on
        train+validation (prices_train_val)

        param prices_val: validation set
        param prices_train_val: train+validation set
        param cutoff: max deviance from stat arb mean
        param SR_cutoff: cutoff for sharpe ratio
        """
        stat_arbs_success = []
        assets = []

        for stat_arb in self.stat_arbs:
            if set(stat_arb.asset_names) not in assets:
                if stat_arb.validate(prices_val, cutoff, SR_cutoff):
                    stat_arb_refit = stat_arb.refit(
                        prices_train_val[stat_arb.asset_names]
                    )
                    stat_arbs_success.append(stat_arb_refit)
                    assets.append(set(stat_arb.asset_names))

        return StatArbGroup(stat_arbs_success)

    def construct_porfolio(self, prices: pd.DataFrame, cutoff: float = 1):
        """
        Constructs portfolio from stat arbs
        """

        # If no stat arbs, return None, i.e., no portfolio
        if not self.stat_arbs:
            return None

        # Initialize portfolio
        positions0 = self.stat_arbs[0].get_positions(prices, cutoff=cutoff)
        portfolio = build_portfolio(prices, positions=positions0)

        # Add other stat arbs to portfolio
        for stat_arb in self.stat_arbs[1:]:
            positions = stat_arb.get_positions(prices, cutoff=cutoff)
            portfolio += build_portfolio(prices, positions=positions)

        return portfolio


@dataclass(frozen=True)
class StatArb:
    """
    Stat arb class
    """

    assets: dict
    mu: float

    def __setitem__(self, __name: int, __value: float) -> None:
        self.assets[__name] = __value

    def __getitem__(self, key: int) -> float:
        return self.assets.__getitem__(key)

    def items(self):
        return self.assets.items()

    def evaluate(self, prices: pd.DataFrame):
        value = 0
        for asset, position in self.items():
            value += prices[asset] * position
        return value

    def refit(self, prices: pd.DataFrame):
        """ "
        returns refitted stat arb
        """

        return construct_stat_arb(prices, s_init=self.s, mu_init=self.mu)

    @property
    def s(self):
        """
        returns the vector of positions
        """
        return np.array(list(self.assets.values())).reshape(-1, 1)

    @property
    def asset_names(self):
        """
        returns list of assets in StatArb
        """
        return list(self.assets.keys())

    @property
    def n(self):
        """
        returns the number of assets
        """
        return len(self.assets)

    def get_q(self, prices: pd.DataFrame, cutoff, exit_last=True):
        """
        returns the vector of investments in stat arb based on trading strategy\
            q_t = mu - p_t until |p_t-mu| <= cutoff\
                rest is zero
        """
        p = self.evaluate(prices)
        q = self.mu - p
        q.name = "q"
        breaches = np.where(np.abs(p - self.mu) >= cutoff)[0]
        if len(breaches) == 0:
            if exit_last:
                q[-1] = 0
            return q
        else:
            first_breach = breaches[0]
            q[first_breach:] = 0
            return q

    def get_positions(self, prices: pd.DataFrame, cutoff, exit_last=True):
        """
        computes the positions of each individual asset over time\
            based on trading strategy\
                q_t = mu - p_t until |p_t-mu| <= cutoff\
                rest is zero
                positions = q_t*self.s

        returns Txn, pandas DataFrame
        """

        q = self.get_q(prices, cutoff, exit_last)
        q = pd.concat([q] * self.n, axis=1)
        s = self.s
        positions = q * (s.T)
        positions.columns = self.asset_names
        return positions

    def validate(self, prices, cutoff, SR_target=None):
        m = self.metrics(prices, cutoff)

        if m is None:
            return False

        if SR_target is not None:
            if m.sr_profit < SR_target:
                return False

        return True

    def metrics(self, prices: pd.DataFrame, cutoff: float = 1, exit_last: bool = True):
        """
        Computes metrics of stat arbs trading strategy\
            q_t = mu - p_t until |p_t-mu| >= cutoff
        """
        # Get price evolution of portfolio

        p = self.evaluate(prices)

        q = self.get_q(prices, cutoff, exit_last=exit_last)
        if q[0] == 0:  # We never enter a position
            return None

        price_changes = p.ffill().diff()
        previous_position = q.shift(1)
        profits = previous_position * price_changes

        # Set first row to NaN
        profits.iloc[0] = np.nan
        return Metrics(daily_profit=profits)

        # profits_daily = port_values1-port_values0

        m = Metrics(daily_profit=profits_daily)
        return m


# TODO: working on this...
class StatArbPortfolioManager:
    def __init__(
        self,
        prices,
        start_date,
        end_date,
        train_len,
        val_len,
        test_len,
        P_max=10,
        n_candidates=100,
    ):
        """
        param prices: Txn, price matrix, pandas DataFrame
        param P_max: scalar, max value of abs(s)@P_bar,\
            where P_bar is the mean of P (row-wise)
        param zero_inds: list of indices of s that should be zero
        param n_candidates: int, number of candidates to try
        param n_passes: int, number of passes to try
        """
        self.prices = prices
        self.start_date = start_date
        self.end_date = end_date

        self.train_len = pd.Timedelta(train_len)
        self.val_len = pd.Timedelta(val_len)
        self.test_len = pd.Timedelta(test_len)

        self.P_max = P_max
        self.n_candidates = n_candidates

        self.portfolios = []
        self.assets = []
        self.obj_vals = []

    def run(self, update_freq=10):
        """
        rund backtest, managing a portfolio of stat arbs, looking for new stat arbs every update_freq days
        """
        update_freq = pd.Timedelta(update_freq)

        prices_train = self.prices.loc[
            self.start_date : self.start_date + self.train_len
        ]
        prices_val = self.prices.loc[
            self.start_date
            + self.train_len : self.start_date
            + self.train_len
            + self.val_len
        ]
        prices_test = self.prices.loc[
            self.start_date + self.train_len + self.val_len : self.end_date
        ]

        # Update start date
        self.start_date = self.start_date + update_freq

        # Get initial stat arb portfolios
        stat_arbs = self.get_stat_arb_portfolios(prices_train)

        # Construct initial portfolio

    # def construct_portfolio(stat_arbs):
    #     cutoff = 1.05

    #     pos0 = traded_stat_arbs[0].get_positions(data_test, cutoff=cutoff,\
    #         exit_last=True)
    #     portfolio = build_portfolio(data_test, positions=pos0)

    #     for stat_arb in traded_stat_arbs[1:]:
    #         pos_temp = stat_arb.get_positions(data_test, cutoff=cutoff, exit_last=True)
    #         portfolio += build_portfolio(data_test, positions=pos_temp)

    def get_stat_arb_portfolios(self, prices_train):
        """
        returns a list of stat arb portfolios
            a stat arb portfolio is a StatArb class instance
        """
        all_data_train = [prices_train] * self.n_candidates
        all_P_max = [self.P_max] * self.n_candidates
        all_zero_inds = [None] * self.n_candidates
        all_i = [i for i in range(self.n_candidates)]
        all_n_candidates = [self.n_candidates] * self.n_candidates
        all_p_init = [None] * self.n_candidates
        all_s_init = [None] * self.n_candidates
        all_mu_init = [None] * self.n_candidates
        all_seeds = [None] * self.n_candidates

        pool = mp.Pool()
        all_stat_arbs = pool.starmap(
            construct_stat_arb,
            zip(
                all_data_train,
                all_P_max,
                all_zero_inds,
                all_i,
                all_n_candidates,
                all_p_init,
                all_s_init,
                all_mu_init,
                all_seeds,
            ),
        )
        pool.close()
        pool.join()

        return all_stat_arbs
