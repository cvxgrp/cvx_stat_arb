from dataclasses import dataclass
import numpy as np
import pandas as pd
import cvxpy as cp
from functools import partial
import multiprocessing as mp
from tqdm import tqdm, trange
import time
import pickle
from sklearn.linear_model import Lasso

import warnings
warnings.filterwarnings("ignore")



from cvx.simulator.metrics import Metrics
from cvx.simulator.portfolio import EquityPortfolio
from cvx.stat_arb.ar_model import ar

with open('../data/sector_to_asset.pkl', 'rb') as f:
    sector_to_asset = pickle.load(f)
with open('../data/asset_to_sector.pkl', 'rb') as f:
    asset_to_sector = pickle.load(f)


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

def construct_stat_arbs_parallel(args, **kwargs):
    """
    Call this when using of imap_unordered in multiprocessing

    param args: tuple
    """
    return construct_stat_arbs(*args, **kwargs)

def construct_stat_arbs(
    prices,
    K=1,
    P_max=None,
    spread_max=1,
    s_init=None,
    mu_init=None,
    seed=None,
    M=None,
    parallel=False,
    **kwargs
):
    """ 
    Construct stat arbs by solving approximately solving variance maximizing
    mean reversion problem using the convex-concave procedure

    param prices: pd.DataFrame of prices
    param K: number of stat arbs to construct
    param P_max: maximum position size
    param spread_max: maximum deviation from mean in optimiation problem
    param s_init: initial position vector
    param mu_init: initial mu vector
    param seed: random seed
    param M: number of assets (size of universe) to consider in each stat arb\
        a random subet of size M is chosen from the universe of assets for each\
        of the K stat arbs
        if M is None, then all assets are considered
    param solver: solver to use in cvxpy
    param verbose: whether to print progress bar
    """

    if seed is not None:
        np.random.seed(seed)

    all_seeds = list(np.random.choice(range(10 * K), K, replace=False))

    all_args = zip(
        [prices] * K,
        [P_max] * K,
        [spread_max] * K,
        [s_init] * K,
        [mu_init] * K,
        all_seeds,
        [M] * K,
    )

    all_stat_arbs = []

    # TODO: remove parallelization??? 
    if parallel:
        pool = mp.Pool()
        # if verbose_arb:
        iterator = tqdm(
            pool.imap(partial(_construct_stat_arbs_helper, **kwargs), all_args), total=K
        )
        # else:
        #     iterator = pool.imap(partial(_construct_stat_arbs_helper, **kwargs), all_args)
        for stat_arb in iterator:
            all_stat_arbs.append(stat_arb)

        pool.close()
        pool.join()

    else:
        for i in range(K):
            all_stat_arbs.append(
                _construct_stat_arb(
                    prices,
                    P_max=P_max,
                    spread_max=spread_max,
                    s_init=s_init,
                    mu_init=mu_init,
                    seed=all_seeds[i],
                    M=M,
                    **kwargs
                )
            )
    

    return StatArbGroup(all_stat_arbs)


def _construct_stat_arbs_helper(args, **kwargs):
    """
    Call this when using of imap_unordered in multiprocessing

    param args: tuple of arguments to pass to construct_stat_arb
    """
    return _construct_stat_arb(*args, **kwargs)


def _construct_stat_arb(
    prices,
    P_max=None,
    spread_max=1,
    s_init=None,
    s_init_orig=None,
    mu_init=None,
    mu_init_orig=None,
    assets_init=None,
    seed=None,
    M=None,
    **kwargs
):
    if seed is not None:
        np.random.seed(seed)

    ### TODO: Test random porfolio construction; uncomment below
    # Three random assets
    # assets = np.random.choice(prices.columns, 3, replace=False)
    # prices = prices[assets]
    # P_bar = prices.mean(axis=0).values.reshape(-1, 1)
    # s = np.random.normal(0, 1, (3, 1))
    # s_at_P_bar = np.abs(s).T @ P_bar
    # mu = np.abs(np.random.normal(0, 1))

    # s_at_P_bar = np.abs(s).T @ P_bar
    # s = s / s_at_P_bar * P_max  # scale s to be feasible

    # scale = np.abs(prices.values @ s - mu).max()

    # s = s / scale
    # mu = mu / scale

    # # print(mu)
    # assets_dict = {assets[i]:s[i] for i in range(3)}

    # stat_arb = StatArb(assets=assets_dict, mu=mu)
    # return stat_arb

    # Drop nan columns in prices; allows for missing data
    prices = prices.dropna(axis=1)

    ### TODO: How to choose M assets?

    # if M is not None:
        # Find M random assets
        # Sectors
        # Choose sector with probability proportional to number of assets in
        # sector
        # sector_sizes = np.array([len(v) for v in sector_to_asset.values()])
        # sector_probs = sector_sizes / sector_sizes.sum()
        # sector = np.random.choice(list(sector_to_asset.keys()), 1, p=sector_probs)[0]

        

        # sector = np.random.choice(list(sector_to_asset.keys()), 1)[0]
        # assets = sector_to_asset[sector]
        # while len(assets)<M:
        #     pass
        #     sector = np.random.choice(list(sector_to_asset.keys()), 1)[0]
            # sector = np.random.choice(list(sector_to_asset.keys()), 1, p=sector_probs)[0]
        #     assets = sector_to_asset[sector]



            # assets += sector_to_asset[sector]
        #     assets = list(set(assets))
        # if len(assets)>M:
        #     assets = np.random.choice(assets, M, replace=False)

        # TODO: Hardcode for Mining, Quarrying, and Oil and Gas Extraction
        # assets = sector_to_asset['Manufacturing']
        # assets = np.random.choice(assets, M, replace=False)

        # assets = np.random.choice(prices.columns, M, replace=False)
        # prices = prices[assets]

        # Chooose M assets from universe
        # assets = np.random.choice(prices.columns, M, replace=False)
        # assets = sector_to_asset["Manufacturing"]
        # assets = np.random.choice(assets, M, replace=False)
        # prices = prices[assets]

    # Scale prices; remember to scale back later
    P_bar = prices.mean(axis=0).values.reshape(-1, 1)
    prices = prices / P_bar.T

    state = _State(prices, P_max=P_max, spread_max=spread_max, **kwargs)
    if s_init is None or mu_init is None:
        state.reset()

        s_times_P_bar = np.abs(state.s.value) * state.P_bar
        s_at_P_bar = np.abs(state.s.value).T @ state.P_bar
        non_zero_inds = np.where(np.abs(s_times_P_bar) > 1e-2 * s_at_P_bar)[0]

        assets_init = prices.columns[non_zero_inds]
        prices = prices.iloc[:,non_zero_inds]
        P_bar = P_bar[non_zero_inds]

        s_init = state.s_init[non_zero_inds]
        mu_init = state.mu_init
        state = _State(prices, P_max=state.P_max, spread_max=state.spread_max, s_init=s_init, mu_init=mu_init, assets_init=assets_init, **state.kwargs)
    else:
        state.s.value = s_init
        state.mu.value = mu_init
        state.s_init = s_init_orig
        state.mu_init = mu_init_orig
        state.assets_init = assets_init

    obj_old, obj_new = 1, 10
    i = 0
    while np.linalg.norm(obj_new - obj_old) / obj_old > 1e-3: # and i < 2:
        state = state.iterate()
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

        return _construct_stat_arb(
            prices_new,
            P_max=None,
            spread_max=spread_max,
            s_init=s_init,
            s_init_orig=state.s_init,
            mu_init=mu_init,
            mu_init_orig=state.mu_init,
            assets_init=state.assets_init,
            seed=None,
            **kwargs
        )

    # Scale s and return stat arb
    state.s.value = state.s.value / P_bar
    stat_arb = state.build()

    return stat_arb


class _State:
    """
    Helper class for constructing stat arb using the convex-concave procedure
    """

    def __init__(self, prices, P_max=None, spread_max=1, s_init=None, mu_init=None, assets_init=None, **kwargs):
        self.T, self.n = prices.shape

        self.P_max = P_max  # allow for P_max=None for second pass
        self.prices = prices
        self.spread_max = spread_max
        self.s_init = s_init
        self.mu_init = mu_init
        self.assets_init = assets_init
        self.kwargs = kwargs


        self.construct_problem()

        # For debugging
        self.solve_times = []

    def construct_problem(self):
        """
        Constructs the convex-concave problem
        """
        self.P_bar = self.prices.mean(axis=0).values.reshape(-1, 1)

        # Construct linearized convex-concave problem
        self.s = cp.Variable((self.n, 1), name="s")
        self.mu = cp.Variable(name="mu", nonneg=True)
        self.p = cp.Variable((self.T, 1), name="p")

        self.grad_g = cp.Parameter((self.T, 1), name="grad_g")

        self.obj = cp.Maximize(self.grad_g.T @ self.p)

        self.cons = [cp.abs(self.p - self.mu) <= self.spread_max]
        self.cons += [self.p == self.prices.values @ self.s]
        if self.P_max is not None:
            self.cons += [cp.abs(self.s).T @ self.P_bar <= self.P_max]

        self.prob = cp.Problem(self.obj, self.cons)

        # Solve once for speedup later
        self.grad_g.value = np.ones((self.T, 1))
        # self.prob.solve(solver=self.solver, verbose=False)
        self.prob.solve(**self.kwargs)

        self.s.value = self.s_init
        self.mu.value = self.mu_init

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
        ### Burte force cointegration init
        # Select random column
        i = np.random.choice(self.n)
        pi = self.prices.iloc[:,i]
        P_minusi = self.prices.drop(columns=self.assets[i])

        # TODO: Hyperparameter
        alpha = 0.00001
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
        model.fit(P_minusi, pi);

        s_minusi = model.coef_.reshape(-1, 1)
        s = np.insert(s_minusi, i, -1, axis=0)
        mu = np.mean(pi - model.predict(P_minusi))

        scale = np.abs(self.prices.values @ s - mu).max()

        s = s / scale
        mu = mu / scale

        s_init_orig = s
        mu_init_orig = mu

        # self.s.value = s
        # self.mu.value = mu
        # self.s_init = s
        # self.mu_init = self.mu

        # nonzero_inds = np.abs(s) >= 0.01*np.max(np.abs(s))
        # s = s[nonzero_inds]
        # prices_new = self.prices.iloc[:, nonzero_inds]

        # return _State(prices_new, P_max=self.P_max, spread_max=self.spread_max,s_init =s_init_orig, mu_init=mu_init_orig,  **self.kwargs)
        self.s.value = s
        self.mu.value = mu
        self.s_init = s_init_orig
        self.mu_init = mu_init_orig




        # print((np.abs(model.coef_) >= 0.01*np.max(np.abs(model.coef_))).sum())





        return
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
            # self.prob.solve(solver=self.solver, verbose=False)
            self.prob.solve(**self.kwargs)
        except Exception:
            print("Solver failed, resetting...")
            self.reset()

        end = time.time()
        self.solve_times.append(end - start)

        return self

    @staticmethod
    def _get_grad_g(pk):
        """
        param pk: Tx1 array of current stat arb price evolution

        returns the gradient of g at pk
        """
        grad_g = np.zeros(pk.shape)
        grad_g[0] = pk[0] - pk[1]
        grad_g[-1] = pk[-1] - pk[-2]
        grad_g[1:-1] = 2 * pk[1:-1] - pk[:-2] - pk[2:]

        return 2 * grad_g

    def build(self):
        """
        Builds stat arb object
        """
        assets_dict = dict(zip(self.assets, self.s.value.flatten()))
        stat_arb = StatArb(assets=assets_dict, mu=self.mu.value, s_init=self.s_init, mu_init=self.mu_init, assets_init=self.assets_init)

        return stat_arb

def _flatten(list_of_lists):
    return list(set([item for sublist in list_of_lists for item in sublist]))


@dataclass(frozen=True)
class StatArbGroup:
    """
    Stores a group of stat arb objects
    """

    stat_arbs: list

    @property
    def assets_names(self):
        return _flatten([stat_arb.asset_names for stat_arb in self.stat_arbs])

    def metrics(self, prices: pd.DataFrame, cutoff: float = 1):
        all_profits_daily = []
        for stat_arb in self.stat_arbs:
            m = stat_arb.metrics(prices, cutoff)
            if m is not None:
                all_profits_daily.append(m.daily_profit)
        # do outer concat
        all_profits_daily = pd.concat(all_profits_daily, axis=1)
        profits_daily = all_profits_daily.sum(axis=1)
        # profits_daily[0] = np.nan
        m = Metrics(daily_profit=profits_daily)

        return m

    def validate(
        self,
        prices_val: pd.DataFrame,
        prices_train_val: pd.DataFrame,
        cutoff_up: float = 1,
        cutoff_down: float = None,
        SR_cutoff: float = None,
        profit_target = None,
        P_max: float = None,
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
            if (
                set(stat_arb.asset_names) not in assets
            ):  # We don't want duplicates of the same stat arb TODO: is this check too harsh?
                if stat_arb.validate(prices_val, cutoff_up, cutoff_down, SR_cutoff, profit_target=profit_target):
                    # TODO: refit or not???
                    # TODO: Currently refits with spread_max=1
                    # stat_arb_refit = stat_arb.refit(
                    #     prices_train_val[stat_arb.asset_names],
                    #     P_max=P_max
                    # )
                    stat_arb_refit = stat_arb
                    stat_arbs_success.append(stat_arb_refit)
                    assets.append(set(stat_arb.asset_names))

        return StatArbGroup(stat_arbs_success)

    def construct_portfolio(self, prices: pd.DataFrame, cutoff_up: float = 1,\
        cutoff_down: float=None, lin_increase=False):
        """
        Constructs portfolio from stat arbs
        """
        cutoff_down = cutoff_down or cutoff_up

        # If no stat arbs, return None, i.e., no portfolio
        if not self.stat_arbs:
            return None

        assert set(self.assets_names).issubset(
            prices.columns
        ), "Stat arb assets not in prices"

        prices = prices[self.assets_names]

        # Initialize portfolio
        portfolio = None

        # Add other stat arbs to portfolio
        for stat_arb in self.stat_arbs:
            prices_temp = prices[stat_arb.asset_names]
            stocks = stat_arb.get_positions(prices_temp, cutoff_up, 
            cutoff_down, lin_increase=lin_increase)
            if np.sum(stocks.values) == 0:
                continue
            if portfolio is None:
                portfolio = EquityPortfolio(prices_temp, stocks=stocks, initial_cash=1) # TODO: make init cash flexible

            else: 
                portfolio += EquityPortfolio(prices_temp, stocks=stocks, initial_cash=1) # TODO: make init cash flexible

        return portfolio


@dataclass(frozen=True)
class StatArb:
    """
    Stat arb class
    """

    assets: dict
    mu: float
    s_init: float = None
    mu_init: float = None
    assets_init: list = None

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

    def refit(self, prices: pd.DataFrame, P_max=None, spread_max=1):
        """ 
        returns refitted stat arb
        """

        return _construct_stat_arb(prices, s_init=self.s, mu_init=self.mu, P_max=P_max, spread_max=spread_max)

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

    @staticmethod
    def q_nonlinear(adj_prices):
        """
        Enters position +1 if adj_prices < 0.25,
        exits when adj_price crosses zero
        enters position -1 if adj_prices > 0.75,
        exits when adj_price crosses zero
        """
        q = np.zeros(len(adj_prices))
        q[adj_prices < -0.75] = 1
        q[adj_prices > 0.75] = -1

        return pd.Series(q, index=adj_prices.index)

    @staticmethod
    def q_pwl(adj_prices):
        """
        Enters position 3*adj_prices+3 if adj_prices < -0.75,
        Enters position 3*adj_prices-3 if adj_prices > 0.75,
        Enters -adj_prices if -0.75 <= adj_prices <= 0.75,
        """
        q = np.zeros(len(adj_prices))
        q[adj_prices < -0.75] = 3 * adj_prices[adj_prices < -0.75] + 3
        q[adj_prices > 0.75] = 3 * adj_prices[adj_prices > 0.75] - 3
        q[(-0.75 <= adj_prices) & (adj_prices <= 0.75)] = -adj_prices[
            (-0.75 <= adj_prices) & (adj_prices <= 0.75)
        ]

        return pd.Series(q, index=adj_prices.index)

    @staticmethod
    def q_fixed(adj_prices, N=1):
        """
        Enters position -adj_prices on first day if -1<=adj_prices<=1,
        Hold this position for N days or until adj_prices crosses +-1
        """
        q = np.zeros(len(adj_prices))
        q[0:N] = -adj_prices[0]

        first_breach = np.where(np.abs(adj_prices) > 1)[0]
        if len(first_breach) > 0:
            q[first_breach[0] :] = 0
        
        return pd.Series(q, index=adj_prices.index)

    @staticmethod
    def q_trend(adj_prices, N=5):
        """
        Enter position -adj_prices if ajd_prices is above its N-day moving average
        """
        q = np.zeros(len(adj_prices))
        rolling_mean = adj_prices.rolling(N).mean().shift(1)
        
        q[(adj_prices > rolling_mean) & (adj_prices<0)] = -adj_prices[
            (adj_prices > rolling_mean) & (adj_prices<0)
        ]
      
        q[(adj_prices < rolling_mean) & (adj_prices>0)] = -adj_prices[
            (adj_prices < rolling_mean) & (adj_prices>0)
        ]

        # Fill between positive values
        # Forward fill zeros
        q[q==0] = np.nan
        q = pd.Series(q).ffill()
        q = q.fillna(0)




        return pd.Series(q, index=adj_prices.index)


    def get_q(self, prices: pd.DataFrame, cutoff_up, cutoff_down=None, exit_last=True, lin_increase=False, validate=False):
        """
        returns the vector of investments in stat arb based on trading strategy\
            q_t = mu - p_t until |p_t-mu| <= cutoff\
                rest is zero
        """
        cutoff_down = cutoff_down or cutoff_up

        p = self.evaluate(prices)
        q = self.mu - p

        # return self.q_trend(p-self.mu)

        # if not validate:
        #     return self.q_fixed(p-self.mu)

        # q = self.q_pwl(p-self.mu)

        # TODO!!!
        # q = self.q_nonlinear(p-self.mu)

        

        q.name = "q"
        # breaches = np.where(np.abs(p - self.mu) >= cutoff)[0]
        breaches_up = np.where(p - self.mu >= cutoff_up)[0]
        breaches_down = np.where(p - self.mu <= -cutoff_down)[0]
        # if len(breaches) == 0:
        if len(breaches_up) ==0 and len(breaches_down) == 0:
            if exit_last:
                q[-1] = 0
        else:
            if len(breaches_up)>0:
                first_breach_up = breaches_up[0]
            else:
                first_breach_up = np.inf
            if len(breaches_down)>0:
                first_breach_down = breaches_down[0]
            else:
                first_breach_down = np.inf
            
            first_breach = min(first_breach_up, first_breach_down)
            q[first_breach:] = 0

        if lin_increase:
            proportions = 1-np.linspace(0, 1, len(q))
            # proportions = np.minimum(proportions, 1)
            q = q * proportions
        return q

    def get_positions(self, prices: pd.DataFrame, cutoff_up, cutoff_down=None, exit_last=True, lin_increase=False):
        """
        computes the positions of each individual asset over time\
            based on trading strategy\
                q_t = mu - p_t until |p_t-mu| <= cutoff\
                rest is zero
                positions = q_t*self.s

        returns Txn, pandas DataFrame
        """
        cutoff_down = cutoff_down or cutoff_up

        q = self.get_q(prices, cutoff_up, cutoff_down, exit_last, lin_increase=lin_increase)
        q = pd.concat([q] * self.n, axis=1)
        s = self.s
        positions = q * (s.T)
        positions.columns = self.asset_names
        return positions

    def validate(self, prices, cutoff_up, cutoff_down=None, SR_target=None, profit_target=None, return_target=10):
        """
        Validates stat arb on validation set (prices) and refits

        param prices: validation set
        param cutoff: max deviance from stat arb mean
        param SR_target: (minimum) target sharpe ratio over validation set
        """
        cutoff_down = cutoff_down or cutoff_up

        ### Make sure at least two assets
        if self.n < 2:
            return False

        # alpha, _ = ar(self, prices)
        # if alpha >= 0.95:
        #     return False

        # Check that prices statys within bounds
        adj_prices = self.evaluate(prices) - self.mu
        if np.max(adj_prices) >= cutoff_up or np.min(adj_prices) <= -cutoff_down:
            return False



                

        ### Makes sure stat arb is profitable
        m = self.metrics(prices, cutoff_up, cutoff_down, exit_last=True, validate=True)
        if m is None:
            return False
        
        # if self.

        # Highe enough Sharpe Ratio
        if SR_target is not None:
            if m.sr_profit < SR_target:
                return False

        # High enough profit
        if profit_target is not None:
            # if m.total_profit < profit_target:
            if m.total_profit <= profit_target:
                return False
        

        return True

    def metrics(self, prices: pd.DataFrame, cutoff_up: float = 1, cutoff_down=None, exit_last: bool = True, lin_increase=False, validate=False):
        """
        Computes metrics of stat arbs trading strategy\
            q_t = mu - p_t until |p_t-mu| >= cutoff
        """
        cutoff_down = cutoff_down or cutoff_up
        # Get price evolution of portfolio

        p = self.evaluate(prices)

        q = self.get_q(prices, cutoff_up, cutoff_down, exit_last=exit_last, lin_increase=lin_increase, validate=validate)
        if q[0] == 0:  # We never enter a position
            return None

        price_changes = p.ffill().diff()
        previous_position = q.shift(1)
        profits = previous_position * price_changes

        return Metrics(daily_profit=profits.dropna())


###########


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
            _construct_stat_arb,
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
