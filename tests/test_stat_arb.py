import pytest
import pandas as pd
import numpy as np

from cvx.stat_arb.ccp import construct_stat_arbs
from cvx.simulator.metrics import Metrics

@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(resource_dir / "price.csv", index_col=0, header=0, parse_dates=True).ffill()

@pytest.fixture()
def prices_train(prices):
    return prices.iloc[:300]

@pytest.fixture()
def prices_val(prices):
    return prices.iloc[300:450]

@pytest.fixture()
def prices_test(prices):
    return prices.iloc[450:]

@pytest.fixture()
def prices_train_val(prices_train, prices_val):
    return pd.concat([prices_train, prices_val], axis=0)

@pytest.fixture()
def stat_arb_group(prices_train):
    return construct_stat_arbs(prices_train, K=10, P_max=10,\
        spread_max=1, M=None, solver="ECOS", seed=1)

def test_stat_arb_constructor(stat_arb_group):

    assets = pd.Series(stat_arb_group.stat_arbs[0].assets).astype(float)
    mu = stat_arb_group.stat_arbs[0].mu

    assets_test = pd.read_csv("resources/stat_arb_assets.csv", index_col=0).squeeze().astype(float)
    mu_test = pd.read_csv("resources/stat_arb_mu.csv", index_col=0).squeeze()

    pd.testing.assert_series_equal(assets, assets_test, check_names=False)
    assert np.allclose(mu, mu_test)


def test_stat_arb_trading(stat_arb_group, prices_val, prices_test,\
     prices_train_val):

    # Simple linear trading strategy
    stat_arbs_validated = stat_arb_group.validate(prices_val,
            prices_train_val, 1.05, -10)
    portfolio = stat_arbs_validated.construct_porfolio(prices_test, 1.05) 

    # Portfolio holdings
    holdings = portfolio.stocks * portfolio.prices
    holdings_test = pd.read_csv("resources/holdings.csv", index_col=0, header=0, parse_dates=True)
    pd.testing.assert_frame_equal(holdings, holdings_test)


    # Portfolio performance  
    m_p = Metrics(portfolio.profit)
    daily_profit = pd.Series(m_p.daily_profit)
    total_profit = m_p.total_profit
    sr_profit = m_p.sr_profit

    daily_profit_test = pd.read_csv("resources/daily_profit.csv", index_col=0, header=0, parse_dates=True).squeeze()
    total_profit_test = pd.read_csv("resources/total_profit.csv", index_col=0, header=0, parse_dates=True).squeeze()
    sr_profit_test = pd.read_csv("resources/sr_profit.csv", index_col=0, header=0, parse_dates=True).squeeze()

    pd.testing.assert_series_equal(daily_profit, daily_profit_test, check_names=False)
    assert np.allclose(total_profit, total_profit_test)
    assert np.allclose(sr_profit, sr_profit_test)



