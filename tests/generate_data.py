import pandas as pd

from cvx.stat_arb.ccp import construct_stat_arbs
from cvx.simulator.metrics import Metrics

# import test data


if __name__ == "__main__":
    prices = pd.read_csv(
        "tests/resources/price.csv", index_col=0, header=0, parse_dates=True
    ).ffill()
    prices_train, prices_val, prices_test = (
        prices.iloc[:300],
        prices.iloc[300:450],
        prices.iloc[450:],
    )
    prices_train_val = pd.concat([prices_train, prices_val], axis=0)

    # Stat arb constructor
    stat_arb_group = construct_stat_arbs(
        prices_train, K=10, P_max=10, spread_max=1, M=None, solver="ECOS", seed=1
    )

    assets = pd.Series(stat_arb_group.stat_arbs[0].assets)
    mu = pd.Series(stat_arb_group.stat_arbs[0].mu)

    assets.to_csv("tests/resources/stat_arb_assets.csv")
    mu.to_csv("tests/resources/stat_arb_mu.csv")

    # Simple linear trading strategy
    stat_arbs_validated = stat_arb_group.validate(
        prices_val, prices_train_val, 1.05, -10
    )
    portfolio = stat_arbs_validated.construct_portfolio(prices_test, 1.05)

    # Portfolio holdings
    holdings = portfolio.stocks * portfolio.prices
    holdings.to_csv("tests/resources/holdings.csv")

    # Portfolio performance
    m_p = Metrics(portfolio.profit)
    daily_profit = pd.Series(m_p.daily_profit)
    total_profit = pd.Series(m_p.total_profit)
    sr_profit = pd.Series(m_p.sr_profit)

    daily_profit.to_csv("tests/resources/daily_profit.csv")
    total_profit.to_csv("tests/resources/total_profit.csv")
    sr_profit.to_csv("tests/resources/sr_profit.csv")
