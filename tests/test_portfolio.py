import numpy as np
import pandas as pd
import pytest

from cvx.simulator.portfolio import EquityPortfolio


def test_build_with_weights(prices):
    portfolio = EquityPortfolio(
        prices,
        stocks=pd.DataFrame(
            columns=["A"],
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=2.0,
        ),
    )

    equity = (
        portfolio.equity.sum(axis=1)
        .dropna()
        .loc[[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")]])

    pd.testing.assert_series_equal(
        equity,
        pd.Series(
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=[3347.56, 3373.80],),)


def test_multiply(prices):
    portfolio = EquityPortfolio(
        prices,
        stocks=pd.DataFrame(
            columns=["A"],
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=2.0,
        ),
    )
    portfolio = portfolio * 2.0

    equity = (
        portfolio.equity.sum(axis=1)
        .dropna()
        .loc[[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")]]
    )

    pd.testing.assert_series_equal(
        equity,
        2.0
        * pd.Series(
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=[3347.56, 3373.80],
        ),
    )


def test_add(prices, resource_dir):
    """
    Tests the addition of two portfolios
    TODP: Currently only tests the positions of the portfolios
    """
    index_left = pd.DatetimeIndex(
        [pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")]
    )
    index_right = pd.DatetimeIndex(
        [
            pd.Timestamp("2013-01-02"),
            pd.Timestamp("2013-01-03"),
            pd.Timestamp("2013-01-04"),
        ]
    )

    pos_left = pd.DataFrame(data={"A": [0, 1], "C": [3, 3]}, index=index_left)
    pos_right = pd.DataFrame(data={"A": [1, 1, 2], "B": [2, 3, 4]}, index=index_right)

    port_left = EquityPortfolio(prices, stocks=pos_left)
    port_right = EquityPortfolio(prices, stocks=pos_right)

    pd.testing.assert_frame_equal(pos_left, port_left.stocks)
    pd.testing.assert_frame_equal(pos_right, port_right.stocks)

    port_add = port_left + port_right
    www = pd.read_csv(resource_dir / "positions.csv", index_col=0, parse_dates=[0])

    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2013-01-02"),
            pd.Timestamp("2013-01-03"),
            pd.Timestamp("2013-01-04"),
        ]
    )
    pd.testing.assert_frame_equal(
        www, port_add.stocks[www.columns].loc[index], check_freq=False
    )
