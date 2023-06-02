# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.simulator.portfolio import build_portfolio


def test_build_with_weights(prices):
    portfolio = build_portfolio(
        prices,
        stocks=pd.DataFrame(
            columns=["A"],
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=2.0,
        ),
    )
    pd.testing.assert_series_equal(
        portfolio.equity.sum(axis=1),
        pd.Series(
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=[3347.56, 3373.80],
        ),
    )


def test_multiply(prices):
    portfolio = build_portfolio(
        prices,
        stocks=pd.DataFrame(
            columns=["A"],
            index=[pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")],
            data=2.0,
        ),
    )
    portfolio = portfolio * 2.0
    pd.testing.assert_series_equal(
        portfolio.equity.sum(axis=1),
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

    port_left = build_portfolio(prices, stocks=pos_left)
    port_right = build_portfolio(prices, stocks=pos_right)

    pd.testing.assert_frame_equal(pos_left, port_left.stocks)
    pd.testing.assert_frame_equal(pos_right, port_right.stocks)

    port_add = port_left + port_right
    www = pd.read_csv(resource_dir / "positions.csv", index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(www, port_add.stocks, check_freq=False)
