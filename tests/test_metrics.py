# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pytest

from cvx.simulator.metrics import Metrics
from cvx.stat_arb.ccp import * 

def test_profit(portfolio):
    m = Metrics(daily_profit=portfolio.profit.dropna())
    pd.testing.assert_series_equal(m.daily_profit, portfolio.profit.dropna())

    assert m.mean_profit == pytest.approx(-5.810981697171386)
    assert m.std_profit == pytest.approx(840.5615726803527)
    assert m.total_profit == pytest.approx(-3492.4000000000033)
    assert m.sr_profit == pytest.approx(-0.10974386369939439)
