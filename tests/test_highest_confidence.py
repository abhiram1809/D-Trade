import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metrics import directional_positions,strategy


def test_directional_winner_can_trade_below_55_percent_and_flat_stays_cash():
    p=np.array([[.09,.16,.35,.21,.19],  # Up .40 wins; flat is largest single bin.
                [.18,.22,.35,.10,.15],  # Down .40 wins.
                [.15,.15,.40,.15,.15]])
    np.testing.assert_array_equal(directional_positions(p,None),[1,-1,0])
    np.testing.assert_array_equal(directional_positions(p),[0,0,0])


def test_equal_top_probabilities_do_not_choose_an_arbitrary_direction():
    p=np.array([[.2,.2,.2,.2,.2],[.2,.2,.4,.1,.1]])
    np.testing.assert_array_equal(directional_positions(p,None),[0,0])


def test_long_short_cash_replay_costs_compounding_and_drawdown():
    frame=pd.DataFrame({'market':['US']*4,'target_date':pd.date_range('2025-01-01',periods=4).strftime('%Y-%m-%d'),
                        'target_log_return':np.log1p([.01,-.02,.5,-.03])})
    p=np.array([[.09,.16,.35,.21,.19],[.18,.22,.35,.10,.15],[.15,.15,.40,.15,.15],[.09,.16,.35,.21,.19]])
    stats,net=strategy(frame,p,confidence_gate=None)
    np.testing.assert_allclose(net,[.009,.019,0,-.031],atol=1e-12)
    assert stats['total_return']==pytest.approx(1.009*1.019*.969-1)
    assert stats['worst_drawdown']==pytest.approx(-.031)
    assert stats['active_fraction']==.75 and stats['two_sided_turnover']==6
    _,stress=strategy(frame,p,2.,confidence_gate=None)
    np.testing.assert_allclose(stress,[.008,.018,0,-.032],atol=1e-12)


def test_invalid_forecasts_are_rejected():
    with pytest.raises(ValueError):directional_positions([[0,0,np.nan,0,0]],None)
    with pytest.raises(ValueError):directional_positions([[.5]*5],None)
