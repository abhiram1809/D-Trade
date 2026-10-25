import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from metrics import score, strategy
from news import NewsIndex,salience,render_news


def test_scoring_prefers_correct_probabilities():
    y=np.array([0,1,2,3,4]*10)
    good=.8*np.eye(5)[y]+.04
    bad=np.roll(good,1,axis=1)
    assert score(y,good)['log_loss']<score(y,bad)['log_loss']
    assert score(y,good)['brier']<score(y,bad)['brier']


def test_intraday_costs_pay_both_sides():
    frame=pd.DataFrame({'market':['US','US'],'target_date':['2025-01-02','2025-01-03'],'target_log_return':[0.,0.]})
    p=np.tile([0.,0.,0.,.8,.2],(2,1))
    r,net=strategy(frame,p)
    np.testing.assert_allclose(net,[-.001,-.001])
    assert r['two_sided_turnover']==4


def test_conditional_direction_excludes_flat_outcomes():
    y=np.array([0,2,4])
    p=np.array([[.5,.3,.1,.05,.05],[.05,.05,.8,.05,.05],[.05,.05,.1,.3,.5]])
    result=score(y,p)
    assert result['nonflat_n']==2
    assert result['nonflat_directional_accuracy']==1.
    only_flat=score(np.array([2]),p[1:2])
    assert only_flat['nonflat_n']==0
    assert only_flat['nonflat_binary_log_loss'] is None


def test_material_fact_overrides_noise_phrase():
    assert salience('Shares are trading lower after a factory explosion')=='potentially_material_requires_verification'
    assert salience('Analyst maintains rating and price target')=='likely_noise'


def test_news_time_gate_and_delay(tmp_path):
    rows=[{'Date':'2023-01-01 10:00:00 UTC','Article_title':'Verified factory explosion','Stock_symbol':'AAPL','Url':'https://example.test/a'}, {'Date':'2023-01-04 10:00:00 UTC','Article_title':'Future earnings report','Stock_symbol':'AAPL','Url':'https://example.test/b'}]
    (tmp_path/'AAPL_fnspid.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    news=NewsIndex(tmp_path)
    assert news.at('AAPL',pd.Timestamp('2023-01-01 23:59Z'))[0]==[]
    found,_=news.at('AAPL',pd.Timestamp('2023-01-02 11:00Z'))
    assert len(found)==1
    assert 'Future' not in found[0]['title']


def test_news_recency_uses_forecast_origin():
    row={'published_utc':'2025-01-01T10:00:00+00:00','available_utc':'2025-01-02T10:00:00+00:00','title':'A corporate announcement','salience':'uncertain'}
    text=render_news([row],pd.Timestamp('2025-01-02T16:00Z'))
    assert 'publication timestamp age 30.0h' in text
    assert 'availability age 6.0h' in text
    with pytest.raises(ValueError,match='Future news'):
        render_news([row],pd.Timestamp('2025-01-02T09:00Z'))


@pytest.fixture(scope='module')
def dataset():
    path=ROOT/'data/processed/examples.parquet'
    if not path.exists():pytest.skip('Integration dataset has not finished preparing')
    return pd.read_parquet(path)


def test_chronological_boundaries_and_stock_size(dataset):
    aapl=dataset[(dataset.symbol=='AAPL')&(dataset.split=='train')]
    assert len(aapl)>=10000
    assert aapl.id.is_unique
    assert (dataset.n_points==100).all()
    assert (dataset.origin_date<dataset.target_date).all()
    assert dataset[dataset.split=='train'].target_date.max()<'2024-01-01'
    assert dataset[dataset.split=='validation'].target_date.min()>='2024-01-01'
    assert dataset[dataset.split=='validation'].target_date.max()<'2025-01-01'
    assert dataset[dataset.split=='test'].target_date.min()>='2025-01-01'
    assert dataset.target_date.max()<'2026-10-25'
    assert not dataset.state.str.contains('Last close=').any()


def test_all_prompt_series_have_100_points(dataset):
    # Explicitly catch accidental tokenizer-oriented shortening of candle sequences.
    for state in dataset.state:
        values=state.split('r_bp=',1)[1].split('\n',1)[0].split(',')
        assert len(values)==100
        assert all(v.strip().lstrip('-').isdigit() for v in values)


def test_raw_target_reconstruction(dataset):
    for symbol in ['AAPL','RELIANCE.NS','EURUSD=X','ASML.AS']:
        raw=pd.read_parquet(ROOT/'data/raw/prices'/f'{symbol}.parquet').set_index('date')
        rows=dataset[dataset.symbol==symbol].iloc[::113]
        for r in rows.itertuples():
            truth=float(np.log(raw.loc[r.target_date,'Close']/raw.loc[r.target_date,'Open']))
            assert r.target_log_return==pytest.approx(truth,abs=1e-10)
            expected=int(np.searchsorted([-1.,-.25,.25,1.],truth/r.sigma,side='right'))
            assert r.label==expected


def test_no_future_news(dataset):
    for row in dataset[dataset.news_json!='[]'].iloc[::13].itertuples():
        for news in json.loads(row.news_json):
            assert pd.Timestamp(news['available_utc'])<=pd.Timestamp(row.origin_utc)


def test_actual_gpu_training_evidence():
    report=json.loads((ROOT/'results/vulkan_parity.json').read_text())
    assert all(r['passed'] and r['weight_delta_l2']>0 for r in report['tests'])
    assert all('RX 560' in r['device'] for r in report['tests'])
    assert max(r['gradient_max_abs'] for r in report['tests'])<1e-5
