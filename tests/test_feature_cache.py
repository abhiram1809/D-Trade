import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import feature_cache


def test_recovery_retains_valid_and_quarantines_empty_or_wrong_identity(tmp_path):
    values=np.arange(4,dtype='float32')
    feature_cache.write_durable(tmp_path/'00000000.npz',x=values,tokens=100,example_id='good')
    (tmp_path/'00000001.npz').write_bytes(b'')
    feature_cache.write_durable(tmp_path/'00000002.npz',x=values,tokens=100,example_id='wrong')
    (tmp_path/'00000003.npz').write_bytes((tmp_path/'00000000.npz').read_bytes()[:30])
    result=feature_cache.recover(tmp_path,[0,1,2,3],{0:'good',1:'empty',2:'actual',3:'truncated'},(4,))
    assert [r['index'] for r in result['quarantined']]==[1,2,3]
    feature_cache.read_validated(tmp_path/'00000000.npz','good',(4,))
    assert not (tmp_path/'00000001.npz').exists()
    assert len(list((tmp_path/'quarantine').glob('*.npz')))==3


def test_failed_write_keeps_previous_completed_file(tmp_path,monkeypatch):
    target=tmp_path/'00000000.npz'
    feature_cache.write_durable(target,x=np.ones(4,dtype='float32'),tokens=100,example_id='old')
    previous=target.read_bytes()
    def fail(output,**kwargs):
        output.write(b'interrupted write')
        raise OSError('simulated storage failure')
    monkeypatch.setattr(feature_cache.np,'savez_compressed',fail)
    with pytest.raises(OSError,match='simulated storage failure'):
        feature_cache.write_durable(target,x=np.zeros(4,dtype='float32'),tokens=100,example_id='new')
    assert target.read_bytes()==previous
    assert list(tmp_path.iterdir())==[target]


@pytest.mark.parametrize('values',[np.array([np.nan]*4),np.ones(3)])
def test_invalid_features_are_not_accepted(tmp_path,values):
    target=tmp_path/'00000000.npz'
    feature_cache.write_durable(target,x=values,tokens=100,example_id='id')
    with pytest.raises(ValueError):feature_cache.read_validated(target,'id',(4,))
