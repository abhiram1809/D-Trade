"""Verify completed artifacts and independently replay the selected Vulkan fit."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from feature_cache import read_validated
from metrics import score, strategy, block_bootstrap_loss_difference
from train_evaluate import expand
from vulkan_scorer import VulkanScorer

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=json.loads((ROOT/'results/summary.json').read_text());checkpoint=summary['checkpoint']
    table=pd.read_parquet(ROOT/'data/processed/examples.parquet')
    assert digest(ROOT/'data/processed/examples.parquet')==checkpoint['dataset_sha256']
    assert checkpoint['base_parameters']<800_000_000 and checkpoint['trainable_parameters']==5120
    train=table.index[(table.symbol=='AAPL')&(table.split=='train')].tolist()
    valid=table.index[(table.symbol=='AAPL')&(table.split=='validation')].tolist()
    assert len(train)==10752 and table.loc[train,'id'].nunique()==10752
    assert table.loc[train,'target_date'].max()<'2024-01-01'
    assert hashlib.sha256('\n'.join(table.loc[train,'id']).encode()).hexdigest()==checkpoint['training_example_ids_sha256']
    folder=ROOT/'results/cache/d3-edge';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['dataset_sha256']==checkpoint['dataset_sha256']
    assert manifest['model_revision']==checkpoint['base_revision']
    vectors={};backends={}
    for i in manifest['indices']:
        path=folder/f'{i:08d}.npz';read_validated(path,table.loc[i,'id'],(1024,))
        assert len(table.loc[i,'state'].split('r_bp=',1)[1].split('\n',1)[0].split(','))==100
        with np.load(path,allow_pickle=False) as item:
            vectors[i]=item['x'].copy()
            backend=str(item['backend']) if 'backend' in item else 'legacy_vulkan_fp32'
            backends[backend]=backends.get(backend,0)+1
    adapter=ROOT/'models/d3-edge-market-head'
    initial=np.load(adapter/'initial_weight.npy');weight=np.load(adapter/'weight.npy')
    assert digest(adapter/'weight.npy')==checkpoint['weights_sha256']
    np.testing.assert_array_equal(initial,np.load(folder/'initial_weight.npy'))
    np.testing.assert_allclose(np.linalg.norm(weight-initial),checkpoint['weight_delta_l2'],atol=1e-7)
    assert np.linalg.norm(weight-initial)>0
    assert digest(ROOT/'native/scorer.spv')==checkpoint['vulkan_shader_sha256']
    assert digest(ROOT/'native/libscorer.so')==checkpoint['vulkan_library_sha256']
    history=json.loads((ROOT/'results/training_history.json').read_text())
    assert len(history)==120 and sum(r['successful_gpu_updates'] for r in history)==10080
    rng=np.random.default_rng(20261025);selected_orders=[];best=checkpoint['best']
    for row in history:
        order=rng.permutation(len(train))
        assert row['sample_order_sha256']==hashlib.sha256(order.tobytes()).hexdigest()
        assert row['examples_seen']==row['unique_examples_seen']==10752
        assert row['successful_gpu_updates']==84
        if row['lr']==best['lr'] and row['l2']==best['l2'] and row['epoch']<=best['epoch']:
            selected_orders.append(order)
    assert min(history,key=lambda r:r['validation_log_loss'])==best
    v=np.stack([vectors[i] for i in valid]);vy=table.loc[valid,'label'].to_numpy()
    np.testing.assert_allclose(score(vy,softmax(v@weight.reshape(5,-1).T,axis=1))['log_loss'],best['validation_log_loss'],atol=1e-6)
    predicted=pd.read_parquet(ROOT/'results/test_predictions.parquet')
    evaluation=json.loads((ROOT/'data/processed/evaluation_indices.json').read_text())
    test_ids=sorted(set([i for i in evaluation if table.loc[i,'split']=='test']+table.index[(table.symbol=='AAPL')&(table.split=='test')].tolist()))
    assert len(predicted)==1246 and predicted.id.tolist()==table.loc[test_ids,'id'].tolist()
    for column in ('state','label','target_date','target_log_return'):
        np.testing.assert_array_equal(predicted[column],table.loc[test_ids,column])
    assert predicted.target_date.min()>='2025-01-01' and predicted.target_date.max()<'2026-10-25'
    assert predicted.symbol.nunique()==34 and int((predicted.symbol=='AAPL').sum())==454
    tx=np.stack([vectors[i] for i in test_ids]);probabilities={}
    for name in ('base','fine_tuned','class_frequency','numeric_logistic'):
        p=predicted[[f'{name}_p{k}' for k in range(5)]].to_numpy()
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        np.testing.assert_allclose(p.sum(1),1,atol=2e-7);probabilities[name]=p
    np.testing.assert_allclose(probabilities['base'],softmax(tx@initial.reshape(5,-1).T,axis=1),atol=2e-6)
    np.testing.assert_allclose(probabilities['fine_tuned'],softmax((tx@weight.reshape(5,-1).T)/checkpoint['additional_validation_temperature'],axis=1),atol=2e-6)
    frequency=(np.bincount(table.loc[train,'label'],minlength=5)+1)/(len(train)+5)
    np.testing.assert_allclose(probabilities['class_frequency'],np.tile(frequency,(len(predicted),1)),atol=1e-12)
    metrics=pd.read_csv(ROOT/'results/forecast_metrics.csv');checked=0
    for record in metrics.to_dict('records'):
        group=record['group']
        mask=np.ones(len(predicted),bool) if group=='all' else (predicted.symbol==group[7:] if group.startswith('symbol:') else predicted.symbol=='AAPL' if group=='AAPL' else predicted.market==group)
        calculated=score(predicted.loc[mask,'label'],probabilities[record['model']][mask])
        for key,value in calculated.items():
            if value is not None:np.testing.assert_allclose(record[key],value,atol=1e-10)
        p=probabilities[record['model']][mask];y=predicted.loc[mask,'label'].to_numpy(int)
        independent_loss=-np.log(np.clip(p[np.arange(len(y)),y],1e-9,1)).mean()
        np.testing.assert_allclose(record['log_loss'],independent_loss,atol=2e-6);checked+=1
    aapl=predicted.symbol=='AAPL'
    trading=json.loads((ROOT/'results/trading.json').read_text())
    for name,p in probabilities.items():
        for label,multiplier in [('AAPL_standard_cost',1.),('AAPL_double_cost',2.)]:
            calculated,_=strategy(predicted.loc[aapl],p[aapl],multiplier)
            assert calculated==trading[name][label]
    bootstrap=block_bootstrap_loss_difference(predicted.loc[aapl].reset_index(drop=True),probabilities['fine_tuned'][aapl],probabilities['class_frequency'][aapl])
    assert bootstrap==summary['bootstrap_fine_tuned_vs_frequency_AAPL']
    # Replay only the already selected checkpoint, not another model search.
    x=np.stack([vectors[i] for i in train]);y=table.loc[train,'label'].to_numpy('uint32')
    engine=VulkanScorer(len(initial),batch_size=128);engine.set_weights(initial)
    updates=0
    try:
        for order in selected_orders:
            for start in range(0,len(order),128):
                ids=order[start:start+128]
                engine.run(expand(x[ids],'class_block'),y[ids],lr=best['lr'],l2=best['l2'],update=True);updates+=1
        replayed=engine.get_weights();device=engine.device_name
    finally:engine.close()
    np.testing.assert_allclose(replayed,weight,atol=1e-6,rtol=1e-6)
    assert updates==checkpoint['selected_checkpoint_updates']==1008
    report={'passed':True,'dataset_sha256':checkpoint['dataset_sha256'],'cache_entries_checked':len(vectors),
            'extraction_backends':backends,'training_distinct_examples':len(train),'test_examples':len(predicted),
            'primary_stock_test_examples':int(aapl.sum()),'metric_rows_recomputed':checked,
            'training_epoch_orders_verified':len(history),'search_gpu_updates':10080,
            'independent_vulkan_checkpoint_replay':{'updates':updates,'device':device,'max_weight_difference':float(abs(replayed-weight).max()),'bitwise_equal':bool(np.array_equal(replayed,weight))},
            'checked_unix':time.time()}
    (ROOT/'results/completed_run_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
