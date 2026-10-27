"""Validation-selected original-head fine-tuning on native Vulkan, then newer test."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from metrics import score, strategy, block_bootstrap_loss_difference
from vulkan_scorer import VulkanScorer

ROOT=Path(__file__).resolve().parents[1]


def expand(x,representation):
    if representation=='class_block':
        out=np.zeros((len(x),5,5*x.shape[-1]),dtype='float32')
        for k in range(5):out[:,k,k*x.shape[-1]:(k+1)*x.shape[-1]]=x
        return out
    return np.asarray(x,dtype='float32')


def load_cache(model,indices):
    folder=ROOT/'results/cache'/model
    arrays=[]
    for i in indices:
        path=folder/f'{i:08d}.npz'
        if not path.exists():raise RuntimeError(f'Missing representation {model} / {i}; do not train on an incomplete dataset')
        arrays.append(np.load(path)['x'])
    return np.array(arrays,dtype='float32')


def logits(x,w,representation):
    if representation=='class_block':return x@w.reshape(5,-1).T
    return x@w


def select_candidate():
    table=pd.read_parquet(ROOT/'data/processed/examples.parquet')
    indices=json.loads((ROOT/'data/processed/evaluation_indices.json').read_text())
    indices=[i for i in indices if table.loc[i,'split']=='validation']
    comparisons=[]
    for model in ['laya','d3-edge']:
        folder=ROOT/'results/cache'/model
        if not (folder/'manifest.json').exists():continue
        manifest=json.loads((folder/'manifest.json').read_text())
        x=load_cache(model,indices);w=np.load(folder/'initial_weight.npy')
        p=softmax(logits(x,w,manifest.get('feature_representation')),axis=1)
        result=score(table.loc[indices,'label'],p)
        result.update(model=model,parameters=manifest['n_parameters'])
        comparisons.append(result)
    if len(comparisons)<2:raise RuntimeError('Both planned candidates must finish validation before selection')
    winner=min(comparisons,key=lambda r:r['log_loss'])['model']
    report={'winner':winner,'selection_metric':'five-bin log loss on predetermined 2024 validation origins','comparisons':comparisons,'test_not_used':True,'selected_unix':time.time()}
    (ROOT/'results/model_selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return winner


def train_and_evaluate(model):
    table=pd.read_parquet(ROOT/'data/processed/examples.parquet')
    folder=ROOT/'results/cache'/model
    manifest=json.loads((folder/'manifest.json').read_text())
    dataset_sha256=hashlib.sha256((ROOT/'data/processed/examples.parquet').read_bytes()).hexdigest()
    if manifest['dataset_sha256']!=dataset_sha256:
        raise ValueError('Representation cache and current dataset hashes differ')
    representation=manifest.get('feature_representation')
    train_ids=table.index[(table.symbol=='AAPL')&(table.split=='train')].tolist()
    valid_ids=table.index[(table.symbol=='AAPL')&(table.split=='validation')].tolist()
    evaluation=json.loads((ROOT/'data/processed/evaluation_indices.json').read_text())
    test_ids=sorted(set([i for i in evaluation if table.loc[i,'split']=='test']+table.index[(table.symbol=='AAPL')&(table.split=='test')].tolist()))
    if len(train_ids)<10000:raise ValueError('10,000 distinct stock examples required')
    x=load_cache(model,train_ids);v=load_cache(model,valid_ids)
    y=table.loc[train_ids,'label'].to_numpy('uint32');vy=table.loc[valid_ids,'label'].to_numpy(int)
    initial=np.load(folder/'initial_weight.npy')
    engine=VulkanScorer(len(initial),batch_size=128)
    rng=np.random.default_rng(20261025)
    histories=[];best=None
    for lr in [1e-4,3e-4,1e-3]:
        for l2 in [0.,.01]:
            engine.set_weights(initial)
            for epoch in range(1,21):
                order=rng.permutation(len(x))
                examples_seen=0
                successful_updates=0
                for start in range(0,len(x),128):
                    ids=order[start:start+128]
                    engine.run(expand(x[ids],representation),y[ids],lr=lr,l2=l2,update=True)
                    examples_seen+=len(ids)
                    successful_updates+=1
                if examples_seen!=len(train_ids) or len(np.unique(order))!=len(train_ids):
                    raise RuntimeError('The training epoch did not cover every distinct stock example')
                w=engine.get_weights();p=softmax(logits(v,w,representation),axis=1)
                result=score(vy,p)
                row={'lr':lr,'l2':l2,'epoch':epoch,'validation_log_loss':result['log_loss'],'examples_seen':examples_seen,'unique_examples_seen':len(np.unique(order)),'successful_gpu_updates':successful_updates,'sample_order_sha256':hashlib.sha256(order.tobytes()).hexdigest()}
                histories.append(row);print(json.dumps(row),flush=True)
                if best is None or result['log_loss']<best['validation_log_loss']:
                    best={**row,'weight':w.copy()}
    weights=best.pop('weight')
    best_logits=logits(v,weights,representation)
    def objective(logtemp):return score(vy,softmax(best_logits/np.exp(logtemp),axis=1))['log_loss']
    fit=minimize_scalar(objective,bounds=(-2.3,2.3),method='bounded');temperature=float(np.exp(fit.x))
    out=ROOT/'models'/f'{model}-market-head';out.mkdir(exist_ok=True)
    np.save(out/'weight.npy',weights)
    np.save(out/'initial_weight.npy',initial)
    from safetensors.numpy import save_file
    if model=='laya':
        save_file({'scorer.3.weight':(weights*manifest['temperature']).reshape(1,-1)},str(out/'original_head_weights.safetensors'))
    else:
        save_file({'readout_first_five_rows':(weights*manifest['temperature']).reshape(5,-1)},str(out/'original_head_weights.safetensors'))
    checkpoint={'model':model,'base_revision':manifest['model_revision'],'base_parameters':manifest['n_parameters'],'trainable_parameters':len(initial),'scope':manifest['fine_tune_scope'],'original_checkpoint_temperature':manifest['temperature'],'additional_validation_temperature':temperature,'training_stock':'AAPL','training_examples':len(train_ids),'training_target_last':table.loc[train_ids,'target_date'].max(),'validation_target_last':table.loc[valid_ids,'target_date'].max(),'best':best,'device':engine.device_name,'training_seconds_unix_finished':time.time(),'representation':representation,'question':manifest['question'],'weight_delta_l2':float(np.linalg.norm(weights-initial)),'weights_sha256':hashlib.sha256((out/'weight.npy').read_bytes()).hexdigest()}
    checkpoint.update(dataset_sha256=dataset_sha256,training_example_ids_sha256=hashlib.sha256('\n'.join(table.loc[train_ids,'id']).encode()).hexdigest(),vulkan_shader_sha256=hashlib.sha256((ROOT/'native/scorer.spv').read_bytes()).hexdigest(),vulkan_library_sha256=hashlib.sha256((ROOT/'native/libscorer.so').read_bytes()).hexdigest(),optimizer='Adam with L2 penalty',batch_size=128,search_training_updates=6*20*int(np.ceil(len(train_ids)/128)),selected_checkpoint_updates=best['epoch']*int(np.ceil(len(train_ids)/128)))
    runtime_path=ROOT/'runtime_config.json'
    checkpoint['feature_runtime']=json.loads(runtime_path.read_text()) if runtime_path.exists() else {'backend':'legacy Vulkan FP32 linear offload with CPU attention'}
    (out/'adapter_config.json').write_text(json.dumps(checkpoint,indent=2))
    (ROOT/'results/training_history.json').write_text(json.dumps(histories,indent=2))
    engine.close()
    # Open the test representations only after all model/hyperparameter decisions freeze.
    tx=load_cache(model,test_ids);test=table.loc[test_ids].reset_index(drop=True)
    base=softmax(logits(tx,initial,representation),axis=1)
    tuned=softmax(logits(tx,weights,representation)/temperature,axis=1)
    frequency=(np.bincount(y,minlength=5)+1)/(len(y)+5)
    baseline=np.tile(frequency,(len(test),1))
    numeric=np.load(ROOT/'data/processed/arrays.npz')['numeric']
    scaler=StandardScaler().fit(numeric[train_ids]);ntrain=scaler.transform(numeric[train_ids]);nvalid=scaler.transform(numeric[valid_ids]);ntest=scaler.transform(numeric[test_ids])
    numeric_scores=[]
    for c in [.001,.01,.1,1.]:
        classifier=LogisticRegression(C=c,max_iter=500).fit(ntrain,y)
        numeric_scores.append((score(vy,classifier.predict_proba(nvalid))['log_loss'],c,classifier))
    _,numeric_c,classifier=min(numeric_scores,key=lambda r:r[0])
    numeric_p=classifier.predict_proba(ntest)
    probabilities={'base':base,'fine_tuned':tuned,'class_frequency':baseline,'numeric_logistic':numeric_p}
    results=[]
    for name,p in probabilities.items():
        for label,sel in [('all',np.ones(len(test),bool)),('AAPL',test.symbol.to_numpy()=='AAPL')]+[(m,test.market.to_numpy()==m) for m in sorted(test.market.unique())]:
            if sel.any():results.append({'model':name,'group':label,**score(test.loc[sel,'label'],p[sel])})
        for symbol,ids in test.groupby('symbol').indices.items():results.append({'model':name,'group':'symbol:'+symbol,**score(test.loc[ids,'label'],p[ids])})
    pd.DataFrame(results).to_csv(ROOT/'results/forecast_metrics.csv',index=False)
    coverage_rows=[]
    primary_stock=test.symbol.to_numpy()=='AAPL'
    retrieved=test.news_json.to_numpy()!='[]'
    for name,p in probabilities.items():
        for group,mask in [('AAPL_with_retrieved_headlines',primary_stock&retrieved),('AAPL_without_retrieved_headlines',primary_stock&~retrieved)]:
            if mask.any():
                coverage_rows.append({'model':name,'group':group,**score(test.loc[mask,'label'],p[mask])})
    pd.DataFrame(coverage_rows).to_csv(ROOT/'results/news_coverage_metrics.csv',index=False)
    for name,p in probabilities.items():
        for k in range(5):test[f'{name}_p{k}']=p[:,k]
    test.to_parquet(ROOT/'results/test_predictions.parquet',index=False)
    return_points=np.load(ROOT/'data/processed/arrays.npz')['returns_bp'][test_ids]
    truth_direction=np.array([0,0,1,2,2])[test.label.to_numpy(int)]
    momentum_direction=np.where(return_points[:,-20:].sum(1)>0,2,np.where(return_points[:,-20:].sum(1)<0,0,1))
    rule_baselines={}
    for group,sel in [('all',np.ones(len(test),bool)),('AAPL',test.symbol.to_numpy()=='AAPL')]:
        rule_baselines[group]={'n':int(sel.sum()),'always_up_three_class_accuracy':float(np.mean(truth_direction[sel]==2)),'trailing_20_point_momentum_three_class_accuracy':float(np.mean(momentum_direction[sel]==truth_direction[sel])),'note':'Hard directional rules have no forecast probability distribution; proper probabilistic scores are not fabricated.'}
    (ROOT/'results/direction_rules.json').write_text(json.dumps(rule_baselines,indent=2))
    aapl=test.symbol.to_numpy()=='AAPL'
    trading={}
    for name,p in probabilities.items():
        # Only complete AAPL daily series supports the continuous backtest headline.
        standard,_=strategy(test.loc[aapl],p[aapl]);stress,_=strategy(test.loc[aapl],p[aapl],2.)
        trading[name]={'AAPL_standard_cost':standard,'AAPL_double_cost':stress}
    (ROOT/'results/trading.json').write_text(json.dumps(trading,indent=2))
    bootstrap=block_bootstrap_loss_difference(test.loc[aapl].reset_index(drop=True),tuned[aapl],baseline[aapl])
    (ROOT/'results/paired_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    summary={'checkpoint':checkpoint,'test_examples':len(test),'primary_stock_test_examples':int(aapl.sum()),'test_target_first':test.target_date.min(),'test_target_last':test.target_date.max(),'numeric_baseline_C':numeric_c,'bootstrap_fine_tuned_vs_frequency_AAPL':bootstrap,'limitations':['Retrospective foundation pretraining overlap unknown','Historical news missing for many markets and test dates','Only final scorer/readout is trained; backbone frozen','Small convenience universe and sparse non-AAPL test sample','Validation reused for model selection/hyperparameters/temperature; validation score optimistic','Illustrative intraday execution and costs; no broker fills']}
    summary['primary_stock_news_counts']={'training_with_headlines':int((table.loc[train_ids,'news_json']!='[]').sum()),'validation_with_headlines':int((table.loc[valid_ids,'news_json']!='[]').sum()),'test_with_headlines':int((aapl&retrieved).sum()),'test_without_headlines':int((aapl&~retrieved).sum())}
    summary['limitations']+=['Headlines-versus-no-headlines diagnostic groups cover different dates; they do not estimate a causal news benefit','The intraday target excludes overnight news gaps','The long AAPL training history spans very different market regimes']
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');p.add_argument('--model',choices=['laya','d3-edge']);args=p.parse_args()
    if args.select:select_candidate()
    elif args.model:train_and_evaluate(args.model)
