"""Official d3 rendering/noncausal path, with an inspectable FP32 Vulkan linear backend."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'models/d3-edge'))
from d3_runtime import D3
from model_features import QUESTION
from feature_cache import recover, write_durable


class CaptureEncoder:
    def __init__(self, encoder):
        self.encoder, self.hidden = encoder, None

    def __call__(self, *args, **kwargs):
        hidden = self.encoder(*args, **kwargs)
        self.hidden = hidden[:, -1].detach().float().cpu().numpy().copy()
        return hidden


class D3Features:
    def __init__(self, vulkan=True, threads=2):
        torch.set_num_threads(threads)
        torch.backends.mha.set_fastpath_enabled(False)
        self.model = D3(ROOT/'models/d3-edge', device='cpu', batch_size=2, max_length=8192)
        self.model.backbone.float().eval().requires_grad_(False)
        self.temperature = self.model.temperature
        self.initial_weight = self.model.readout[:5].float().cpu().numpy().copy().reshape(-1) / self.temperature
        self.n_parameters = sum(p.numel() for p in self.model.backbone.parameters()) + self.model.readout.numel()
        if self.n_parameters > 800_000_000:
            raise ValueError('Parameter limit exceeded')
        if self.model.encoder is None:
            raise ValueError('This adapter requires the declared noncausal encoder path')
        self.capture = CaptureEncoder(self.model.encoder)
        self.model.encoder = self.capture
        self.engine = None
        if vulkan:
            from vulkan_linear import offload
            self.engine = offload(self.model.backbone)

    @torch.no_grad()
    def extract(self, states):
        sequences = []
        for state in states:
            text = self.model.text(state, QUESTION)
            ids = self.model.tokenizer(text, add_special_tokens=False)['input_ids']
            if len(ids) > 8192:
                raise ValueError('Cannot preserve complete input')
            sequences.append(ids)
        logits = self.model.logits(sequences, [5]*len(states))[:, :5].cpu().numpy() / self.temperature
        hidden = self.capture.hidden
        derived = hidden @ self.initial_weight.reshape(5, -1).T
        np.testing.assert_allclose(logits, derived, atol=.003, rtol=.003)
        return hidden, [len(s) for s in sequences]

    def close(self):
        if self.engine:
            self.engine.close()


def cache_features(batch_size=2, phase='validation'):
    import pandas as pd
    folder=ROOT/'results/cache/d3-edge'
    folder.mkdir(parents=True, exist_ok=True)
    table=pd.read_parquet(ROOT/'data/processed/examples.parquet')
    evaluation=json.loads((ROOT/'data/processed/evaluation_indices.json').read_text())
    aapl=table.index[(table.symbol=='AAPL')].tolist()
    indices=sorted(set(aapl+evaluation))
    if phase=='validation':
        order=[i for i in evaluation if table.loc[i,'split']=='validation']
    else:
        order=[i for i in indices if table.loc[i,'split']=='validation']+[i for i in indices if table.loc[i,'split']=='train']+[i for i in indices if table.loc[i,'split']=='test']
    digest=hashlib.sha256((ROOT/'data/processed/examples.parquet').read_bytes()).hexdigest()
    mp=folder/'manifest.json'
    revision=json.loads((ROOT/'vendor/d3-edge/metadata.json').read_text())['sha']
    if mp.exists():
        old=json.loads(mp.read_text())
        if old['dataset_sha256']!=digest or old['question']!=QUESTION or old['model_revision']!=revision:
            raise ValueError('Cache provenance changed; use a new cache directory')
    recovery=recover(folder,indices,table['id'],(1024,))
    (ROOT/'results/d3-edge_cache_recovery.json').write_text(json.dumps(recovery,indent=2))
    print('Cache recovery: '+json.dumps(recovery),flush=True)
    model=D3Features(vulkan=True)
    manifest={'dataset_sha256':digest,'question':QUESTION,'model_revision':json.loads((ROOT/'vendor/d3-edge/metadata.json').read_text())['sha'],'indices':indices,'n_parameters':model.n_parameters,'trainable_parameters':len(model.initial_weight),'temperature':model.temperature,'feature_dtype':'float32','feature_representation':'class_block','fine_tune_scope':'first five original answer-code readout rows only; frozen backbone','precision_note':'BF16 checkpoint promoted to FP32 for CPU/Vulkan arithmetic'}
    mp.write_text(json.dumps(manifest,indent=2))
    np.save(folder/'initial_weight.npy',model.initial_weight)
    remaining=[i for i in order if not (folder/f'{i:08d}.npz').exists()]
    started=time.time()
    for start in range(0,len(remaining),batch_size):
        chosen=remaining[start:start+batch_size]
        begin=time.time()
        features,tokens=model.extract(table.loc[chosen,'state'].tolist())
        for idx,x,count in zip(chosen,features,tokens):
            target=folder/f'{idx:08d}.npz'
            write_durable(target,x=x,tokens=count,example_id=table.loc[idx,'id'])
        completed=start+len(chosen);elapsed=time.time()-started
        status={'model':'d3-edge','phase':phase,'completed_this_run':completed,'total_this_phase':len(remaining),'batch_seconds':time.time()-begin,'examples_per_second':completed/elapsed,'estimated_remaining_hours':(len(remaining)-completed)/(completed/elapsed)/3600,'updated_unix':time.time()}
        (ROOT/'results/d3_feature_progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    model.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=2);p.add_argument('--phase',choices=['validation','all'],default='validation');p.add_argument('--benchmark',action='store_true');args=p.parse_args()
    if args.benchmark:
        m=D3Features(vulkan=True)
        state='100 daily returns in basis points, oldest first: '+','.join(str(i%107-53) for i in range(100))+'. sigma=0.02. News coverage unknown.'
        for batch in [1,2,4]:
            t=time.time();x,n=m.extract([state]*batch);print(json.dumps({'batch':batch,'seconds':time.time()-t,'tokens':n,'shape':x.shape,'parameters':m.n_parameters}),flush=True)
        m.close()
    else:
        cache_features(args.batch_size,args.phase)
