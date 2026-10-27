"""Choose runtime settings by throughput and output parity, never market labels."""
import gc
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from d3_features import D3Features, ROOT
from vulkan_linear import VulkanEngine


def main():
    table=pd.read_parquet(ROOT/'data/processed/examples.parquet',columns=['state'])
    indices=[0,3072,6167]
    states=table.loc[indices,'state'].tolist()
    reference=np.stack([np.load(ROOT/f'results/cache/d3-edge/{i:08d}.npz')['x'] for i in indices])
    rows=[]
    original=VulkanEngine.run
    profile=defaultdict(lambda:{'seconds':0.,'calls':0})
    def timed(engine,index,x,out_features):
        started=time.perf_counter()
        result=original(engine,index,x,out_features)
        shape=str((int(x.numel()//x.shape[-1]),out_features,int(x.shape[-1])))
        profile[shape]['seconds']+=time.perf_counter()-started
        profile[shape]['calls']+=1
        return result
    VulkanEngine.run=timed
    for vulkan in (True,False):
        model=D3Features(vulkan=vulkan)
        w=model.initial_weight.reshape(5,-1)
        ref_p=softmax(reference@w.T,axis=1)
        for threads in (1,2,4,8):
            torch.set_num_threads(threads)
            durations=[];features=[]
            profile.clear()
            for state in states:
                started=time.perf_counter()
                x,tokens=model.extract([state])
                durations.append(time.perf_counter()-started)
                features.append(x[0])
            features=np.array(features)
            delta=float(np.max(abs(features-reference)))
            pdiff=float(np.max(abs(softmax(features@w.T,axis=1)-ref_p)))
            np.testing.assert_allclose(features,reference,atol=.003,rtol=.003)
            if pdiff>1e-4:raise RuntimeError('Runtime changed decision probabilities beyond tolerance')
            row={'vulkan':vulkan,'threads':threads,'indices':indices,'seconds':durations,
                 'median_seconds':float(np.median(durations)),'max_feature_difference':delta,
                 'max_probability_difference':pdiff,'vulkan_linear_by_shape':dict(profile)}
            rows.append(row)
            print(json.dumps(row),flush=True)
            (ROOT/'results/d3_runtime_benchmark.json').write_text(json.dumps(rows,indent=2))
        model.close();del model;gc.collect()


if __name__=='__main__':main()
