"""Measure batched CPU extraction and compare each context with its frozen cache."""
import json,time
import numpy as np
import pandas as pd
from scipy.special import softmax
from d3_features import D3Features,ROOT

table=pd.read_parquet(ROOT/'data/processed/examples.parquet',columns=['state'])
indices=[0,300,1024,2048,3072,4096,5120,6167]
states=table.loc[indices,'state'].tolist()
reference=np.stack([np.load(ROOT/f'results/cache/d3-edge/{i:08d}.npz')['x'] for i in indices])
model=D3Features(vulkan=False,threads=4)
w=model.initial_weight.reshape(5,-1)
rows=[]
for batch in (2,4,8):
    started=time.perf_counter();x,tokens=model.extract(states[:batch]);elapsed=time.perf_counter()-started
    np.testing.assert_allclose(x,reference[:batch],atol=.003,rtol=.003)
    difference=float(abs(softmax(x@w.T,axis=1)-softmax(reference[:batch]@w.T,axis=1)).max())
    if difference>1e-4:raise RuntimeError('Batched forecasts exceed probability parity tolerance')
    row={'device':'cpu','threads':4,'batch':batch,'seconds':elapsed,'seconds_per_example':elapsed/batch,
         'tokens':tokens,'max_feature_difference':float(abs(x-reference[:batch]).max()),'max_probability_difference':difference}
    rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'results/d3_batch_benchmark.json').write_text(json.dumps(rows,indent=2))
model.close()
