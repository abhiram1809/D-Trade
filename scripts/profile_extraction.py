"""Profile actual unmodified model operations before the sustained run."""
import json,time
import numpy as np,pandas as pd,torch
from model_features import LayaFeatures,ROOT
from vulkan_linear import VulkanEngine
profile={'linear_seconds':0.,'calls':0}
original=VulkanEngine.run
def timed(self,*args,**kwargs):
 t=time.perf_counter()
 try:return original(self,*args,**kwargs)
 finally:profile['linear_seconds']+=time.perf_counter()-t;profile['calls']+=1
VulkanEngine.run=timed
m=LayaFeatures(vulkan=True)
file=next((ROOT/'results/cache/laya').glob('[0-9]*.npz'))
idx=int(file.stem);old=np.load(file)['x'];table=pd.read_parquet(ROOT/'data/processed/examples.parquet');state=table.loc[idx,'state']
reports=[]
for threads in [2,4]:
 torch.set_num_threads(threads)
 profile.update(linear_seconds=0.,calls=0)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
  t=time.perf_counter();x,tokens=m.extract([state]);elapsed=time.perf_counter()-t
 np.testing.assert_allclose(x[0],old,atol=.003,rtol=.003)
 report={'threads':threads,'elapsed':elapsed,'vulkan_linear_seconds':profile['linear_seconds'],'linear_calls':profile['calls'],'tokens':tokens,'max_feature_difference_from_cache':float(np.max(abs(x[0]-old)))}
 reports.append(report);print(json.dumps(report),flush=True);print(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=12),flush=True)
m.close();(ROOT/'results/extraction_profile.json').write_text(json.dumps(reports,indent=2))
