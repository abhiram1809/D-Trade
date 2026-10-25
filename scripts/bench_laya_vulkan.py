import sys,time,json,copy
from pathlib import Path
import torch,numpy as np
from transformers.initialization import no_init_weights
sys.path.insert(0,str(Path('models/laya').resolve()))
from rl_agent_api import RLAgent
from vulkan_linear import offload

torch.set_num_threads(2)
torch.backends.mha.set_fastpath_enabled(False)
t=time.time()
with no_init_weights():agent=RLAgent('models/laya',device='cpu')
print('loaded',time.time()-t,flush=True)
agent.cfg['max_len']=2048
state='daily returns basis points: '+','.join(str(i%107-53) for i in range(100))+'. News coverage unknown. sigma=0.02.'
q={'return':{'type':'choice','instructions':'Forecast the next session open-to-close return relative to historical volatility.','criteria':{'large_down':'below -1 sigma','down':'-1 to -0.25 sigma','flat':'-0.25 to +0.25 sigma','up':'+0.25 to +1 sigma','large_up':'above +1 sigma'}}}
t=time.time();ref=agent.system_one(state,q);print('cpu',time.time()-t,ref,flush=True)
t=time.time();engine=offload(agent.model);print('offload',time.time()-t,flush=True)
for i in range(3):
 t=time.time();r=agent.system_one(state,q);print('Vulkan',time.time()-t,r,flush=True)
 a=ref['answers']['return']['probabilities'];b=r['answers']['return']['probabilities'];assert max(abs(a[k]-b[k]) for k in a)<.001
engine.close()
