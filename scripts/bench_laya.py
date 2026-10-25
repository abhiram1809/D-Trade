import sys,time,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path('models/laya').resolve()))
from rl_agent_api import RLAgent
for n in [4]:
 torch.set_num_threads(n)
 t=time.time();agent=RLAgent('models/laya',device='cpu');print('loaded',time.time()-t,flush=True)
 agent.cfg['max_len']=1024
 state='daily returns basis points: '+','.join(str(i%107-53) for i in range(100))+'. News coverage unknown. sigma=0.02.'
 q={'return':{'type':'choice','instructions':'Forecast the next session open-to-close return relative to historical volatility.','criteria':{'large_down':'below -1 sigma','down':'-1 to -0.25 sigma','flat':'-0.25 to +0.25 sigma','up':'+0.25 to +1 sigma','large_up':'above +1 sigma'}}}
 for _ in range(3):
  t=time.time();r=agent.system_one(state,q);print('elapsed',time.time()-t,r,flush=True)
