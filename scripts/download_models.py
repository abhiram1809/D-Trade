import json
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1]
for model,pin in json.loads((root/'models.lock.json').read_text()).items():
 folder=root/'vendor'/model.split('/')[-1];folder.mkdir(parents=True,exist_ok=True)
 metadata=folder/'metadata.json'
 if not metadata.exists():metadata.write_text(json.dumps({'id':model,'sha':pin['sha'],'safetensors':{'total':pin['safetensors_total']}}))
 meta=json.loads(metadata.read_text())
 if meta['sha']!=pin['sha']:raise ValueError('Model revision differs from models.lock.json')
 print('Downloading',model,meta['sha'],flush=True)
 snapshot_download(model,revision=meta['sha'],local_dir=root/'models'/model.split('/')[-1],allow_patterns=['*.py','*.json','*.safetensors','tokenizer/*','encoder/*','*.jinja','*.txt'],ignore_patterns=['multilingual/*','typed-decisions/*','eval/*'])
