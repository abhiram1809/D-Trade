"""Dynamically share independent frozen-encoder jobs across measured CPU/GPU workers."""
import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def worker(name,device,inbox,outbox,save):
    import numpy as np
    import torch
    from d3_features import D3Features
    from feature_cache import write_durable
    try:
        model=D3Features(vulkan=device=='vulkan',threads=1)
        outbox.put({'kind':'ready','worker':name,'initial_sha':hashlib.sha256(model.initial_weight.tobytes()).hexdigest()})
        threads=1
        while True:
            job=inbox.get()
            if job is None:break
            index,state,example_id,requested_threads=job
            if threads!=requested_threads:
                torch.set_num_threads(requested_threads);threads=requested_threads
            started=time.perf_counter()
            x,tokens=model.extract([state])
            if x.shape!=(1,1024) or not np.isfinite(x).all():raise ValueError('Invalid extracted representation')
            if save:
                write_durable(ROOT/f'results/cache/d3-edge/{index:08d}.npz',x=x[0],tokens=tokens[0],example_id=example_id,backend=device+'_fp32')
            outbox.put({'kind':'result','worker':name,'index':index,'seconds':time.perf_counter()-started,
                        'features':None if save else x[0]})
        model.close()
    except BaseException:
        outbox.put({'kind':'error','worker':name,'error':traceback.format_exc()})
        raise


class Pool:
    def __init__(self,devices,save):
        ctx=mp.get_context('spawn');self.results=ctx.Queue();self.processes={};self.inboxes={}
        try:
            for name,device in devices.items():
                inbox=ctx.Queue()
                process=ctx.Process(target=worker,args=(name,device,inbox,self.results,save))
                process.start();self.processes[name]=process;self.inboxes[name]=inbox
            import numpy as np
            expected=hashlib.sha256(np.load(ROOT/'results/cache/d3-edge/initial_weight.npy').tobytes()).hexdigest()
            for _ in devices:
                message=self.receive()
                if message['kind']!='ready' or message['initial_sha']!=expected:
                    raise ValueError('Worker does not match original frozen readout')
        except BaseException:
            self.close();raise

    def receive(self):
        while True:
            try:message=self.results.get(timeout=5)
            except queue.Empty:
                failed={n:p.exitcode for n,p in self.processes.items() if p.exitcode is not None}
                if failed:raise RuntimeError(f'Feature workers exited: {failed}')
                continue
            if message['kind']=='error':raise RuntimeError(message['error'])
            return message

    def run(self,jobs,specs,on_result=None):
        pending=iter(jobs);active=set();completed=[]
        def dispatch(name):
            job=next(pending,None)
            if job is None:return
            self.inboxes[name].put((*job,specs[name]['threads']));active.add(name)
        started=time.perf_counter()
        for name in specs:dispatch(name)
        while active:
            message=self.receive()
            name=message['worker']
            if message['kind']!='result' or name not in active:raise RuntimeError('Unexpected worker reply')
            active.remove(name);completed.append(message)
            if on_result:on_result(completed,started)
            dispatch(name)
        return completed,time.perf_counter()-started

    def close(self):
        for inbox in self.inboxes.values():inbox.put(None)
        for process in self.processes.values():
            process.join(timeout=3)
            if process.is_alive():process.terminate();process.join(timeout=10)
        for inbox in self.inboxes.values():inbox.close()
        self.results.close()


def main(benchmark,extended=False):
    import numpy as np
    import pandas as pd
    from scipy.special import softmax
    from feature_cache import recover
    from model_features import QUESTION
    table=pd.read_parquet(ROOT/'data/processed/examples.parquet')
    folder=ROOT/'results/cache/d3-edge'
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['dataset_sha256']!=hashlib.sha256((ROOT/'data/processed/examples.parquet').read_bytes()).hexdigest():raise ValueError('Dataset changed')
    if manifest['question']!=QUESTION or manifest['model_revision']!=json.loads((ROOT/'vendor/d3-edge/metadata.json').read_text())['sha']:raise ValueError('Model or question changed')
    if benchmark:
        indices=[0,300,1024,2048,3072,4096,5120,6167]*(3 if extended else 1)
        reference={i:np.load(folder/f'{i:08d}.npz')['x'] for i in indices}
        devices={'cpu0':'cpu','cpu1':'cpu','gpu':'vulkan'}
        presets=[{'cpu0':{'device':'cpu','threads':4}},
                 {'cpu0':{'device':'cpu','threads':2},'cpu1':{'device':'cpu','threads':2}},
                 {'cpu0':{'device':'cpu','threads':3},'gpu':{'device':'vulkan','threads':1}},
                 {'cpu0':{'device':'cpu','threads':2},'cpu1':{'device':'cpu','threads':2},'gpu':{'device':'vulkan','threads':1}}]
        if extended:
            presets=[presets[1],presets[3],{'cpu0':{'device':'cpu','threads':4},'cpu1':{'device':'cpu','threads':4}}]
    else:
        config=json.loads((ROOT/'runtime_config.json').read_text())
        specs=config['feature_workers'];devices={n:s['device'] for n,s in specs.items()}
        recovery=recover(folder,manifest['indices'],table['id'],(1024,))
        (ROOT/'results/parallel_cache_recovery.json').write_text(json.dumps(recovery,indent=2))
        indices=[i for split in ('validation','train','test') for i in manifest['indices'] if table.loc[i,'split']==split and not (folder/f'{i:08d}.npz').exists()]
        presets=[specs]
    if not indices:
        print('All planned representations already valid',flush=True);return
    jobs=[(i,table.loc[i,'state'],table.loc[i,'id']) for i in indices]
    w=np.load(folder/'initial_weight.npy').reshape(5,-1)
    reports=[];pool=Pool(devices,save=not benchmark)
    try:
        for specs in presets:
            def progress(completed,started):
                if benchmark:return
                elapsed=time.perf_counter()-started;done=len(completed)
                status={'model':'d3-edge','phase':'all','completed_this_run':done,'total_this_phase':len(jobs),
                        'examples_per_second':done/elapsed,'estimated_remaining_hours':(len(jobs)-done)/(done/elapsed)/3600,
                        'workers':specs,'updated_unix':time.time()}
                (ROOT/'results/d3_feature_progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
            completed,elapsed=pool.run(jobs,specs,progress)
            if benchmark:
                probability_difference=0.;feature_difference=0.
                for result in completed:
                    expected=reference[result['index']];actual=result['features']
                    np.testing.assert_allclose(actual,expected,atol=.003,rtol=.003)
                    feature_difference=max(feature_difference,float(abs(actual-expected).max()))
                    probability_difference=max(probability_difference,float(abs(softmax(actual@w.T)-softmax(expected@w.T)).max()))
                if probability_difference>1e-4:raise ValueError('Runtime changes probabilities beyond tolerance')
                report={'workers':specs,'examples':len(completed),'seconds':elapsed,'seconds_per_example':elapsed/len(completed),
                        'max_feature_difference':feature_difference,'max_probability_difference':probability_difference,
                        'work_distribution':{n:sum(r['worker']==n for r in completed) for n in specs},
                        'worker_seconds':{n:[r['seconds'] for r in completed if r['worker']==n] for n in specs}}
                reports.append(report);print(json.dumps(report),flush=True)
                destination='parallel_runtime_extended.json' if extended else 'parallel_runtime_benchmark.json'
                (ROOT/'results'/destination).write_text(json.dumps(reports,indent=2))
        if not benchmark:
            missing=[i for i in manifest['indices'] if not (folder/f'{i:08d}.npz').exists()]
            if missing:raise RuntimeError(f'Missing {len(missing)} representations after worker completion')
    finally:pool.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');parser.add_argument('--extended',action='store_true');args=parser.parse_args()
    main(args.benchmark,args.extended)
