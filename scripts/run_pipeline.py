"""One resumable sequential GPU worker; completion is recorded only after real evaluation."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'.venv/bin/python')
STATUS=ROOT/'results/run_status.json'


def status(phase,state='running',**kwargs):
    STATUS.write_text(json.dumps({'phase':phase,'state':state,'updated_unix':time.time(),'runner_pid':os.getpid(),**kwargs},indent=2))


def run(phase,args):
    status(phase)
    print(phase,'started',time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    runtime=ROOT/'runtime_config.json'
    if phase=='vulkan_fine_tuning_and_test' and runtime.exists():
        threads=str(json.loads(runtime.read_text()).get('training_cpu_threads',4))
        env.update(OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,MKL_NUM_THREADS=threads)
    with (ROOT/'results'/f'{phase}.log').open('a') as log:
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(phase,worker_pid=process.pid)
        code=process.wait()
        if code:
            raise subprocess.CalledProcessError(code,args)


def main():
    # An enabled user service must not retrain an already completed experiment
    # after a later login. A changed dataset invalidates this completion marker.
    summary_path=ROOT/'results/summary.json'
    report_path=ROOT/'results/REPORT.md'
    if STATUS.exists() and summary_path.exists() and report_path.exists():
        previous=json.loads(STATUS.read_text())
        if previous.get('state')=='complete':
            import hashlib
            summary=json.loads(summary_path.read_text())
            digest=hashlib.sha256((ROOT/'data/processed/examples.parquet').read_bytes()).hexdigest()
            if summary['checkpoint']['dataset_sha256']==digest:
                print('Existing completed experiment matches dataset; no restart needed',flush=True)
                return
    # A separate authorized preparation process is already running. Do not race it.
    status('await_prepared_dataset')
    while not (ROOT/'data/processed/manifest.json').exists():time.sleep(3)
    manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    if not manifest.get('quality_audit'):
        run('causal_data_audit',[PY,'scripts/clean_prepared.py'])
    if (ROOT/'data/raw/news/recent_archive_manifest.json').exists() and not (ROOT/'results/news_augmentation_audit.json').exists():
        run('augment_recent_news',[PY,'scripts/augment_recent_news.py'])
    run('integrity_tests',[PY,'-m','pytest','-q','tests'])
    run('laya_validation',[PY,'scripts/model_features.py','--phase','validation','--batch-size','1'])
    run('d3_validation',[PY,'scripts/d3_features.py','--phase','validation','--batch-size','1'])
    run('candidate_selection',[PY,'scripts/train_evaluate.py','--select'])
    selected=json.loads((ROOT/'results/model_selection.json').read_text())['winner']
    feature_script='scripts/model_features.py' if selected=='laya' else 'scripts/d3_features.py'
    if selected=='d3-edge' and (ROOT/'runtime_config.json').exists():
        run('selected_model_full_features',[PY,'scripts/parallel_features.py'])
    else:
        run('selected_model_full_features',[PY,feature_script,'--phase','all','--batch-size','1'])
    run('vulkan_fine_tuning_and_test',[PY,'scripts/train_evaluate.py','--model',selected])
    run('final_report',[PY,'scripts/report.py'])
    status('complete','complete',selected_model=selected,report=str(ROOT/'results/REPORT.md'))
    print('Completed measured experiment',flush=True)


if __name__=='__main__':
    # Prevent duplicate concurrent GPU workers when resuming the goal.
    import fcntl
    lock=(ROOT/'results/pipeline.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A pipeline worker already holds the GPU lock')
    (ROOT/'results/runner_process.json').write_text(json.dumps({
        'pid':os.getpid(),'started_unix':time.time(),'command':'scripts/run_pipeline.py',
        'launch':'systemd user service' if os.environ.get('INVOCATION_ID') else 'manual runner',
    },indent=2))
    try:main()
    except Exception as exc:
        status('failed','failed',error=repr(exc),traceback=traceback.format_exc())
        raise
