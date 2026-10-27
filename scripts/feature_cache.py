"""Durable representation files and recovery after an interrupted machine."""
import os
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np


def read_validated(path, example_id, shape):
    with np.load(path, allow_pickle=False) as item:
        x=item['x']
        if x.shape!=shape or not np.isfinite(x).all():
            raise ValueError('Invalid feature shape or nonfinite values')
        if str(item['example_id'])!=example_id:
            raise ValueError('Cached example identity differs from dataset')
        if item['tokens'].shape!=() or int(item['tokens'])<=0:
            raise ValueError('Invalid token count')


def recover(folder, indices, example_ids, shape):
    checked=0
    quarantined=[]
    for i in indices:
        path=folder/f'{i:08d}.npz'
        if not path.exists():
            continue
        checked+=1
        try:
            read_validated(path, example_ids[i], shape)
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile, zlib.error) as error:
            quarantine=folder/'quarantine'
            quarantine.mkdir(exist_ok=True)
            destination=quarantine/f'{path.stem}.{time.time_ns()}.npz'
            path.replace(destination)
            quarantined.append({'index':i,'error':str(error),'preserved_at':str(destination)})
    return {'checked':checked,'quarantined':quarantined,'checked_unix':time.time()}


def write_durable(target, *, x, tokens, example_id):
    target=Path(target)
    temporary=target.with_name('.'+target.name+'.tmp')
    try:
        with temporary.open('wb') as output:
            np.savez_compressed(output,x=x,tokens=tokens,example_id=example_id)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary,target)
        directory=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
