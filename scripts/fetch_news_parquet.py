"""Read only headline/date/symbol columns from an FNSPID parquet mirror over HTTP ranges."""
import io,json,time
from pathlib import Path
import requests,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
class RemoteFile(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.session=requests.Session();self.downloaded=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  if n==0:return b''
  end=min(self.size-1,self.pos+n-1)
  for attempt in range(3):
   try:
    r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{end}'},timeout=60);r.raise_for_status()
    if r.status_code!=206:raise RuntimeError(f'Range request returned {r.status_code}')
    b=r.content
    if len(b)!=end-self.pos+1:raise RuntimeError(f'Range length {len(b)} != {end-self.pos+1}')
    self.pos+=len(b);self.downloaded+=len(b);return b
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 def readinto(self,b):
  x=self.read(len(b));b[:len(x)]=x;return len(x)

def main():
 repo='benstaf/FNSPID-filtered-nasdaq-100'
 meta=requests.get(f'https://huggingface.co/api/datasets/{repo}').json();rev=meta['sha']
 files=requests.get(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/data').json()
 symbols={'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','JPM','XOM','KO','IBM'}
 folder=ROOT/'data/raw/news';outputs={s:[] for s in symbols};logs=[]
 for f in files:
  if not f['path'].endswith('.parquet'):continue
  raw=RemoteFile(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{f["path"]}',f['size'])
  pf=pq.ParquetFile(raw);print(f['path'],'groups',pf.num_row_groups,'schema',pf.schema.names,flush=True)
  wanted=[c for c in ['Date','Article_title','Stock_symbol','Url','Publisher'] if c in pf.schema.names]
  sym_idx=pf.schema.names.index('Stock_symbol')
  for i in range(pf.num_row_groups):
   stat=pf.metadata.row_group(i).column(sym_idx).statistics
   if stat and stat.has_min_max and not any(stat.min<=s<=stat.max for s in symbols):continue
   table=pf.read_row_group(i,columns=wanted).to_pandas()
   for s in symbols:
    outputs[s].extend(table[table.Stock_symbol==s].to_dict('records'))
   print('group',i,'total',sum(map(len,outputs.values())),'downloaded MB',round(raw.downloaded/1e6,1),flush=True)
  logs.append({'file':f['path'],'size':f['size'],'bytes_downloaded':raw.downloaded})
 for s,rows in outputs.items():
  if not rows:continue
  (folder/f'{s}_fnspid.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in rows))
  (folder/f'{s}_fnspid_status.json').write_text(json.dumps({'complete':True,'source':repo,'revision':rev,'original_source':'Zihan1004/FNSPID','rows':len(rows),'date_min':min(r['Date'] for r in rows),'date_max':max(r['Date'] for r in rows)},indent=2))
 (folder/'parquet_mirror_manifest.json').write_text(json.dumps({'repo':repo,'revision':rev,'files':logs,'counts':{s:len(r) for s,r in outputs.items()}},indent=2))
if __name__=='__main__':main()
