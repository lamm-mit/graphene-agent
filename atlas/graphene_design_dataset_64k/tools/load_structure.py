"""Retrieve a standard extxyz member without downloading an entire shard."""
import gzip,hashlib,io
from functools import lru_cache
import requests
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download,hf_hub_url
from ase.io import read
REPO='lamm-mit/graphene-design-universe-64k'

@lru_cache(maxsize=8)
def _catalog(shard,revision):
 path=hf_hub_download(REPO,f'data/designs-{shard:04d}.parquet',repo_type='dataset',revision=revision)
 return pq.read_table(path,columns=['index','design_id','coordinate_shard','coordinate_offset','coordinate_bytes','coordinates_sha256']).to_pylist()

def get_extxyz(index,revision='v1.0.0'):
 index=int(str(index).replace('GDU-',''))
 if not 0<=index<64000:raise ValueError('Design index must be 0..63999')
 row=_catalog(index//256,revision)[index%256]
 assert row['index']==index
 start=row['coordinate_offset'];size=row['coordinate_bytes']
 url=hf_hub_url(REPO,row['coordinate_shard'],repo_type='dataset',revision=revision)
 # Use a distinct query for each member to avoid intermediaries caching another range.
 response=requests.get(url+f'?member={index}',headers={'Range':f'bytes={start}-{start+size-1}'},timeout=180)
 response.raise_for_status()
 if response.status_code==206:
  if not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise ValueError('Unexpected HTTP byte range')
  data=response.content
 elif response.status_code==200:
  data=response.content[start:start+size]
 else:raise ValueError(f'Unexpected HTTP status {response.status_code}')
 if len(data)!=size:raise ValueError('Incomplete coordinate member')
 raw=gzip.decompress(data)
 if hashlib.sha256(raw).hexdigest()!=row['coordinates_sha256']:raise ValueError('Coordinate checksum mismatch')
 return raw

def load_structure(index,revision='v1.0.0'):
 return read(io.StringIO(get_extxyz(index,revision).decode()),format='extxyz')
