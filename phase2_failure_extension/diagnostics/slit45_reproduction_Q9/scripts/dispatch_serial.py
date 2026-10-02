"""Sole serial Q9 queue; never overwrite or silently retry an attempt."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
(ROOT / 'results').mkdir(parents=True, exist_ok=True)
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 lock=open(ROOT/'logs/queue.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
 jobs=json.loads((ROOT/'jobs.json').read_text())
 for j in jobs:
  p=ROOT/'runs'/j['id'];rp=p/'record.json'
  if p.exists():
   r=json.loads(rp.read_text()) if rp.exists() else {}
   if r.get('status')=='completed':continue
   write(ROOT/'results/status.json',{'time':now(),'queue_pid':os.getpid(),'status':'requires_review','case':j['id'],'reason':'Existing non-complete attempt requires child-process accounting; no automatic retry.'});return
  with open(ROOT/'logs'/f"{j['id']}.log",'x',buffering=1) as log:
   child=subprocess.Popen([sys.executable,str(ROOT/'scripts/replay.py'),j['id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
   while child.poll() is None:
    write(ROOT/'results/status.json',{'time':now(),'queue_pid':os.getpid(),'status':'running','case':j['id'],'child_pid':child.pid,'pending':[x['id'] for x in jobs if not (ROOT/'runs'/x['id']).exists()]});time.sleep(15)
   rc=child.returncode
  if rc!=0:
   write(ROOT/'results/status.json',{'time':now(),'queue_pid':os.getpid(),'status':'requires_review','case':j['id'],'returncode':rc});return
 write(ROOT/'results/status.json',{'time':now(),'queue_pid':os.getpid(),'status':'loading_jobs_complete_review_pending'})
if __name__=='__main__':main()
