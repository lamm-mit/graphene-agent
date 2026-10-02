"""Q9a: run only the requested 45-degree single case; never resume the batch queue."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def main():
    lock=open(ROOT/'logs/priority_single.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    control=json.loads((ROOT/'results/execution_control.json').read_text())
    for pid in control['suspended_pids']:
        state=subprocess.check_output(['ps','-p',str(pid),'-o','state='],text=True).strip()
        assert 'T' in state,(pid,state)
    case='R2_original_single'
    assert not (ROOT/'runs'/case).exists(),'Never overwrite an existing attempt'
    with (ROOT/'logs'/f'{case}.log').open('x',buffering=1) as log:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts/replay.py'),case],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
        while child.poll() is None:
            write(ROOT/'results/status.json',dict(time=now(),status='running',case=case,
                supervisor_pid=os.getpid(),child_pid=child.pid,suspended_batch=control['suspended_pids'],
                deferred=['R3_original_single_repeat'],amendment='amendment_Q9a.json'))
            time.sleep(10)
        code=child.returncode
    write(ROOT/'results/status.json',dict(time=now(),status='single_replay_complete_review_pending' if code==0 else 'requires_review',
        case=case,returncode=code,supervisor_pid=os.getpid(),suspended_batch=control['suspended_pids'],
        deferred=['R3_original_single_repeat'],amendment='amendment_Q9a.json',
        note='No further simulations start automatically; original batch remains suspended.'))
if __name__=='__main__':main()
