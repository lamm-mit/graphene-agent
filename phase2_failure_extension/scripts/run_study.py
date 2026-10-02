"""Durable study queue with a pilot review gate and atomic progress status."""
from study_common import ROOT,now,write_json,sha
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
(ROOT / 'results').mkdir(parents=True, exist_ok=True)
import fcntl,json,os,subprocess,sys,time

def main():
    lock=open(ROOT/'logs/queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    jobs=json.loads((ROOT/'protocol/jobs.json').read_text());running={}
    checks=json.loads((ROOT/'protocol/execution_code_checksums.json').read_text())
    for p,h in checks.items():
        if sha(ROOT/p)!=h:raise RuntimeError('Frozen execution code changed: '+p)
    stages=['pilot','production','resolution'];last_analysis=0
    while True:
        records={}
        for job in jobs:
            p=ROOT/'runs'/job['case_id']/'record.json'
            if p.exists():records[job['case_id']]=json.loads(p.read_text())
        for case,(proc,stream,started) in list(running.items()):
            rc=proc.poll()
            if rc is not None:
                stream.close();del running[case]
                p=ROOT/'runs'/case/'record.json'
                if not p.exists():write_json(p,dict(case_id=case,status='execution_error',error='Child exited before creating record',returncode=rc,finished_utc=now()))
                elif json.loads(p.read_text()).get('status')=='running':
                    r=json.loads(p.read_text());r.update(status='execution_error',error='Process exited without finalizing',returncode=rc,finished_utc=now());write_json(p,r)
                print(now(),'finished',case,'returncode',rc,flush=True)
        terminal=lambda j:j['case_id'] in records and records[j['case_id']].get('status')!='running'
        active_stage=next((s for s in stages if not all(terminal(j) for j in jobs if j['stage']==s)),None)
        ctl=json.loads((ROOT/'protocol/queue_control.json').read_text())
        gate=None;blocked=None
        if active_stage in ['production','resolution']:
            gp=ROOT/'protocol/production_gate.json'
            gate=json.loads(gp.read_text()) if gp.exists() else None
            if not gate or not gate.get('approved'):blocked='Pilot review required before production.'
        if ctl.get('paused'):blocked='Queue paused by control file.'
        for j in jobs:
            c=j['case_id']
            if c in records and records[c].get('status')=='running' and c not in running:
                blocked='Orphan running record requires explicit review: '+c
        if not blocked and active_stage:
            for j in jobs:
                if len(running)>=int(ctl['max_workers']):break
                case=j['case_id']
                if j['stage']!=active_stage or case in records or case in running or (ROOT/'runs'/case).exists():continue
                stream=open(ROOT/'logs'/f'{case}.log','a',buffering=1)
                env=dict(os.environ,PYTHONUNBUFFERED='1')
                proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_case.py'),case],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
                running[case]=(proc,stream,time.time());print(now(),'started',case,'pid',proc.pid,flush=True)
        progress=[]
        for case,(proc,_,started) in running.items():
            p=ROOT/'runs'/case/'progress.json';v=json.loads(p.read_text()) if p.exists() else {}
            progress.append(dict(case_id=case,pid=proc.pid,elapsed_s=time.time()-started,**v))
        statuses={}
        for j in jobs:
            state=records.get(j['case_id'],{}).get('status','pending');statuses[state]=statuses.get(state,0)+1
        write_json(ROOT/'results/status.json',dict(updated_utc=now(),queue_pid=os.getpid(),active_stage=active_stage,blocked=blocked,max_workers=ctl['max_workers'],counts=statuses,running=progress))
        if time.time()-last_analysis>120 and (ROOT/'scripts/analyze.py').exists():
            with open(ROOT/'logs/analysis.log','a') as f:subprocess.run([sys.executable,str(ROOT/'scripts/analyze.py'),'--partial'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            last_analysis=time.time()
        if active_stage is None and not running:
            if (ROOT/'scripts/analyze.py').exists():
                with open(ROOT/'logs/analysis.log','a') as f:subprocess.run([sys.executable,str(ROOT/'scripts/analyze.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            print(now(),'All scheduled jobs accounted for; scientific/numerical review still required.',flush=True);break
        time.sleep(15)

if __name__=='__main__':main()
