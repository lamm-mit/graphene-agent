"""Extract first-damage minimizer history, retaining final-state residual provenance."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGES={285:'trial relaxation',303:'full relaxation',193:'transverse relaxation',331:'extra no-stall relaxation'}

def main():
    output=[]
    for job in json.loads((ROOT/'jobs.json').read_text()):
        folder=ROOT/'runs'/job['id']
        if not (folder/'curve.json').exists():continue
        curve=json.loads((folder/'curve.json').read_text())
        damaged=[x for x in curve if x['n_broken_cum']>0]
        if not damaged:continue
        frame=damaged[0];strain=frame['eps_x'];target=job['target_index']
        text=(folder/'minimizer_calls.jsonl').read_text()
        calls=[json.loads(x) for x in text.splitlines(keepends=True) if x.endswith('\n')]
        # Rejected earlier trials can have the same strain. Start with the final
        # trial at this strain; subsequent trials belong to later accepted steps.
        starts=[i for i,c in enumerate(calls) if c['callsite_line']==285 and c['active'][target]
                and abs(c['target_eps']-strain)<1e-10]
        assert starts
        start=starts[-1];end=next((i for i in range(start+1,len(calls)) if calls[i]['callsite_line']==285),len(calls))
        selected=[dict(time=c['time'],stage=STAGES.get(c['callsite_line'],str(c['callsite_line'])),
                       source_line=c['callsite_line'],strain=c['target_eps'],transverse_strain=c['target_eps_y'],
                       iterations=c['iterations'],budget=c['budget'],converged=bool(c['converged'][target]),
                       force_residual_eV_A=c['fmax'][target],target_motion_max_component_A=c['target_max_component_motion_A'])
                  for c in calls[start:end] if c['active'][target]]
        with np.load(folder/'frames'/f"{frame['frame']:05d}.npz") as z:
            residual=float(np.linalg.norm(z['force'],axis=1).max())
        assert np.isclose(residual,frame['max_force_eV_A'],atol=1e-7,rtol=1e-7)
        output.append(dict(job=job['id'],first_damage_frame=frame,independent_saved_force_norm_eV_A=residual,
                           recorded_fire_flag=frame['fire_converged'],
                           final_force_meets_target=residual<=.02,sequence=selected,
                           trace_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           note='Sequence starts after the last trial at the accepted first-damage strain; earlier rejected trials at the same strain are retained in the raw log. The saved FIRE flag refers to the main relaxation and does not certify later transverse relaxations.'))
    result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),cases=output,
                scientific_protocol_changed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'results/first_damage_relaxation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'case':x['job'],'first_damage_strain':x['first_damage_frame']['eps_x'],
                       'final_force_eV_A':x['independent_saved_force_norm_eV_A'],
                       'stages':len(x['sequence'])} for x in output],indent=2))

if __name__=='__main__':main()
