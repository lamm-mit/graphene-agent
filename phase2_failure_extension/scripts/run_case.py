"""Execute one isolated, parent-linked rerun; persist every accepted frame."""
from __future__ import annotations
import argparse, csv, dataclasses, json, os, platform, subprocess, sys, time, traceback
from pathlib import Path
from study_common import ROOT,CODE,EV,now,write_json,sha,stop_event,endpoint_metrics,paired_results,load_curve
sys.path.insert(0,str(CODE))
import numpy as np
import torch, ase
from ase.io import read,write
from potentials.rebo2scr.pytorch.torch_rebo2scr import TorchRebo2Scr
from simulation.system import BatchedSystem
from simulation.aqs.aqs import AQSRunner,AQSConfig
from analysis.metrics import compute_metrics

def run(job):
    started=now(); t0=time.time()
    case=ROOT/'runs'/job['case_id'];case.mkdir(parents=True,exist_ok=False)
    (case/'frames').mkdir()
    write_json(case/'job.json',job)
    base=ROOT/'inputs/baseline/runs'/job['parent_run_id']
    parent=json.loads((base/'record.json').read_text())
    atoms=read(base/'initial.extxyz')
    assert len(atoms)==parent['n_atoms']
    assert list(atoms.pbc)==parent['boundary_conditions']['pbc']
    for module in ['potentials','simulation','analysis']:
        assert (CODE/module).is_dir()
    # For reproducibility, use the per-parent settings, not campaign defaults.
    cfg_dict=dict(parent['aqs_config']);cfg_dict.update(job['overrides'])
    cfg=AQSConfig(**cfg_dict)
    device=job.get('device',parent['device']);precision=job.get('precision',parent['precision'])
    dtype=getattr(torch,precision)
    eng=TorchRebo2Scr(device=device,dtype=dtype)
    assert eng.info.parameter_checksum==parent['engine']['parameter_checksum']
    assert Path(sys.modules['simulation.aqs.aqs'].__file__).resolve().is_relative_to(CODE)
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,ase=ase.__version__,numpy=np.__version__,platform=platform.platform(),
        cpu=(platform.processor() or platform.machine()),device=str(eng.device),precision=precision)
    record=dict(case_id=job['case_id'],name=parent['name'],parent_run_id=parent['run_id'],source_phase=job['source_phase'],
        source_campaign=parent['campaign'],campaign='phase2_failure_extension',role=job['role'],stage=job['stage'],status='running',started_utc=started,
        environment=environment,engine=dataclasses.asdict(eng.info),aqs_config=cfg_dict,parent_aqs_config=parent['aqs_config'],
        parent_record_sha256=sha(base/'record.json'),initial_geometry_sha256=sha(base/'initial.extxyz'),
        protocol_sha256=sha(ROOT/'protocol/protocol_v1.json'),code_snapshot_sha256=sha(ROOT/'protocol/execution_code_checksums.json'),
        n_atoms=len(atoms),parent_descriptors=parent['descriptors'],parent_family=parent['family'],
        independence_note='Paired rerun of the parent design; not a new independent material sample.')
    write_json(case/'record.json',record);write(case/'initial.extxyz',atoms)
    fields=['eps_x','eps_y','sigma_xx','sigma_yy','sigma_xy','energy','n_bonds','n_broken_new','n_formed_new','n_broken_cum','fire_iterations','fire_converged','transverse_iterations','spanning','largest_fragment','n_fragments','wall_time','attempted_step','max_force_eV_A']
    history={k:[] for k in fields}
    curve_file=open(case/'stress_strain.csv','w',buffering=1)
    writer=csv.DictWriter(curve_file,fieldnames=fields);writer.writeheader()
    virtual=None
    def frame_cb(runner,b,fr):
        nonlocal virtual
        v=dataclasses.asdict(fr);sigma=v.pop('sigma');v.update(sigma_xx=sigma[0],sigma_yy=sigma[1],sigma_xy=sigma[2]);v['fire_converged']=int(v['fire_converged']);v['spanning']=int(v['spanning'])
        writer.writerow({k:v[k] for k in fields});curve_file.flush()
        for k in fields:history[k].append(v[k])
        f=len(history['eps_x'])-1
        tmp=case/'frames'/f'{f:05d}.tmp.npz';dest=case/'frames'/f'{f:05d}.npz'
        np.savez_compressed(tmp,positions=runner.positions_frames[b][-1],cell=runner.cell_frames[b][-1],
            peratom_energy=runner.peratom_energy_frames[b][-1],peratom_virial=runner.peratom_virial_frames[b][-1],
            bonds=runner.bond_frames[b][-1],coordination=runner.coordination_frames[b][-1])
        os.replace(tmp,dest)
        if virtual is None:
            virtual=stop_event(history,parent['aqs_config'])
            if virtual is not None:write_json(case/'virtual_original_stop.json',dict(event=virtual,metrics=endpoint_metrics(history,virtual)))
        write_json(case/'progress.json',dict(updated_utc=now(),frames=f+1,attempted_step=fr.attempted_step,strain=fr.eps_x,stress_N_m=float(fr.sigma[0])*EV,
            force_residual_eV_A=fr.max_force_eV_A,transverse_stress_N_m=float(fr.sigma[1])*EV,wall_time_s=time.time()-t0,
            virtual_original_stop=virtual,spanning=fr.spanning,cumulative_broken=fr.n_broken_cum))
    try:
        system=BatchedSystem([atoms],eng,names=[parent['name']])
        result=AQSRunner(eng,system,cfg,log=lambda x:print(x,flush=True),on_frame=frame_cb).run()[0]
        curve_file.close()
        original=load_curve(base/'stress_strain.csv');new=load_curve(case/'stress_strain.csv')
        paired=paired_results(parent,original,new,cfg_dict,result['termination'])
        finite=all(np.isfinite(new[k]).all() for k in ['eps_x','sigma_xx','sigma_yy','energy','max_force_eV_A'])
        driver_failure=result['termination'] in ['not spanning (fractured)','stress collapsed']
        assert paired['C_extended']['failure_observed']==driver_failure, 'Chronological metric/driver disagreement'
        if driver_failure:assert paired['C_extended']['endpoint_index']==len(new['eps_x'])-1
        legacy=compute_metrics(result)
        quality=dict(all_recorded_values_finite=finite,
            force_tolerance_eV_A=cfg.fmax,frames_above_force_tolerance=int(np.sum(new['max_force_eV_A']>cfg.fmax)),
            maximum_recorded_force_residual_eV_A=float(new['max_force_eV_A'].max()),
            frames_above_transverse_tolerance=int(np.sum(abs(new['sigma_yy'])>cfg.sigma_t_tol)),
            maximum_transverse_stress_residual_N_m=float(np.max(abs(new['sigma_yy']))*EV),
            numerical_termination='numerical' in result['termination'],driver_and_chronological_metric_agree=True)
        np.savez_compressed(case/'trajectory.npz',positions=np.stack(result['positions']),cells=np.stack(result['cells']),
            peratom_energy=np.stack(result['peratom_energy']),peratom_virial=np.stack(result['peratom_virial']),
            coordination=np.stack(result['coordination']),L0=result['L0'],**{k:np.asarray(v) for k,v in history.items()})
        write_json(case/'damage_events.json',result['broken_bond_events'])
        write(case/'relaxed.extxyz',system.to_atoms(0,positions=result['positions'][0],cell=result['cells'][0]))
        write(case/'final.extxyz',system.to_atoms(0,positions=result['positions'][-1],cell=result['cells'][-1]))
        record.update(status='completed' if finite and not quality['numerical_termination'] else 'numerical_limitation',finished_utc=now(),wall_time_s=time.time()-t0,
            termination=result['termination'],paired=paired,quality=quality,legacy_metrics_for_comparison=legacy,stats=result['stats'])
        write_json(case/'record.json',record)
        print(json.dumps(dict(case=job['case_id'],status=record['status'],endpoint=paired['C_extended'],reproduction_flags=paired['reproduction_flags'])),flush=True)
    except BaseException as error:
        curve_file.close();record.update(status='execution_error',finished_utc=now(),error=repr(error),traceback=traceback.format_exc(),wall_time_s=time.time()-t0)
        write_json(case/'record.json',record);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case_id');a=p.parse_args()
    jobs=json.loads((ROOT/'protocol/jobs.json').read_text())
    job=next(j for j in jobs if j['case_id']==a.case_id)
    run(job)
