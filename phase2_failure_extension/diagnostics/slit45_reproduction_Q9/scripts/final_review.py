"""Complete Q9a review from preserved trajectories; no simulations or manuscript edits."""
from pathlib import Path
from datetime import datetime, timezone
import json,csv,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare import ROOT, STUDY, PARENT, regular_path, replay_path, matched

def csv_write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 original=regular_path('Phase 1 single shot',ROOT/'inputs'/PARENT,True)
 replay=replay_path({'id':'R2_original_single'})
 primary=regular_path('Phase 2 longer loading',STUDY/'runs/ext__S2_slit_45deg')
 half=regular_path('Phase 2 half strain step',STUDY/'runs/res__S2_slit_45deg')
 batch=replay_path({'id':'R1_original_batch4'})
 paths=[original,replay,primary,half,batch]
 names=['Phase 1 single shot','45° rerun, original rules','Phase 2 longer loading','Phase 2 half strain step','Batch replay (suspended)']
 colors=['#313b45','#9466a7','#287ba5','#58a495','#df810f']
 styles=['-','--','-','--',':']
 record=json.loads((ROOT/'runs/R2_original_single/record.json').read_text())
 assert record['status']=='completed' and record['termination']=='max strain reached'
 checks=json.loads((ROOT/'source_checksums.json').read_text())
 assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h for rel,h in checks.items())
 summaries=[];points=[];interpolated=[]
 limit=.32
 def integral_to(p,x):
  xx=np.r_[p['strain'][p['strain']<x],x]
  return float(np.trapezoid(np.interp(xx,p['strain'],p['stress']),xx))
 for p,name in zip(paths,names):
  new=np.array([len(b-p['bonds'][0]) for b in p['bonds']]);p['new']=new
  ii=np.flatnonzero(new>0);dam=np.flatnonzero(p['broken']>0)
  end=int(np.flatnonzero(p['strain']<=limit+1e-10)[-1])
  row=dict(path=name,status=p['status'],frames=len(p['strain']),last_strain=float(p['strain'][-1]),
    maximum_recorded_stress_N_m=float(p['stress'].max()),strain_at_maximum=float(p['strain'][p['stress'].argmax()]),
    first_new_connection_strain=float(p['strain'][ii[0]]) if len(ii) else None,
    first_damage_strain=float(p['strain'][dam[0]]) if len(dam) else None,
    new_pairs_at_last_sample_at_or_below_032=int(new[end]),that_sample_strain=float(p['strain'][end]),
    cumulative_losses_at_that_sample=int(p['broken'][end]),
    force_flagged_frames=int((p['force']>.02).sum()),transverse_flagged_frames=int((abs(p['transverse'])>.005*16.0217663).sum()),
    endpoint_force_eV_A=float(p['force'][-1]),endpoint_transverse_N_m=float(p['transverse'][-1]))
  if p['strain'][-1]>=limit:
   row.update(stress_at_032_linear_interpolation_N_m=float(np.interp(limit,p['strain'],p['stress'])),
     integral_through_032_J_m2=integral_to(p,limit))
  else:row.update(stress_at_032_linear_interpolation_N_m=None,integral_through_032_J_m2=None)
  summaries.append(row)
  for k,x in enumerate(p['strain']):
   points.append(dict(path=name,status=p['status'],frame=k,strain=float(x),stress_N_m=float(p['stress'][k]),
     new_pairs_present=int(new[k]),cumulative_losses=int(p['broken'][k]),force_eV_A=float(p['force'][k]),
     transverse_stress_N_m=float(p['transverse'][k]),force_flag=bool(p['force'][k]>.02),
     force_provenance='CPU float64 reevaluation of saved coordinates' if p is original else 'recorded MPS float32 force'))
 grid=np.linspace(0,limit,1001);ref=np.interp(grid,original['strain'],original['stress'])
 comparison=[]
 for p,name in zip(paths[1:],names[1:]):
  if p['strain'][-1]<limit:continue
  yy=np.interp(grid,p['strain'],p['stress'])
  comparison.append(dict(path=name,interval=[0,limit],grid_points=len(grid),
     stress_rmse_N_m=float(np.sqrt(np.mean((yy-ref)**2))),
     normalized_stress_rmse=float(np.sqrt(np.mean((yy-ref)**2))/ref.max()),
     integral_relative_difference=integral_to(p,limit)/integral_to(original,limit)-1))
  for x,a,b in zip(grid,ref,yy):interpolated.append(dict(path=name,strain=float(x),original_stress_N_m=float(a),comparison_stress_N_m=float(b)))
 exact=[]
 for other in [original,primary,half,batch]:exact.extend(matched(other,replay))
 transitions=[]
 for other in [original,primary,half,batch]:
  rr=matched(other,replay)
  first=next((x for x in rr if x['edge_symmetric_difference']>0),None)
  transitions.append(dict(reference=other['name'],first_graph_difference_on_shared_saved_grid=first))
 # Inspect trace budgets, final residuals and neighbor-list events without rerunning forces.
 trace=[json.loads(x) for x in (ROOT/'runs/R2_original_single/minimizer_calls.jsonl').read_text().splitlines()]
 neighbor=[json.loads(x) for x in (ROOT/'runs/R2_original_single/neighbor_rebuilds.jsonl').read_text().splitlines()]
 calls=[c for c in trace if .16-1e-10<=c['target_eps']<=.18+1e-10]
 neigh=[c for c in neighbor if c['target_eps'] is not None and .16-1e-10<=c['target_eps']<=.18+1e-10]
 (ROOT/'results/R2_transition_trace.json').write_text(json.dumps({'minimizer_calls':calls,'neighbor_rebuilds':neigh},indent=2)+'\n')
 firstaudit=json.loads((ROOT/'results/first_damage_relaxation_audit.json').read_text())
 r2audit=next(c for c in firstaudit['cases'] if c['job']=='R2_original_single')
 # Independently verify recorded force norms for every saved R2 configuration.
 curve=json.loads((ROOT/'runs/R2_original_single/curve.json').read_text())
 for row in curve:
  f=np.load(ROOT/f'runs/R2_original_single/frames/{row["frame"]:05d}.npz')
  assert np.isclose(np.linalg.norm(f['force'],axis=1).max(),row['max_force_eV_A'],rtol=1e-6,atol=1e-7)
 csv_write(ROOT/'results/final_review_summaries.csv',summaries)
 csv_write(ROOT/'results/final_review_plotted_data.csv',points)
 csv_write(ROOT/'results/final_review_interpolated_stress.csv',interpolated)
 csv_write(ROOT/'results/R2_exact_matched_comparisons.csv',exact)
 result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),scope='Q9a: only R2 required; R1 suspended; R3 deferred',
   source_checksums_verified=len(checks),summaries=summaries,common_window_comparison=comparison,
   exact_saved_grid_divergences=transitions,first_damage_relaxation=r2audit,
   all_R2_saved_force_norms_verified=True,neighbor_rebuild_records=len(neighbor),minimizer_call_records=len(trace),
   stress_interpolation_note='Linear interpolation of scalar stress only, no extrapolation, 1001 equal points from strain 0 to 0.32 for RMSE. No configurations, bonds or damage events interpolated.',
   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'results/final_review.json').write_text(json.dumps(result,indent=2)+'\n')
 plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.linewidth':.6,
  'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axs=plt.subplots(2,2,figsize=(9.4,6.25));axs=axs.ravel()
 for p,name,color,style in zip(paths,names,colors,styles):
  x=p['strain'];m=x<=.32375+1e-10;t=(x>=.159)&(x<=.181)
  axs[0].plot(x[m],p['stress'][m],c=color,ls=style,lw=1.25,label=name)
  axs[1].plot(x[t],p['stress'][t],'.',c=color,ls=style,lw=1.0,ms=4)
  axs[2].step(x[m],p['new'][m],where='post',c=color,ls=style,lw=1.15)
  axs[3].plot(x[t],p['force'][t],'.',c=color,ls=style,lw=1.0,ms=4)
  if p in paths[:2]:
   axs[0].scatter(x[-1],p['stress'][-1],s=28,facecolors='white',edgecolors=color,zorder=6)
 for i,ax in enumerate(axs):
  ax.set_xlabel('engineering strain');ax.text(-.15,1.04,chr(97+i),transform=ax.transAxes,fontsize=12,fontweight='bold')
 axs[0].set(xlim=(0,.333),ylim=(-.3,20),ylabel='2D stress (N/m)')
 axs[1].set(xlim=(.159,.181),ylim=(.9,3.15),ylabel='2D stress (N/m)')
 axs[2].set(xlim=(0,.333),ylim=(-4,210),ylabel='currently present new connections')
 axs[3].set(xlim=(.159,.181),yscale='log',ylim=(.005,3),ylabel='maximum force (eV/Å)')
 axs[3].axhline(.02,c='#555555',ls=':',lw=.9);axs[3].text(.1805,.022,'target',ha='right',fontsize=7,color='#555555')
 for ax in [axs[1],axs[3]]:ax.set_xticks([.16,.17,.18])
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=8)
 fig.subplots_adjust(left=.095,right=.98,bottom=.09,top=.87,hspace=.40,wspace=.30)
 for ext in ['png','svg']:fig.savefig(ROOT/f'figures/single_replay_comparison.{ext}',dpi=220,bbox_inches='tight',pad_inches=.07)
 plt.close(fig)
 caption='''(a) Stress within the original loading window. Open endpoint circles mark the original prescribed total-strain stops, not observed failure. Existing Phase-2 paths are cropped to this window. The suspended batch is incomplete. (b) First-damage transition on the actual saved grids. (c) Currently present geometric connections absent from each path's initial graph (distance <2 Å); these are not electronic bond orders or cumulative distinct-pair counts. (d) Actual final force residuals near the transition; dotted line is the original 0.02 eV/Å target. Phase-1 residuals use CPU float64 reevaluation of saved coordinates; all newer paths use recorded MPS float32 forces. Lines connect saved samples; no configurations or graph events are interpolated. The separate tabulated stress-at-0.32 and common-window RMSE use explicitly documented scalar linear interpolation. The new original-rule single replay reproduces the high-stress response closely, while local transition details and intermediate relaxation remain imperfect. No curve is substituted into the manuscript.\n'''
 (ROOT/'figures/single_replay_comparison_caption.md').write_text(caption)
 a,b=summaries[:2];comp=comparison[0]
 report=f'''# Direct 45-degree rerun completed (Q9a)

The requested standalone rerun completed on September 26 at 14:17 UTC (10:17 a.m. EDT), after {record['wall_time_s']/60:.2f} minutes. It used the original initial geometry, original numerical parameters and original stopping rules. The outcome closely reproduces the original high-stress response. It is not an exact atom-by-atom reproduction or a convergence certificate.

## Numerical comparison

| Quantity | Phase 1 single shot | New original-rule single replay | Existing Phase-2 primary |
|---|---:|---:|---:|
| Maximum recorded stress (N/m) | {a['maximum_recorded_stress_N_m']:.5f} | {b['maximum_recorded_stress_N_m']:.5f} | {summaries[2]['maximum_recorded_stress_N_m']:.5f} |
| Strain at that maximum | {a['strain_at_maximum']:.5f} | {b['strain_at_maximum']:.5f} | {summaries[2]['strain_at_maximum']:.5f} |
| Stress at strain 0.32, linear interpolation (N/m) | {a['stress_at_032_linear_interpolation_N_m']:.5f} | {b['stress_at_032_linear_interpolation_N_m']:.5f} | {summaries[2]['stress_at_032_linear_interpolation_N_m']:.5f} |
| Integral through strain 0.32 (J/m²) | {a['integral_through_032_J_m2']:.6f} | {b['integral_through_032_J_m2']:.6f} | {summaries[2]['integral_through_032_J_m2']:.6f} |
| First recorded connection loss | {a['first_damage_strain']:.5f} | {b['first_damage_strain']:.5f} | {summaries[2]['first_damage_strain']:.5f} |
| New connections at last saved strain ≤0.32 | {a['new_pairs_at_last_sample_at_or_below_032']} at {a['that_sample_strain']:.5f} | {b['new_pairs_at_last_sample_at_or_below_032']} at {b['that_sample_strain']:.5f} | {summaries[2]['new_pairs_at_last_sample_at_or_below_032']} at {summaries[2]['that_sample_strain']:.5f} |

The endpoint maxima differ by {100*(b['maximum_recorded_stress_N_m']/a['maximum_recorded_stress_N_m']-1):.3f}%, but their endpoint strains differ by 0.00125, so this is not a same-strain strength comparison. Both are still rising at their prescribed total-strain stop; operational failure was not observed. At the common strain of 0.32, the scalar-interpolated stress difference is {100*(b['stress_at_032_linear_interpolation_N_m']/a['stress_at_032_linear_interpolation_N_m']-1):.3f}%. The normalized stress-curve RMSE over 0–0.32 is {100*comp['normalized_stress_rmse']:.3f}% (normalization: original maximum on the comparison grid), and the integral differs by {100*comp['integral_relative_difference']:.3f}%. Interpolation is only for scalar descriptive diagnostics, not bond events or configurations. These integrals are not fracture toughness.

## What explains the discrepancy, and what remains unresolved

1. The original black curve was correctly plotted. All 79 frozen source/input checksums pass. Initial coordinates and cells were verified identical in float32 before the run, and the force-field parameter checksum matched. Geometry seed differences and a wrong input design are not supported explanations.
2. A standalone run can recover the original high-stress response. Restoring the original batch layout is therefore not necessary to obtain that response; batch size alone does not explain why the existing Phase-2 primary follows a lower-stress path. A same-setting repeat was deferred, so repeatability statistics are not available.
3. Differences arise before the longer observation window. The new single replay includes an accepted 0.165 state, as the original did; the existing primary and suspended batch accepted 0.17 directly. Original and replay graphs match at every exactly shared saved strain through 0.17125. Their first shared-grid graph difference is at 0.17250: the replay has eight lost connections and the original has none (24 edges differ). Original first loss follows at 0.17375. The existing primary already differs from both at 0.17125. These are recorded adaptive-step and topology differences, not a different random seed or a consequence of later stopping.
4. The new trace identifies a concrete relaxation limitation at its first-damage state. The main relaxation passes with maximum force 0.01635 eV/Å, but the final transverse relaxation exhausts its 400-step budget and leaves 1.61340 eV/Å (target 0.02). The stored FIRE-converged flag refers to the main relaxation and does not certify the final state. All 51 saved force norms were independently checked against the saved arrays. Across this run, 8/51 frames exceed the force target and 19/51 exceed the transverse-stress target. Its final state meets both recorded targets. The original and existing Phase-2 paths also contain residual flags, with different force-evaluation provenance for the original audit.
5. Small coordinate/stress differences already exist after initial relaxation, before topology changes. Floating-point execution, neighbor rebuild history, adaptive bisection and finite transverse/minimizer budgets remain possible contributors. The recorded rebuild/minimizer traces are preserved; none isolates a unique causal trigger. The earlier synthetic neighbor-list guard issue is not established as the cause of these real trajectories. A close macroscopic replay does not prove a unique or converged physical path.

## Disposition and reproducibility

R2 is complete. R1 and its old scheduler remain intentionally SIGSTOP-suspended; R1 has 26 saved states through strain 0.20250 and is not a completed material response. R3 remains deferred. No further simulations have been launched or resumed. The Q9a follow-up is ready to pause after this review; additional causal interventions would need their own explicit, preserved branch.

The manuscript, Figure 10, original Figure S11, all original simulations and the hierarchy follow-up are unchanged by this diagnostic review. Figure 10 continues to show its existing Phase-2 paths; these diagnostic results have not been substituted.

Reproduce with the PyTorch environment's Python: `scripts/compare.py`, then `scripts/inspect_first_damage.py`, then `scripts/final_review.py`. Outputs: `results/final_review.json`, `results/final_review_summaries.csv`, `results/final_review_plotted_data.csv`, `results/final_review_interpolated_stress.csv`, `results/R2_exact_matched_comparisons.csv`, `results/R2_transition_trace.json`, and editable SVG/PNG `figures/single_replay_comparison` with its separate caption. Scientific trajectories and full checkpoints remain under `runs/`.
'''
 (ROOT/'results/single_replay_report.md').write_text(report)
 print(json.dumps({'common_window':comparison,'summary':summaries[1]},indent=2))

if __name__=='__main__':main()
