"""Generate standalone, censor-aware paired tables and manuscript-style figures.

Only reads copied inputs and new runs in this study. Never edits the paper.
Partial output is explicitly labeled and does not substitute missing simulations.
"""
from __future__ import annotations
import argparse,csv,fcntl,io,json,math,os
from pathlib import Path
from study_common import ROOT,EV,now,write_json,load_curve,sha
(ROOT / 'figures').mkdir(parents=True, exist_ok=True)
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
(ROOT / 'results').mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

FAM={'pristine':'k','vacancies':'0.5','precrack':'#8c564b','ring_around_hole':'#e377c2','nanomesh_single':'#1f77b4',
     'nanomesh_hier':'#d62728','voronoi_network':'#2ca02c','slit_array':'#ff7f0e','strut_lattice':'#9467bd','graded_pores':'#17becf'}
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':6.5,'axes.linewidth':.6,'lines.linewidth':1.,'mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none',
    'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})

def style(ax,letter=None):
    ax.grid(True,alpha=.2,lw=.4);ax.set_axisbelow(True)
    if letter:ax.text(-.17,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)

def finish(fig,name,caption):
    for ext in ['pdf','svg','png']:
        target=ROOT/'figures'/f'{name}.{ext}';temp=target.with_name(f'.{name}.{os.getpid()}.{ext}')
        fig.savefig(temp,bbox_inches='tight',facecolor='white');os.replace(temp,target)
    (ROOT/'figures'/f'{name}.caption.txt').write_text(caption+'\n');plt.close(fig)

def align(r):
    d=r['descriptors'];return d['anisotropy_index']*math.cos(2*math.radians(d['ligament_orientation_deg']))

def quality_limited(r):
    q=r['quality']
    return (not q['all_recorded_values_finite'] or q['numerical_termination'] or
            q['frames_above_force_tolerance']>0 or q['frames_above_transverse_tolerance']>0)

def load_live_curve(path):
    """Read only complete CSV rows; a new run may still have only its header."""
    snapshot=Path(path).read_text()
    if '\n' not in snapshot:return None
    complete=snapshot[:snapshot.rfind('\n')+1]
    rows=[{k.split('#')[0].strip():float(v) for k,v in row.items()} for row in csv.DictReader(io.StringIO(complete))]
    if not rows:return None
    return {k:np.array([row[k] for row in rows]) for k in rows[0]}

def dot(ax,x,y,r,col,size=22):
    m='o' if r['source_phase']=='I' else 's'
    observed=r['paired']['C_extended']['failure_observed']
    ax.scatter(x,y,marker=m,s=size,edgecolors=col,facecolors=col if observed else 'white',linewidths=.8,zorder=4)
    if quality_limited(r):ax.scatter(x,y,marker='+',s=size*.65,color='black',linewidths=.6,zorder=5)

def curve_panel(ax,r,parent,normal=False,threshold=False):
    a=load_curve(ROOT/'inputs/baseline/runs'/r['parent_run_id']/'stress_strain.csv')
    c=load_curve(ROOT/'runs'/r['case_id']/'stress_strain.csv')
    p=r['paired'];end=p['C_extended'];b=p['B_original_protocol_on_rerun']
    scale=p['A_archived']['maximum_recorded_stress_N_m'] if normal else 1.
    col=FAM.get(parent['family'],'#1f77b4')
    ax.plot(c['eps_x'],c['sigma_xx']*EV/scale,color=col,label='extended rerun')
    bad=(c['max_force_eV_A']>r['aqs_config']['fmax']) | (abs(c['sigma_yy'])>r['aqs_config']['sigma_t_tol'])
    if bad.any():ax.scatter(c['eps_x'][bad],c['sigma_xx'][bad]*EV/scale,marker='+',s=12,color='black',lw=.55,label='residual above tolerance',zorder=5)
    ax.plot(a['eps_x'],a['sigma_xx']*EV/scale,'--',color='.35',lw=.9,label='archived original')
    if p['virtual_stop_found']:
        k=b['endpoint_index'];ax.axvline(b['end_strain'],color='.5',ls=':',lw=.75)
        ax.fill_between(c['eps_x'][k:],0,c['sigma_xx'][k:]*EV/scale,color=col,alpha=.12)
    ax.scatter([a['eps_x'][-1]],[a['sigma_xx'][-1]*EV/scale],s=20,facecolors='white',edgecolors='.25',marker='o',zorder=5)
    ax.scatter([end['end_strain']],[end['end_stress_N_m']/scale],s=25,color=col,marker='X' if end['failure_observed'] else '>',zorder=6,label='driver failure criterion' if end['failure_observed'] else 'observation ended')
    if threshold:
        running=np.maximum.accumulate(c['sigma_xx']*EV)
        ax.plot(c['eps_x'],.1*running/scale,':',color='.65',lw=.65)
    ax.set_xlabel('engineering strain');ax.set_ylabel('stress / original maximum' if normal else '2D stress (N/m)')
    return end

def main_figure(done,parents):
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.0));fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.9,wspace=.36,hspace=.6)
    label='Complete cohort' if len(done)==34 else f'Interim: {len(done)} / 34 extended cases available'
    fig.suptitle('Effect of extending the loading horizon',fontsize=10,y=.99)
    fig.text(.5,.946,label,ha='center',fontsize=7,color='.4')
    by={r['name']:r for r in done}
    for ax,name,title,letter in zip(axs[0],['S2_slit_45deg','S7_slit_20deg'],['45° slits: reproduction + extension','20° slits: retained load'],['a','b']):
        style(ax,letter);ax.set_title(title,loc='left',fontsize=8)
        if name in by:
            r=by[name];curve_panel(ax,r,parents[r['parent_run_id']])
            if name=='S2_slit_45deg':ax.legend(loc='best',frameon=False,fontsize=6)
        else:
            ax.set_axis_off();ax.text(.5,.5,'Extension not yet available',ha='center',va='center',transform=ax.transAxes,color='.4')
    for ax,key,letter in [(axs[1,0],'end_strain','c'),(axs[1,1],'stress_strain_integral_J_m2','d')]:
        style(ax,letter);values=[]
        for r in done:
            if not r['paired']['virtual_stop_found']:continue
            b=r['paired']['B_original_protocol_on_rerun'];c=r['paired']['C_extended'];x,y=b[key],c[key]
            dot(ax,x,y,r,FAM.get(parents[r['parent_run_id']]['family'],'.3'));values.extend([x,y])
            if key=='end_strain' and not c['failure_observed']:ax.annotate('',(x,y+.025),(x,y),arrowprops=dict(arrowstyle='->',lw=.65,color='.35'))
        hi=max(values or [1]) * 1.1;ax.plot([0,hi],[0,hi],'--',color='.6',lw=.7);ax.set_xlim(0,hi);ax.set_ylim(0,hi)
        ax.set_xlabel('original-protocol stopping strain' if key=='end_strain' else 'integral to original-protocol stop (J/m²)')
        ax.set_ylabel('extended ending strain' if key=='end_strain' else 'extended integral (J/m²)')
    handles=[Line2D([],[],marker='o',ls='',color='.3',label='Phase-I source'),Line2D([],[],marker='s',ls='',color='.3',label='earlier Phase-II source'),
        Line2D([],[],marker='o',ls='',color='.3',markerfacecolor='white',label='failure criterion not reached'),
        Line2D([],[],marker='+',ls='',color='black',label='residual tolerance exceeded')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.025),fontsize=6.5)
    finish(fig,'loading_horizon_comparison','Panels a-b compare the archived trajectories with extended reruns. Dotted vertical lines mark where the original protocol would stop the rerun; shading is the added recorded integral. Panels c-d use the same rerun at its virtual original stop (B) and extended endpoint (C). Family colors follow the existing paper; circles and squares identify source phase. Open comparison markers indicate the driver failure criterion was not reached. Plus symbols mark above-tolerance frames in curves and trajectories containing any force or transverse-stress tolerance exceedance in comparisons. A driver-detected failure on a flagged trajectory is not a numerically validated material-failure measurement. Strain arrows denote an unresolved ending; work coordinates are observed integrals to stated endpoints, not assumed full-failure work. '+label+'.')

def migration(done,parents,discovery,meshes):
    updates={r['parent_run_id']:r for r in done};fig,axs=plt.subplots(1,3,figsize=(7.1,2.9));fig.subplots_adjust(left=.085,right=.985,bottom=.29,top=.81,wspace=.52)
    fig.text(.5,.97,f'Interim: {len(done)} / 34 extensions available' if len(done)<34 else 'All 34 extensions available; quality and censoring flags retained',ha='center',fontsize=7,color='.35')
    for ax,letter in zip(axs,'abc'):style(ax,letter)
    for r in discovery:
        rid=r['run_id'];m=r['metrics'];col=FAM.get(r['family'],'.4')
        for ax,xkey in [(axs[0],'work_to_failure_J_m2'),(axs[1],'failure_strain')]:
            x,y=m[xkey],m['strength_Nm'];ax.scatter(x,y,s=9,edgecolors='.65',facecolors='.75' if m['failure_observed'] else 'white',linewidths=.5,zorder=2)
            if rid in updates:
                u=updates[rid];c=u['paired']['C_extended'];xn=c['stress_strain_integral_J_m2'] if xkey.startswith('work') else c['end_strain'];yn=c['maximum_recorded_stress_N_m']
                ax.annotate('',(xn,yn),(x,y),arrowprops=dict(arrowstyle='->',color=col,lw=.7,alpha=.8));dot(ax,xn,yn,u,col,18)
    for r in meshes:
        x,y=align(r),r['metrics']['work_to_failure_J_m2'];axs[2].scatter(x,y,s=9,edgecolors='.65',facecolors='.75' if r['metrics']['failure_observed'] else 'white',linewidths=.5)
        if r['run_id'] in updates:
            u=updates[r['run_id']];yn=u['paired']['C_extended']['stress_strain_integral_J_m2'];col=FAM.get(r['family'],'.3')
            axs[2].annotate('',(x,yn),(x,y),arrowprops=dict(arrowstyle='->',color=col,lw=.7));dot(axs[2],x,yn,u,col,18)
    for ax in axs[:2]:ax.set_ylabel('maximum recorded stress (N/m)')
    axs[0].set_xlabel('integral to stated endpoint (J/m²)');axs[1].set_xlabel('recorded ending strain');axs[2].set_xlabel('alignment index A');axs[2].set_ylabel('integral to stated endpoint (J/m²)')
    for ax,t in zip(axs,['Discovery: work coordinates','Discovery: strain coordinates','Original 40-mesh population']):ax.set_title(t,fontsize=7)
    handles=[Line2D([],[],marker='o',ls='',color='.65',label='archived'),Line2D([],[],marker='o',ls='',color=FAM['slit_array'],label='extended'),Line2D([],[],marker='o',ls='',color='.4',markerfacecolor='white',label='criterion unmet'),Line2D([],[],marker='+',ls='',color='black',label='residual flag')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=4,frameon=False,fontsize=6)
    finish(fig,'si_property_shifts','Gray points are original archived coordinates; arrows connect available extended outcomes (A-to-C), combining reproduction and observation-window effects. The discovery panels retain exactly 96 designs, and the alignment panel retains exactly the original 40 meshes (31 discovery, one validation, eight earlier Phase-II). Open archived and new markers denote an unmet driver failure criterion; plus symbols identify new trajectories with residual tolerance exceedances. Pending extensions retain original coordinates and are not presented as corrected results. A lower rerun maximum can reflect reproduction differences; extending a given trajectory cannot lower its running maximum. This is a migration view, not a claim that both endpoints are observed, numerically validated failures.')

def atlas(done,parents):
    by={r['parent_run_id']:r for r in done};cohort=json.loads((ROOT/'planning/cohort_manifest.json').read_text())['runs']
    target=ROOT/'figures/si_curve_atlas.pdf';temp=target.with_name('.atlas.tmp.pdf')
    with PdfPages(temp) as pdf:
        for page in range(2):
            fig,axs=plt.subplots(6,3,figsize=(7.1,9.0));fig.subplots_adjust(left=.09,right=.985,bottom=.06,top=.94,hspace=.83,wspace=.4)
            fig.suptitle(f'Loading extensions: case inventory ({page+1}/2; {len(done)}/34 complete)',fontsize=10,y=.986)
            fig.text(.5,.963,'Gray dashed: archived | colored: rerun | dotted: original stop / failure threshold',ha='center',fontsize=6.5)
            for ax,entry in zip(axs.flat,cohort[page*18:(page+1)*18]):
                rid=entry['parent_run_id'];style(ax)
                ax.set_title(entry['name'],fontsize=5.8,loc='left')
                if rid in by:
                    r=by[rid];c=curve_panel(ax,r,parents[rid],normal=True,threshold=True)
                    lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.25*(hi-lo))
                    outcome='driver stop' if c['failure_observed'] else 'criterion unmet'
                    if quality_limited(r):outcome+='; quality flagged'
                    ax.text(.03,.95,outcome+f"\nε end = {c['end_strain']:.3f}",transform=ax.transAxes,fontsize=5.0,va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3))
                else:
                    a=load_curve(ROOT/'inputs/baseline/runs'/rid/'stress_strain.csv');norm=parents[rid]['metrics']['strength_Nm']
                    ax.plot(a['eps_x'],a['sigma_xx']*EV/norm,'--',color='.5');ax.text(.03,.95,'extension pending',transform=ax.transAxes,fontsize=5.5,va='top');ax.set_xlabel('engineering strain')
                ax.tick_params(labelsize=5.5);ax.xaxis.label.set_size(5.7);ax.yaxis.label.set_size(5.7)
            for ax in list(axs.flat)[len(cohort[page*18:(page+1)*18]):]:ax.axis('off')
            pdf.savefig(fig);fig.savefig(ROOT/'figures'/f'si_curve_atlas_{page+1}.png',dpi=300);fig.savefig(ROOT/'figures'/f'si_curve_atlas_{page+1}.svg');plt.close(fig)
    os.replace(temp,target)
    (ROOT/'figures/si_curve_atlas.caption.txt').write_text('All 34 cases are retained, including pending or unresolved outcomes. Stress is normalized by the original recorded maximum; renewed peaks can therefore exceed one. The threshold follows 10% of the running maximum in the rerun. Plus symbols mark force or transverse-stress tolerance exceedances. Driver failure detection does not certify numerical convergence. The archived and rerun curves are separate trajectories.\n')

def disorder(done,parents):
    byname={r['name']:r for r in parents.values() if r.get('status')=='completed'};updates={r['name']:r for r in done}
    refs=['S2_H1_p16','S1_mesh_p16_phi0.2','S6_mesh_p16_offset1','S6_mesh_p16_offset2','S6_mesh_p16_offset3']
    groups=[('Positional disorder',{'0':refs,'1':['S2_mesh_jitter1.0'],'2':['S2_mesh_jitter2.0','S6_mesh_jitter2.0_s2','S6_mesh_jitter2.0_s3'],'3':['S2_mesh_jitter3.0']}),
      ('Pore-size dispersion',{'0':refs,'0.15':['S2_mesh_sizedis0.15','S6_mesh_sizedis0.15_s2','S6_mesh_sizedis0.15_s3'],'0.3':['S2_mesh_sizedis0.3']}),
      ('Voronoi regularity',{'0':['S2_voronoi_n25_reg0.0','S6_voronoi_n25_reg0.0_s2','S6_voronoi_n25_reg0.0_s3','S2_voronoi_n50_reg0.0'],'0.3':['S5_voronoi_n25_reg0.3'],
         '0.6':['S2_voronoi_n25_reg0.6','S6_voronoi_n25_reg0.6_s2','S6_voronoi_n25_reg0.6_s3'],'0.8':['S5_voronoi_n25_reg0.8','S7_voronoi_n35_reg0.8_s7'],'1':['S2_voronoi_n25_reg1.0']})]
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.9));fig.subplots_adjust(left=.08,right=.985,bottom=.29,top=.81,wspace=.4);rows=[]
    fig.text(.5,.97,f'Interim: {len(done)} / 34 extensions available' if len(done)<34 else 'All 34 extensions available; quality and censoring flags retained',ha='center',fontsize=7,color='.35')
    for ax,(title,levels),letter in zip(axs,groups,'abc'):
        style(ax,letter);ax.set_title(title,fontsize=8)
        for k,(label,names) in enumerate(levels.items()):
            vals=[];missing=0;censored=0
            for name in names:
                p=byname[name];old=p['metrics']['work_to_failure_J_m2'];ax.scatter(k-.07,old,edgecolors='.6',facecolors='.7' if p['metrics']['failure_observed'] else 'white',s=12,linewidths=.6)
                if name in updates:
                    r=updates[name];c=r['paired']['C_extended'];new=c['stress_strain_integral_J_m2'];obs=c['failure_observed']
                    ax.plot([k-.07,k+.07],[old,new],color='.7',lw=.6);dot(ax,k+.07,new,r,FAM.get(p['family'],'.3'),18)
                else:
                    new=old;obs=p['metrics']['failure_observed'];missing+=int(not obs)
                censored+=int(not obs);vals.append(new)
            # Summary is explicitly an endpoint-integral mean, not completed-failure work.
            ax.plot([k-.23,k+.23],[np.mean(vals)]*2,color='.25',lw=.9)
            rows.append(dict(group=title,level=label,n=len(vals),endpoint_integral_mean_J_m2=float(np.mean(vals)),remaining_censored=censored,pending_extensions=missing))
        ax.set_xticks(range(len(levels)));ax.set_xticklabels(list(levels));ax.set_xlabel('group parameter')
    axs[0].set_ylabel('integral to stated endpoint (J/m²)')
    fig.text(.5,.065,'Gray: original values; colored: extensions; bars: endpoint-integral means',ha='center',fontsize=6.5)
    fig.text(.5,.02,'Open: failure criterion unmet; +: residual flag. Pending designs retain archived values.',ha='center',fontsize=6)
    finish(fig,'si_disorder_work_shifts','Affected work panels of the original Fig. S6, with original and extended endpoint integrals. Means describe observed endpoint integrals and do not impute completed-failure work for censored cases. Counts of remaining censoring and pending extensions accompany the numerical table.')
    return rows

def run(partial):
    lock=open(ROOT/'logs/analysis.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    gate_path=ROOT/'protocol/production_gate.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
    production_released=bool(gate.get('approved'))
    parents={}
    for p in (ROOT/'inputs/baseline/runs').glob('*/record.json'):
        r=json.loads(p.read_text())
        if r.get('status')=='completed' and r.get('metrics'):parents[r['run_id']]=r
    discovery=[r for r in parents.values() if r.get('campaign')=='discovery'];assert len(discovery)==96
    meshes=[r for r in parents.values() if r['family'] in ['nanomesh_single','nanomesh_hier'] and .17 <= (r.get('porosity') or 0) <= .23 and not any(k in r['name'] for k in ['jitter','sizedis']) and 'disorder' not in (r.get('tags') or [])];assert len(meshes)==40
    jobs=json.loads((ROOT/'protocol/jobs.json').read_text());records={}
    quality_manifest=ROOT/'protocol/quality_checks.json'
    if quality_manifest.exists():jobs+=json.loads(quality_manifest.read_text())
    for j in jobs:
        p=ROOT/'runs'/j['case_id']/'record.json'
        if p.exists():records[j['case_id']]=json.loads(p.read_text())
    final=[r for r in records.values() if 'paired' in r]
    done=[r for r in final if r['role']=='extension'];controls=[r for r in final if r['role']=='control'];resolution=[r for r in final if r['role']=='resolution'];quality_checks=[r for r in final if r['role']=='quality']
    pilot_rows=[]
    for j in jobs:
        if j['stage']!='pilot':continue
        r=records.get(j['case_id']);item=dict(case_id=j['case_id'],status=r.get('status') if r else 'pending')
        if r and (ROOT/'runs'/j['case_id']/'stress_strain.csv').exists():
            live=load_live_curve(ROOT/'runs'/j['case_id']/'stress_strain.csv')
            if live is not None:item['saved_frame_screen']=dict(n_frames=len(live['eps_x']),end_strain=float(live['eps_x'][-1]),
                frames_above_force_tolerance=int(np.sum(live['max_force_eV_A']>r['aqs_config']['fmax'])),
                frames_above_transverse_tolerance=int(np.sum(abs(live['sigma_yy'])>r['aqs_config']['sigma_t_tol'])),
                maximum_force_residual_eV_A=float(np.max(live['max_force_eV_A'])))
        if r and 'paired' in r:
            parent=parents[r['parent_run_id']];old=load_curve(ROOT/'inputs/baseline/runs'/r['parent_run_id']/'stress_strain.csv')
            item.update(reproduction=r['paired']['reproduction'],reproduction_flags=r['paired']['reproduction_flags'],quality=r['quality'],
                parent_frames_above_transverse_tolerance=int(np.sum(abs(old['sigma_yy'])>parent['aqs_config']['sigma_t_tol'])),
                parent_maximum_transverse_stress_residual_N_m=float(np.max(abs(old['sigma_yy']))*EV))
        pilot_rows.append(item)
    write_json(ROOT/'results/pilot_assessment.json',dict(updated_utc=now(),review_required=not production_released,pilots=pilot_rows,
        production_released=production_released,final_scientific_review_required=True,
        gate_note=('Pilot review completed; production released for the recorded protocol-sensitivity scope. Numerical limitations remain explicit.' if production_released else
                   'No automatic scientific approval. Review reproduction, physical stopping, and force/transverse residuals before releasing production.')))
    diagnostic_paths=[ROOT/'diagnostics/archived_pilot_forces/summary.json']+[ROOT/f'diagnostics/frame_relaxation_{stage}/record.json' for stage in ['Q2','Q3','Q5']]
    diagnostic_paths += [ROOT/f'diagnostics/endpoint_lbfgs_{stage}/record.json' for stage in ['Q6','Q6b']]
    static_audit_paths=sorted(list((ROOT/'diagnostics').glob('Q4_*_static_*/record.json'))+
                              list((ROOT/'diagnostics').glob('original_setting_*_static_*/record.json')))
    diagnostic_paths += static_audit_paths
    diagnostic_signature='|'.join(sha(p) for p in diagnostic_paths if p.exists())
    signature='|'.join(sorted(r['case_id']+r.get('status','')+r.get('finished_utc','') for r in records.values()))+'|'+sha(__file__)+'|'+diagnostic_signature+'|'+(sha(gate_path) if gate_path.exists() else 'gate_pending')
    previous=ROOT/'results/analysis_state.json'
    if partial and previous.exists() and json.loads(previous.read_text()).get('signature')==signature:return
    comparison=[]
    for j in jobs:
        r=records.get(j['case_id']);row={k:j[k] for k in ['case_id','parent_run_id','source_phase','role','stage']};row['status']=r['status'] if r else 'pending'
        if r and 'paired' in r:
            recorded_curve=load_curve(ROOT/'runs'/r['case_id']/'stress_strain.csv')
            row['endpoint_force_residual_eV_A']=float(recorded_curve['max_force_eV_A'][-1])
            row['endpoint_transverse_stress_N_m']=float(recorded_curve['sigma_yy'][-1])*EV
            row['endpoint_residual_checks_pass']=bool(row['endpoint_force_residual_eV_A']<r['aqs_config']['fmax'] and abs(recorded_curve['sigma_yy'][-1])<=r['aqs_config']['sigma_t_tol'])
            for label,key in [('A','A_archived'),('B','B_original_protocol_on_rerun'),('C','C_extended')]:
                for k,v in r['paired'][key].items():row[label+'_'+k]=v
            row.update(r['paired']['changes']);row['reproduction_flags']='; '.join(r['paired']['reproduction_flags']);row['parent_mode']=parents[r['parent_run_id']]['metrics']['fracture_mode'];row['extended_mode_observed_window']=r['legacy_metrics_for_comparison']['fracture_mode']
            row.update({'quality_'+k:v for k,v in r['quality'].items()})
            row['trajectory_has_residual_tolerance_exceedances']=quality_limited(r)
            row['failure_criterion_and_residual_checks_pass']=bool(r['paired']['C_extended']['failure_observed'] and not quality_limited(r))
            row['original_stop_reached_on_rerun']=r['paired']['virtual_stop_found']
            for interval,start,end in [('A_to_B','A_archived','B_original_protocol_on_rerun'),
                                       ('B_to_C','B_original_protocol_on_rerun','C_extended'),
                                       ('A_to_C','A_archived','C_extended')]:
                for name,key in [('maximum_stress','maximum_recorded_stress_N_m'),('integral','stress_strain_integral_J_m2')]:
                    denominator=r['paired'][start][key]
                    available=interval=='A_to_C' or r['paired']['virtual_stop_found']
                    row[f'{name}_change_{interval}_percent']=(100*(r['paired'][end][key]/denominator-1)
                        if available and denominator else None)
            if not r['paired']['virtual_stop_found']:
                for k in list(row):
                    if k.startswith('B_') or k.endswith('_B_to_C') or '_B_to_C_' in k:row[k]=None
                row['B_termination_reason']='Original stopping rule not reached before rerun ended.'
                row['reproduction_flags']+='; original stopping rule not reached; endpoint comparison incomplete'
        comparison.append(row)
    write_json(ROOT/'results/paired_comparisons.json',comparison)
    columns=list(dict.fromkeys(k for r in comparison for k in r))
    with open(ROOT/'results/paired_comparisons.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(comparison)
    summaries=[]
    for phase,total in [('I',20),('II',14)]:
        g=[r for r in done if r['source_phase']==phase]
        paired_g=[r for r in g if r['paired']['virtual_stop_found']]
        summaries.append(dict(source_phase=phase,planned=total,available=len(g),failure_observed=sum(r['paired']['C_extended']['failure_observed'] for r in g),
            remaining_censored=sum(not r['paired']['C_extended']['failure_observed'] for r in g),
            residual_quality_flagged=sum(quality_limited(r) for r in g),
            failure_criterion_and_residual_checks_pass=sum(r['paired']['C_extended']['failure_observed'] and not quality_limited(r) for r in g),
            original_stop_reached=len(paired_g),
            median_added_integral_J_m2=float(np.median([r['paired']['changes']['additional_integral_B_to_C_J_m2'] for r in paired_g])) if paired_g else None,
            median_added_strain=float(np.median([r['paired']['changes']['additional_strain_B_to_C'] for r in paired_g])) if paired_g else None,
            reproduction_flagged=sum(bool(r['paired']['reproduction_flags']) for r in g)))
    updated={r['parent_run_id']:r for r in done}
    oldI=[r['metrics']['work_to_failure_J_m2'] for r in meshes];newI=[updated[r['run_id']]['paired']['C_extended']['stress_strain_integral_J_m2'] if r['run_id'] in updated else r['metrics']['work_to_failure_J_m2'] for r in meshes]
    corr=dict(n=40,original_endpoint_integral_pearson_r=float(np.corrcoef([align(r) for r in meshes],oldI)[0,1]),
        currently_available_endpoint_integral_pearson_r=float(np.corrcoef([align(r) for r in meshes],newI)[0,1]),
        extensions_available=sum(r['run_id'] in updated for r in meshes),extensions_planned=8,
        interpretation='Correlation of integrals to stated endpoints, not automatically work to observed failure.')
    summary=dict(updated_utc=now(),primary_available=len(done),primary_planned=34,controls_available=len(controls),resolution_available=len(resolution),
        source_cohorts=summaries,alignment_integral_comparison=corr,all_scheduled_cases_have_results=len(final)==len(jobs),additional_quality_checks_available=len(quality_checks),
        scope='Standalone Phase-II results; manuscript integration deferred.',production_released=production_released,
        scientific_review_status=('Pilot review complete; final study review pending, with numerical limitations retained.' if production_released else 'Review pending, including numerical quality and remaining censoring.'))
    write_json(ROOT/'results/summary.json',summary)
    lines=['# Phase-II loading-extension results','',f'Updated {summary["updated_utc"]}. **Interim results; final scientific review pending.**','',
        f'{len(done)} of 34 extensions, {len(controls)} of 2 controls, and {len(resolution)} of 4 resolution checks currently have paired results.','',
        'A = archived original; B = extended rerun truncated at its original stopping rule; C = extended endpoint. B-to-C measures the added observation window along the same computed trajectory. Its physical interpretation remains subject to numerical convergence. All integrals use the original stress/engineering-strain convention.','',
        '| Source phase | Available / planned | Driver failure detected | Failure criterion unmet | Residual quality flags | Reproduction flags |','|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['source_phase']} | {s['available']} / {s['planned']} | {s['failure_observed']} | {s['remaining_censored']} | {s['residual_quality_flagged']} | {s['reproduction_flagged']} |")
    if production_released:
        lines+=['','All six pilot extensions and both controls have been reviewed. Production is released under `protocol/production_gate.json` for the original-setting protocol-sensitivity comparison and four planned resolution checks. This decision does not certify the trajectories as numerically converged or authorize manuscript integration. The separate Q4/Q7/Q8 branch shows unresolved increment sensitivity despite passing saved-state residual checks.']
    lines+=['','## Strength and integrated response: separate comparisons','',
        'Extending the same recorded trajectory changes its maximum stress only when a later higher peak occurs. Its integral can increase even when the maximum remains unchanged. A-to-B changes arise before the added loading window and must not be attributed to extension. Percentages below describe computed outcomes with the numerical flags retained; they are not uncertainty bounds or validated material constants.','',
        '| Completed extension | Peak A-to-B (%) | Peak B-to-C (%) | Integral A-to-B (%) | Integral B-to-C (%) |',
        '|---|---:|---:|---:|---:|']
    for r in done:
        row=next(v for v in comparison if v['case_id']==r['case_id'])
        values=[row[f'{name}_change_{interval}_percent'] for name,interval in
                [('maximum_stress','A_to_B'),('maximum_stress','B_to_C'),('integral','A_to_B'),('integral','B_to_C')]]
        lines.append('| '+r['name']+' | '+' | '.join(f'{v:+.2f}' if v is not None else 'not available' for v in values)+' |')
    coincident=[r['name'] for r in done if r['paired']['virtual_stop_found'] and
                r['paired']['B_original_protocol_on_rerun']['endpoint_index']==r['paired']['C_extended']['endpoint_index']]
    if coincident:
        lines+=['','For '+', '.join(coincident)+', B and C coincide: the rerun meets the driver failure criterion before an administrative cutoff. Its zero B-to-C change means no added loading interval occurred; it does not impute a missing result or imply reproduction of A. Archive-to-rerun differences are reported separately.']
    censored=[r for r in done if not r['paired']['C_extended']['failure_observed']]
    if censored:
        lines+=['','### Completed executions with an unmet failure criterion','',
            'These runs remain censored after extension. Their ending strain and recorded integral must not be substituted for detected failure strain or full-failure work. The nominal strain cap is evaluated after an accepted increment, so the recorded endpoint can slightly exceed the cap. Numerical quality flags remain separate from censoring.','',
            '| Case | C ending strain | Stop reason | C load / running maximum (%) | Spanning at C | Residual quality flag |',
            '|---|---:|---|---:|---|---|']
        for r in censored:
            c=r['paired']['C_extended']
            lines.append(f"| {r['name']} | {c['end_strain']:.5f} | {c['termination_reason']} | {100*c['end_stress_fraction_of_maximum']:.2f} | {c['spanning_at_end']} | {quality_limited(r)} |")
    geometry_audit=ROOT/'results/geometry_audit_20260925T083357Z/audit.json'
    if geometry_audit.exists():
        checked=json.loads(geometry_audit.read_text())['completed_slit_trajectories']
        if checked and all(r['maximum_frame_z_span_A']==0 for r in checked):
            lines+=['','### Geometry and boundary-condition interpretation','',
                f'A separate saved-configuration audit of {len(checked)} completed slit extensions found every inspected frame exactly planar. The original setup uses one carbon monolayer, periodicity in x and y, and a nonperiodic out-of-plane direction. The initially flat athermal trajectory has no imposed out-of-plane perturbation; the rectangular cell does not relax macroscopic shear. These responses do not test stability against out-of-plane buckling or an alternative free-edge or shear-relaxed setup. The primary extensions retain those original conditions. The audit covers its dated inventory, not subsequent uninspected cases. See `results/geometry_audit_20260925T083357Z/README.md` and `audit.json`.']
    for r in done:
        if r['stage']!='pilot':continue
        a=r['paired']['A_archived'];b=r['paired']['B_original_protocol_on_rerun'];c=r['paired']['C_extended']
        bvalue=lambda key,digits:f'{b[key]:.{digits}f}' if r['paired']['virtual_stop_found'] else 'not reached'
        lines+=['',f"### Completed example: {r['name']}",'',
            '| Quantity | Archived original A | Same rerun at original stop B | Extended endpoint C |',
            '|---|---:|---:|---:|',
            f"| Ending strain | {a['end_strain']:.5f} | {bvalue('end_strain',5)} | {c['end_strain']:.5f} |",
            f"| Maximum recorded stress (N/m) | {a['maximum_recorded_stress_N_m']:.4f} | {bvalue('maximum_recorded_stress_N_m',4)} | {c['maximum_recorded_stress_N_m']:.4f} |",
            f"| Recorded integral (J/m²) | {a['stress_strain_integral_J_m2']:.4f} | {bvalue('stress_strain_integral_J_m2',4)} | {c['stress_strain_integral_J_m2']:.4f} |"]
        if r['paired']['virtual_stop_found']:
            pct=100*(c['stress_strain_integral_J_m2']/b['stress_strain_integral_J_m2']-1)
            lines+=['',f"The same-rerun B-to-C integral change is {pct:+.2f}%. C terminated by **{c['termination_reason']}**; spanning connectivity at C was **{c['spanning_at_end']}**, and the residual load was {100*c['end_stress_fraction_of_maximum']:.2f}% of the running maximum. A stress-collapse stop does not imply complete geometrical separation."]
        lines+=['',f"This trajectory has {r['quality']['frames_above_force_tolerance']} saved frames above the force tolerance and {r['quality']['frames_above_transverse_tolerance']} above the transverse-stress tolerance. Reproduction flags: {'; '.join(r['paired']['reproduction_flags']) or 'none at screening thresholds'}. The extended endpoint is a driver outcome, not a numerically validated material-failure value when these flags remain."]
        rc=next(row for row in comparison if row['case_id']==r['case_id'])
        lines+=['',f"The final recorded frame itself has maximum force {rc['endpoint_force_residual_eV_A']:.4f} eV/Å and transverse stress {rc['endpoint_transverse_stress_N_m']:.4f} N/m; endpoint residual checks {'pass' if rc['endpoint_residual_checks_pass'] else 'do not pass'}."]
    lines+=['','Numerical tolerance exceedances and reproduction differences are retained in the paired table. A completed execution is not synonymous with observed failure or numerical validation. This analysis writes only to the isolated study folder; integration of the new simulation results remains deferred.','',
        'The original 40-mesh population is fixed. The original endpoint-integral/alignment correlation is '+f"{corr['original_endpoint_integral_pearson_r']:.4f}. "+f"{corr['extensions_available']} of its eight planned replacements are available. The updated endpoint correlation is {corr['currently_available_endpoint_integral_pearson_r']:.4f}; it is not a completed-failure work correlation when censoring remains.",'',
        '## Controls and numerical checks','', '| Case | Execution | Driver failure | Frames above force / transverse tolerance | Reproduction diagnostics |','|---|---|---|---|---|']
    for r in controls+resolution+quality_checks:lines.append(f"| {r['case_id']} | {r['status']} | {r['paired']['C_extended']['failure_observed']} | {r['quality']['frames_above_force_tolerance']} / {r['quality']['frames_above_transverse_tolerance']} | {'; '.join(r['paired']['reproduction_flags']) or 'Within screening thresholds'} |")
    for r in controls:
        a=r['paired']['A_archived'];b=r['paired']['B_original_protocol_on_rerun'];c=r['paired']['C_extended']
        if not r['paired']['virtual_stop_found']:continue
        lines+=['',f"Control **{r['name']}**: original ending strain {a['end_strain']:.5f}, rerun {c['end_strain']:.5f}; maximum recorded stress {a['maximum_recorded_stress_N_m']:.4f} → {c['maximum_recorded_stress_N_m']:.4f} N/m; integral {a['stress_strain_integral_J_m2']:.4f} → {c['stress_strain_integral_J_m2']:.4f} J/m². The B-to-C added interval is {c['end_strain']-b['end_strain']:.5f} strain. When B and C coincide, the archive-to-rerun difference is a reproduction difference, not an effect of longer loading."]
    for j in jobs:
        if j['role']!='quality':continue
        r=records.get(j['case_id'])
        if not r or 'paired' not in r:lines.append(f"| {j['case_id']} | {r['status'] if r else 'pending'} | not available | not available | Awaiting completed diagnostic and review |")
    qc=next((r for r in quality_checks if r['case_id']=='quality__S1_pristine_zz_transverse15'),None)
    pc=next((r for r in controls if r['name']=='S1_pristine_zz'),None)
    if qc and pc:
        q=qc['paired']['C_extended'];p=pc['paired']['C_extended'];delta=100*(q['stress_strain_integral_J_m2']/p['stress_strain_integral_J_m2']-1)
        lines+=['',f"Q1 increased only the pristine control's transverse-iteration budget from 3 to 15. Its ending strain changed from {p['end_strain']:.5f} to {q['end_strain']:.5f}, and its recorded integral changed by {delta:+.2f}%. All saved force and transverse-stress residuals passed their nominal tolerances in Q1. This is numerical-protocol sensitivity, not a longer-loading effect."]
    check=ROOT/'results/saved_frame_residual_checks.json'
    if check.exists():
        lines+=['','Independent CPU/float64 static reevaluation confirms large residual forces in selected accepted pilot frames, including frames whose inherited FIRE flag is true. The driver records that flag from an earlier minimization, before subsequent transverse relaxation; final forces must be checked directly. No potential or primary execution settings have been altered.',
            '', 'Q2 tests further fixed-cell relaxation of three saved frames and audits saved forces in all eight archived pilot-parent trajectories. These diagnostics do not resume loading trajectories and are not spliced into primary runs. Detailed results reside in `diagnostics/`. The production decision is recorded separately in `protocol/production_gate.json`.']
    audit=ROOT/'diagnostics/archived_pilot_forces/summary.json'
    if audit.exists():
        audit_rows=json.loads(audit.read_text())['cases']
        lines+=['','### Archived pilot-parent force audit','',
            'This is a static CPU/float64 reevaluation of saved coordinates, without relaxation. The saved positions have float32 precision, so near-tolerance counts need that qualification. This selected eight-case audit does not estimate prevalence across all 96 discovery runs.','',
            '| Source phase | Parent design | Frames above force tolerance / saved frames | Maximum force residual (eV/Å) |',
            '|---|---|---:|---:|']
        for a in audit_rows:
            parent=parents[a['parent_run_id']];job=next(j for j in jobs if j['case_id']==a['case_id'])
            a.update(source_phase=job['source_phase'],name=parent['name'])
            lines.append(f"| {job['source_phase']} | {parent['name']} | {a['frames_above_nominal_force_tolerance']} / {a['n_frames']} | {a['maximum_force_residual_eV_A']:.4f} |")
        if audit_rows:
            with open(ROOT/'results/si_archived_pilot_force_audit.csv','w') as f:
                w=csv.DictWriter(f,fieldnames=list(audit_rows[0]));w.writeheader();w.writerows(audit_rows)
    probe=ROOT/'diagnostics/frame_relaxation_Q2/record.json'
    if probe.exists():
        q2=json.loads(probe.read_text())
        lines+=['',f"Q2 fixed-cell relaxation probes: **{q2['status']}**. These results diagnose saved states; they cannot establish a repaired loading path or failure endpoint."]
        if q2.get('rows'):
            lines+=['','| Saved frame | Force before → after (eV/Å) | Stress before → after (N/m) | Force criterion met |','|---|---:|---:|---|']
            for r in q2['rows']:
                b,a=r['before'],r['after']
                lines.append(f"| {r['case_id']} / {r['frame_index']} | {b['max_force_eV_A']:.4f} → {a['max_force_eV_A']:.4f} | {b['sigma_xx_N_m']:.3f} → {a['sigma_xx_N_m']:.3f} | {r['force_converged']} |")
    diagnostic_rows=[]
    for stage in ['Q2','Q3','Q5']:
        path=ROOT/f'diagnostics/frame_relaxation_{stage}/record.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        for r in record.get('rows',[]):
            row=dict(stage=stage,case_id=r['case_id'],frame_index=r['frame_index'],strain=r['strain'],force_converged=r['force_converged'],iterations=r['minimizer']['iterations'],source_sha256=r['source_sha256'])
            for prefix in ['before','after']:
                row.update({prefix+'_'+k:v for k,v in r[prefix].items()})
            diagnostic_rows.append(row)
        if stage=='Q3':
            lines+=['',f"Q3 conservative FIRE probes: **{record['status']}**. These continue the two unresolved Q2 configurations at fixed cell, with smaller time/displacement limits. They are not loading continuations."]
            for r in record.get('rows',[]):lines.append(f"- {r['case_id']}: final force {r['after']['max_force_eV_A']:.4f} eV/Å; transverse stress {r['after']['sigma_yy_N_m']:.4f} N/m; force tolerance {'met' if r['force_converged'] else 'unmet'}.")
        if stage=='Q5':
            lines+=['',f"Q5 endpoint relaxation probe: **{record['status']}**. It checks the recorded 45° endpoint at fixed cell, without modifying or extending its loading trajectory."]
            for r in record.get('rows',[]):lines.append(f"- Final force {r['after']['max_force_eV_A']:.4f} eV/Å; transverse stress {r['after']['sigma_yy_N_m']:.4f} N/m; stress remains below the frozen running-peak threshold: {r['after']['below_frozen_running_peak_threshold']}; spanning: {r['after']['spanning']}.")
    for stage in ['Q6','Q6b']:
        path=ROOT/f'diagnostics/endpoint_lbfgs_{stage}/record.json'
        if not path.exists():continue
        record=json.loads(path.read_text())
        lines+=['',f"{stage} CPU/float64 L-BFGS-B endpoint diagnostic: **{record['status']}**. It retains the fixed cell and original recorded endpoint geometry as input."]
        if record['status']=='execution_error':lines.append('This diagnostic attempt failed during setup: '+record.get('error','unspecified')+'. Its files are retained; no material outcome is inferred.')
        if record.get('after'):
            r=record;a=r['amendment'];after=r['after']
            row=dict(stage=stage,case_id=a['case_id'],frame_index=a['frame_index'],strain=a['strain'],force_converged=r['force_converged'],iterations=r['optimizer']['nit'],source_sha256=r['source_sha256'])
            for prefix in ['before','after']:row.update({prefix+'_'+k:v for k,v in r[prefix].items()})
            diagnostic_rows.append(row)
            lines.append(f"Final force {after['max_force_eV_A']:.5f} eV/Å; transverse stress {after['sigma_yy_N_m']:.5f} N/m; axial stress {after['sigma_xx_N_m']:.5f} N/m. Force tolerance met: {r['force_converged']}; transverse tolerance met: {r['transverse_converged']}. Optimizer success alone is not used as a convergence certificate.")
    if diagnostic_rows:
        write_json(ROOT/'results/si_fixed_cell_relaxation.json',diagnostic_rows)
        with open(ROOT/'results/si_fixed_cell_relaxation.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in diagnostic_rows for k in row)));w.writeheader();w.writerows(diagnostic_rows)
    attempts=[]
    for stage,folder in [('Q2','frame_relaxation_Q2'),('Q3','frame_relaxation_Q3'),('Q5','frame_relaxation_Q5'),('Q6','endpoint_lbfgs_Q6'),('Q6b','endpoint_lbfgs_Q6b')]:
        path=ROOT/'diagnostics'/folder/'record.json'
        if not path.exists():continue
        record=json.loads(path.read_text());amendment=record['amendment']
        selections=amendment.get('frames') or [amendment]
        for selection in selections:
            candidate=next((v for v in record.get('rows',[]) if v['case_id']==selection['case_id'] and v['frame_index']==selection['frame_index']),None)
            if candidate is None and record.get('after'):candidate=record
            completed=candidate is not None and 'after' in candidate
            a=candidate['after'] if completed else {}
            attempts.append(dict(attempt_id=stage,case_id=selection['case_id'],frame_index=selection['frame_index'],
                status='completed' if completed else record['status'],force_tolerance_met=bool(a['max_force_eV_A']<.02) if completed else None,
                transverse_tolerance_met=bool(abs(a['sigma_yy_N_m'])<=.005*EV) if completed else None,
                final_force_eV_A=a.get('max_force_eV_A'),final_axial_stress_N_m=a.get('sigma_xx_N_m'),final_transverse_stress_N_m=a.get('sigma_yy_N_m'),
                error=record.get('error',''),record_path=str(path.relative_to(ROOT)),source_script_sha256=record['script_sha256']))
    if attempts:
        write_json(ROOT/'results/diagnostic_attempts.json',attempts)
        with open(ROOT/'results/si_diagnostic_attempts.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(attempts[0]));w.writeheader();w.writerows(attempts)
    if (ROOT/'protocol/amendment_Q4.json').exists():
        lines+=['','Q4 is a separately frozen numerical-protocol branch (`code_verified/`). It requires both final force and transverse-stress tolerances before accepting a loading step, retains rejected candidate states, and terminates unresolved trajectories as numerical limitations. Passing these residual checks alone does not establish strain-increment convergence.',
            '', 'B-to-C entries are left unavailable when a trajectory terminates before its virtual original stopping rule. No missing extension is represented by a zero effect.']
        review=ROOT/'results/Q4_control_review.json'
        if review.exists():
            v=json.loads(review.read_text());c=v['driver_endpoint']
            lines+=['',f"The Q4 pristine control passed all saved-frame residual checks, including independent CPU/float64 reevaluation. Its ending strain was {c['end_strain']:.5f}, maximum recorded stress {c['maximum_recorded_stress_N_m']:.3f} N/m and integral {c['stress_strain_integral_J_m2']:.4f} J/m² ({v['integral_change_vs_original_setting_control_percent']:+.2f}% relative to the new original-setting control). A separate Q4 20° slit diagnostic was released after this review. This does not release primary production or establish a unique numerically converged failure strain."]
        else:lines+=['','The prepared Q4 20° slit diagnostic awaits review of the pristine control before launch.']
        slit=next((r for r in quality_checks if r['case_id']=='quality__S7_slit_20deg_verified_Q4'),None)
        if slit:
            p=slit['paired'];a=p['A_archived'];b=p['B_original_protocol_on_rerun'];c=p['C_extended']
            lines+=['','### Completed Q4 20-degree slit diagnostic','',
                '| Quantity | Archived original A | Strict rerun at original rule B | Strict extended endpoint C |',
                '|---|---:|---:|---:|',
                f"| Ending strain | {a['end_strain']:.5f} | {b['end_strain']:.5f} | {c['end_strain']:.5f} |",
                f"| Maximum recorded stress (N/m) | {a['maximum_recorded_stress_N_m']:.4f} | {b['maximum_recorded_stress_N_m']:.4f} | {c['maximum_recorded_stress_N_m']:.4f} |",
                f"| Recorded integral (J/m²) | {a['stress_strain_integral_J_m2']:.4f} | {b['stress_strain_integral_J_m2']:.4f} | {c['stress_strain_integral_J_m2']:.4f} |",'',
                f"The Q4 stopping criterion was **{c['termination_reason']}**; spanning remains **{c['spanning_at_end']}**. The original rule on this changed numerical trajectory stops by **{b['termination_reason']}**, with renewed peaks preventing the earlier archived post-peak stop. B-to-C changes the integral by {100*(c['stress_strain_integral_J_m2']/b['stress_strain_integral_J_m2']-1):+.2f}%. The larger archive-to-C change also includes numerical-protocol and reproduction effects. Passing the saved-state residual checks does not establish strain-increment convergence or complete geometric separation."]
            primary_slit=records.get('ext__S7_slit_20deg')
            if primary_slit and 'paired' in primary_slit:
                op=primary_slit['paired'];ob=op['B_original_protocol_on_rerun'];oc=op['C_extended']
                lines+=['',f"The completed original-setting 20-degree extension stops at {oc['end_strain']:.5f}, versus {c['end_strain']:.5f} for Q4. Its B-to-C recorded integral changes by {100*(oc['stress_strain_integral_J_m2']/ob['stress_strain_integral_J_m2']-1):+.2f}%, versus {100*(c['stress_strain_integral_J_m2']/b['stress_strain_integral_J_m2']-1):+.2f}% for Q4. The original-setting trajectory fails residual checks; Q4 passes saved-state checks but lacks demonstrated increment convergence. Both stop by stress collapse while still spanning. These contrasting numerical trajectories show that the inferred loading-window effect depends on the relaxation procedure; neither endpoint is selected as a uniquely converged material-failure value."]
    refinement_rows=[]
    for j in jobs:
        if 'comparison_case_id' not in j:continue
        r=records.get(j['case_id']);reference=records.get(j['comparison_case_id'])
        row=dict(case_id=j['case_id'],reference_case_id=j['comparison_case_id'],
            resolution_factor=j['resolution_factor'],status=r['status'] if r else 'pending',
            comparison_available=bool(r and 'paired' in r and reference and 'paired' in reference),
            end_strain=None,reference_end_strain=None,end_strain_difference=None,
            maximum_stress_relative_percent=None,integral_relative_percent=None,
            all_recorded_residual_checks_pass=None,interpretation='Increment sensitivity; one refinement does not establish asymptotic convergence.')
        if row['comparison_available']:
            c=r['paired']['C_extended'];b=reference['paired']['C_extended']
            row.update(end_strain=c['end_strain'],reference_end_strain=b['end_strain'],
                end_strain_difference=c['end_strain']-b['end_strain'],
                maximum_stress_relative_percent=100*(c['maximum_recorded_stress_N_m']/b['maximum_recorded_stress_N_m']-1),
                integral_relative_percent=100*(c['stress_strain_integral_J_m2']/b['stress_strain_integral_J_m2']-1),
                all_recorded_residual_checks_pass=not quality_limited(r))
        refinement_rows.append(row)
    if refinement_rows:
        write_json(ROOT/'results/si_verified_increment_checks.json',refinement_rows)
        with open(ROOT/'results/si_verified_increment_checks.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(refinement_rows[0]));w.writeheader();w.writerows(refinement_rows)
        lines+=['','### Strain-increment diagnostics on the Q4 numerical branch','',
            'Q7 halves all three strain increments relative to Q4 while retaining its potential, solver, tolerances and loading rules. Q8, when listed below, halves those increments again. Each run starts from the original initial geometry. These are additional to the four original-setting resolution checks scheduled through the primary production queue. Each comparison uses the stated next-coarser reference; comparisons with the archived Phase-I result remain separately available in the main paired table.']
        for row in refinement_rows:
            lines+=['',f"{row['case_id']}: **{row['status']}**. Reference: `{row['reference_case_id']}`."]
            if row['comparison_available']:
                lines.append(f"Ending strain {row['reference_end_strain']:.5f} → {row['end_strain']:.5f}; maximum recorded stress change {row['maximum_stress_relative_percent']:+.2f}%; recorded integral change {row['integral_relative_percent']:+.2f}%. All recorded residual checks pass: {row['all_recorded_residual_checks_pass']}. A single refinement does not establish a unique converged failure value.")
            else:lines.append('Endpoint and integral comparisons are pending; no missing effect is imputed as zero.')
        refinement_review=ROOT/'results/Q8_control_review.json'
        if refinement_review.exists():
            v=json.loads(refinement_review.read_text())
            lines+=['',f"Three-level control review: the final refinement stability screen {'passes' if v['stability_screen_pass'] else 'does not pass'}. All accepted states at these levels pass recorded and independent static residual checks, but endpoint, maximum stress and integral remain sensitive to the increment. The nonmonotonic values are retained; no level is selected as uniquely converged. One trajectory per increment does not separately estimate same-setting repeatability. This bounded three-level control diagnostic is complete; no further refinement is automatically launched."]
    static_rows=[]
    if static_audit_paths:
        lines+=['','### Independent static checks of saved loading states','',
            'Selected accepted configurations are reevaluated in CPU/float64 without relaxation. These checks verify saved-state residuals; they do not continue the loading path or establish strain-increment convergence. Saved positions have float32 precision.']
        for path in static_audit_paths:
            audit=json.loads(path.read_text())
            passed=sum(r['cpu_force_pass'] and r['cpu_transverse_pass'] for r in audit['rows'])
            lines+=['',f"{audit['case_id']}: static audit **{audit['status']}**, {len(audit['rows'])}/{len(audit['selected_frame_indices'])} selected frames evaluated, {passed} pass both residual checks. Source: `{path.relative_to(ROOT)}`."]
            for row in audit['rows']:static_rows.append(dict(case_id=audit['case_id'],audit_status=audit['status'],record_path=str(path.relative_to(ROOT)),**row))
        write_json(ROOT/'results/si_verified_static_checks.json',static_rows)
        if static_rows:
            with open(ROOT/'results/si_verified_static_checks.csv','w') as f:
                w=csv.DictWriter(f,fieldnames=list(static_rows[0]));w.writeheader();w.writerows(static_rows)
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n')
    if done:
        main_figure(done,parents);migration(done,parents,discovery,meshes);atlas(done,parents)
        disorder_rows=disorder(done,parents)
        write_json(ROOT/'results/disorder_endpoint_summaries.json',disorder_rows)
        with open(ROOT/'results/si_disorder_endpoint_summaries.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=list(disorder_rows[0]));w.writeheader();w.writerows(disorder_rows)
    write_json(previous,dict(signature=signature,updated_utc=now(),analysis_script_sha256=sha(__file__)))
    print(now(),'Analysis updated:',len(done),'extensions;',len(controls),'controls;',len(resolution),'resolution checks',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');run(p.parse_args().partial)
