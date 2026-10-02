"""Compact control-resolution diagnostic, retaining pending outcomes explicitly."""
import json
import numpy as np
from matplotlib.lines import Line2D
from analyze import plt, style
from study_common import ROOT, EV, load_curve, now, sha, write_json


def main():
    selections=[('quality__S1_pristine_zz_verified_Q4','1 (Q4)','#0072b2'),
                ('quality__S1_pristine_zz_halfstep_Q7','1/2 (Q7)','#009e73'),
                ('quality__S1_pristine_zz_quarterstep_Q8','1/4 (Q8)','#9467bd')]
    first=json.loads((ROOT/'runs'/selections[0][0]/'record.json').read_text())
    base=ROOT/'inputs/baseline/runs'/first['parent_run_id']
    old=load_curve(base/'stress_strain.csv');parent=json.loads((base/'record.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.2))
    fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.78,wspace=.48)
    for ax,letter in zip(axes,'abc'):style(ax,letter)
    fig.suptitle('Pristine control: strain-increment sensitivity',x=.075,y=.99,ha='left',fontsize=10,fontweight='bold')
    fig.text(.075,.915,'Colored curves share the Q4 equilibrium procedure; gray is the archived Phase-I reference.',fontsize=6.7,color='.35')
    axes[0].plot(old['eps_x'],old['sigma_xx']*EV,'--',color='.55',lw=1)
    axes[0].set(xlabel='Engineering strain',ylabel='2D stress (N/m)',xlim=(-.005,.285),ylim=(-1,43))
    axes[1].axhline(parent['metrics']['work_to_failure_J_m2'],color='.55',ls='--',lw=.8)
    axes[2].axhline(parent['metrics']['failure_strain'],color='.55',ls='--',lw=.8)
    axes[1].set(ylabel='Recorded integral (J/m²)',ylim=(5.3,7.0))
    axes[2].set(ylabel='Driver stopping strain',ylim=(.23,.285))
    rows=[];sources={str((base/'record.json').relative_to(ROOT)):sha(base/'record.json'),str((base/'stress_strain.csv').relative_to(ROOT)):sha(base/'stress_strain.csv')}
    plot_strain=[float(old['eps_x'][-1])];plot_stress=[float(old['sigma_xx'].max()*EV)]
    for k,(cid,label,color) in enumerate(selections):
        path=ROOT/'runs'/cid/'record.json'
        r=json.loads(path.read_text()) if path.exists() else {}
        row=dict(case_id=cid,status=r.get('status','pending'),recorded_integral_J_m2=None,driver_stopping_strain=None)
        if 'paired' in r:
            c=load_curve(path.parent/'stress_strain.csv');e=r['paired']['C_extended'];q=r['quality']
            good=not (q['frames_above_force_tolerance'] or q['frames_above_transverse_tolerance'] or q['numerical_termination'])
            integral=float(np.trapezoid(c['sigma_xx']*EV,c['eps_x']))
            assert np.isclose(integral,e['stress_strain_integral_J_m2'])
            axes[0].plot(c['eps_x'],c['sigma_xx']*EV,color=color,lw=1)
            plot_strain.append(float(c['eps_x'][-1]));plot_stress.append(float(c['sigma_xx'].max()*EV))
            axes[0].scatter(c['eps_x'][-1],c['sigma_xx'][-1]*EV,color=color,marker='X' if e['failure_observed'] else '>',s=20,zorder=4)
            for ax,value in [(axes[1],integral),(axes[2],e['end_strain'])]:
                ax.scatter(k,value,s=27,edgecolors=color,facecolors=color if e['failure_observed'] else 'white',zorder=4)
                if not good:ax.scatter(k,value,marker='+',color='black',s=18,zorder=5)
                ax.annotate(f'{value:.3f}',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=6.5,color=color)
            row.update(recorded_integral_J_m2=integral,driver_stopping_strain=e['end_strain'],all_recorded_residual_checks_pass=good)
            sources[str(path.relative_to(ROOT))]=sha(path);sources[str((path.parent/'stress_strain.csv').relative_to(ROOT))]=sha(path.parent/'stress_strain.csv')
        else:
            for ax in axes[1:]:ax.text(k,.09,'pending',transform=ax.get_xaxis_transform(),ha='center',fontsize=6.5,color='.4')
        rows.append(row)
    integral_values=[parent['metrics']['work_to_failure_J_m2']]+[r['recorded_integral_J_m2'] for r in rows if r['recorded_integral_J_m2'] is not None]
    strain_values=[parent['metrics']['failure_strain']]+[r['driver_stopping_strain'] for r in rows if r['driver_stopping_strain'] is not None]
    axes[0].set_xlim(-.005,max(plot_strain)*1.06);axes[0].set_ylim(-1,max(plot_stress)*1.10)
    axes[1].set_ylim(min(integral_values)-.35,max(integral_values)+.35)
    axes[2].set_ylim(min(strain_values)-.015,max(strain_values)+.015)
    for ax in axes[1:]:
        ax.set_xticks(range(3),[s[1] for s in selections]);ax.set_xlim(-.45,2.45);ax.set_xlabel('Strain-increment multiplier')
    handles=[Line2D([],[],color='.55',ls='--',label='Archived original')]+[Line2D([],[],color=c,label='Steps × '+label) for _,label,c in selections]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.075),ncol=4,frameon=False,fontsize=6.2)
    fig.text(.075,.025,'Same potential; all three strain increments scale together. Passing residual checks alone does not establish step convergence.',fontsize=6.2,color='.35')
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'pristine_increment_diagnostic.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    (ROOT/'figures/pristine_increment_diagnostic.caption.txt').write_text('Pristine control diagnostics Q4, Q7 and Q8 use the same residual-enforced numerical procedure and potential, scaling d_strain, d_strain_elastic and d_strain_min together by 1, 1/2 and 1/4 relative to Q4. Each starts from the original initial geometry. Gray dashed curves and horizontal lines denote the archived Phase-I reference, whose numerical settings differ from Q4. Only completed trajectories enter the quantitative panels; pending observations are not imputed. X denotes the driver stopping event. Integrals use the recorded stress and engineering strain convention. These deterministic paired diagnostics do not provide statistical error bars or establish a unique converged value.\n')
    write_json(ROOT/'results/pristine_increment_figure_provenance.json',dict(created_utc=now(),script_sha256=sha(__file__),source_sha256=sources,rows=rows))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
