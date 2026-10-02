"""Interim numerical-diagnostic preview, using only completed controls/probes."""
import json
from study_common import ROOT,EV,load_curve,write_json,now,sha
from analyze import plt,style
from matplotlib.lines import Line2D
import numpy as np

def main():
    primary=json.loads((ROOT/'runs/ctrl__S1_pristine_zz/record.json').read_text())
    stricter=json.loads((ROOT/'runs/quality__S1_pristine_zz_transverse15/record.json').read_text())
    q2=json.loads((ROOT/'diagnostics/frame_relaxation_Q2/record.json').read_text());assert q2['status']=='completed'
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7))
    fig.subplots_adjust(left=.075,right=.99,bottom=.26,top=.84,wspace=.51)
    for ax,letter in zip(axes,'abc'):style(ax,letter)
    for path,label,color,ls in [(ROOT/'inputs/baseline/runs'/primary['parent_run_id'],'archived original','.6','--'),
                              (ROOT/'runs'/primary['case_id'],'original settings','black','-'),
                              (ROOT/'runs'/stricter['case_id'],'15 transverse iterations','#0072b2','-')]:
        c=load_curve(path/'stress_strain.csv');axes[0].plot(c['eps_x'],c['sigma_xx']*EV,label=label,color=color,ls=ls,lw=.9)
    axes[0].set(xlabel='engineering strain',ylabel='2D stress (N/m)',title='Pristine control: Q1')
    axes[0].legend(frameon=False,fontsize=5.6,loc='upper left')
    axes[0].text(.55,.40,'ending strain\n0.2713 → 0.2538\nintegral: −10.01%',transform=axes[0].transAxes,ha='center',va='top',fontsize=6.5,color='#0072b2')
    labels=['45° slit','20° slit','armchair']
    for k,r in enumerate(q2['rows']):
        for ax,key in [(axes[1],'max_force_eV_A'),(axes[2],'sigma_xx_N_m')]:
            b,a=r['before'][key],r['after'][key]
            ax.plot([k-.12,k+.12],[b,a],color='.6',lw=.8)
            ax.scatter(k-.12,b,s=22,color='.55',zorder=4)
            ax.scatter(k+.12,a,s=25,edgecolors='#0072b2',facecolors='#0072b2' if r['force_converged'] else 'white',zorder=4)
        axes[1].text(k,.008,'pass' if r['force_converged'] else 'unresolved',fontsize=5.6,ha='center',color='.3')
    for ax in axes[1:]:ax.set_xticks(range(3),labels);ax.tick_params(axis='x',labelsize=6);ax.set_xlim(-.45,2.4)
    axes[1].set(yscale='log',ylabel='maximum force (eV/Å)',ylim=(.005,8),title='Fixed-cell relaxation: Q2')
    axes[1].axhline(.02,color='.3',ls=':',lw=.8);axes[1].text(2.32,.022,'tol.',fontsize=5.5,ha='right')
    axes[2].set(ylabel='2D stress (N/m)',title='Stress change at fixed strain',ylim=(0,14))
    handles=[Line2D([],[],marker='o',ls='',color='.55',label='saved frame'),
             Line2D([],[],marker='o',ls='',color='#0072b2',label='after further relaxation'),
             Line2D([],[],marker='o',ls='',color='#0072b2',markerfacecolor='white',label='force tolerance unmet')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.045),ncol=3,frameon=False,fontsize=6)
    fig.text(.52,.015,'Interim diagnostics. Q2 does not continue or repair the loading trajectories.',ha='center',fontsize=6,color='.35')
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'numerical_diagnostics_preview.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    (ROOT/'figures/numerical_diagnostics_preview.caption.txt').write_text('Q1 changes only the pristine control transverse-iteration budget from 3 to 15; the integral change is relative to the new original-setting control. Q2 relaxes selected saved configurations at fixed cell using unchanged FIRE and MPS/float32, with 8000 steps maximum and no soft stall acceptance. Open blue markers mean the force tolerance remains unmet. These partial diagnostics must not be interpreted as repaired loading paths or extended failure measurements.\n')
    write_json(ROOT/'results/numerical_preview_provenance.json',dict(created_utc=now(),script_sha256=sha(__file__),source_files=['runs/ctrl__S1_pristine_zz/record.json','runs/quality__S1_pristine_zz_transverse15/record.json','diagnostics/frame_relaxation_Q2/record.json']))

if __name__=='__main__':main()
