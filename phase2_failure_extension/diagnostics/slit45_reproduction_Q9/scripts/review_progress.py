"""Interim trace audit and compact diagnostics; never alters simulation state."""
from pathlib import Path
from datetime import datetime, timezone
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare import ROOT, STUDY, PARENT, regular_path, replay_path

def complete_json_lines(path):
    # A live writer can leave an incomplete final line; only complete records count.
    text = path.read_text()
    return [json.loads(s) for s in text.splitlines(keepends=True) if s.endswith('\n')]

def main():
    jobs = json.loads((ROOT/'jobs.json').read_text())
    original = regular_path('Phase 1 single shot', ROOT/'inputs'/PARENT, original=True)
    primary = regular_path('Phase 2 primary', STUDY/'runs/ext__S2_slit_45deg')
    paths = [original, primary]
    labels = ['Phase 1 single shot', 'Phase 2 primary']
    colors = ['#313b45', '#287ba5', '#df810f', '#7d63a8', '#58a495']
    for job, label in zip(jobs, ['Original-batch replay', 'Single replay', 'Single repeat']):
        path = replay_path(job)
        if path is not None:
            paths.append(path)
            suffix = {'running':' (in progress)', 'suspended':' (suspended)', 'completed':''}.get(path['status'], ' (incomplete)')
            labels.append(label + suffix)
    callpath = ROOT/'runs/R1_original_batch4/minimizer_calls.jsonl'
    calls = complete_json_lines(callpath) if callpath.exists() else []
    trial = None
    extra = []
    for call in calls:
        if call['callsite_line']==285:
            trial = call
        elif call['callsite_line']==303 and trial and call['active'][2] and trial['converged'][2]:
            extra.append(call)
    original_log = (ROOT/'inputs/original_batch.log').read_text().splitlines()
    replay_log = (ROOT/'logs/R1_original_batch4.log').read_text().splitlines()
    relevant = lambda lines: [x for x in lines if any(f'step {v:3d} ' in x or f'step {v}:' in x for v in range(16,25))]
    audit = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                 original_transition_log=relevant(original_log), replay_transition_log=relevant(replay_log),
                 extra_full_minimizations_after_target_passed_trial=extra,
                 observed='Original attempt at strain 0.17 was rejected after the post-trial processing, then 0.165 was accepted before returning to 0.17. R1 accepted 0.17 directly. This is a difference in the adaptive loading history, not proof that the intermediate step alone causes the later stress difference.',
                 qualification='The number of active, already-converged target retries is explicitly reported; zero target motion in such a call only rules out extra coordinate relaxation during that call. Neighbor refreshes, roundoff, transverse relaxation and path selection remain under review.',
                 accepted_strain_sequences={p['name']:[float(x) for x in p['strain'] if .16-1e-10<=x<=.18+1e-10] for p in paths},
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'results/increment_history_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.linewidth':.6,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axs = plt.subplots(1,3,figsize=(10,3.25))
    plotted=[]
    for p,label,color in zip(paths,labels,colors):
        window=(p['strain']>=.15-1e-10)&(p['strain']<=.18+1e-10)
        axs[0].plot(p['strain'][window],p['stress'][window],'.-',lw=1,ms=4,c=color,label=label)
        indices=np.flatnonzero((p['strain']>=.16-1e-10)&(p['strain']<=.18+1e-10))
        axs[1].plot(np.arange(len(indices)),p['strain'][indices],'.-',lw=1,ms=5,c=color)
        axs[2].plot(p['strain'][window],p['force'][window],'.-',lw=1,ms=4,c=color)
        for i in np.flatnonzero(window):
            plotted.append(dict(path=p['name'],status=p['status'],frame=int(i),strain=float(p['strain'][i]),
                                stress_N_m=float(p['stress'][i]),max_force_eV_A=float(p['force'][i]),
                                accepted_index_from_016=int(np.flatnonzero(indices==i)[0]) if i in indices else ''))
    axs[0].set(xlabel='engineering strain',ylabel='2D stress (N/m)',xlim=(.149,.181))
    axs[1].set(xlabel='accepted state after strain 0.16',ylabel='engineering strain')
    axs[2].set(xlabel='engineering strain',ylabel='maximum force (eV/Å)',yscale='log',xlim=(.149,.181))
    axs[2].axhline(.02,color='#555555',ls=':',lw=.9)
    axs[2].text(.1798,.023,'target',ha='right',fontsize=7,color='#555555')
    for i,ax in enumerate(axs):
        ax.text(-.17,1.03,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.065,right=.98,bottom=.20,top=.79,wspace=.40)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(out/f'transition_diagnostic.{ext}',dpi=180,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
    with (ROOT/'results/transition_diagnostic_data.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    (out/'transition_diagnostic_caption.md').write_text('''Interim reproduction diagnostic, generated from saved states. (a) Recorded stress near the first damage event. (b) Accepted strain sequence starting at 0.16; rejected trials are not plotted, but appear in the accompanying log audit. The original path includes an accepted 0.165 state; the current batch replay reaches 0.17 directly. (c) Actual force residuals; the dotted line is the original 0.02 eV/Angstrom target. Original residuals are independently evaluated on saved coordinates in CPU float64; the newer paths use recorded MPS residuals. Lines join samples only; no configurations or bond events are interpolated. An in-progress path stops at its latest accepted sample; it is not a failure endpoint. The visual records a loading-history difference and does not establish its unique cause or converged material behavior. The plot is a separate Q9 diagnostic and is not included in the paper.\n''')
    print(json.dumps({'paths':labels,'converged_target_extra_calls':len(extra),
                      'maximum_target_motion_in_those_calls_A':max([x['target_max_component_motion_A'] for x in extra],default=None)},indent=2))

if __name__=='__main__':
    main()
