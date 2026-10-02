"""Index old/new point identities and changes by their original paper panel."""
from pathlib import Path
import json,csv
import numpy as np
OUT=Path(__file__).resolve().parents[1];DATA=OUT/'data';D=json.loads((DATA/'designs.json').read_text());BY={r['name']:r for r in D};N=json.loads((DATA/'source_figure_numbers.json').read_text())
disc=[r for r in D if r['campaign']=='discovery'];por=[r for r in disc if r['porosity']>.03]
mesh=[r for r in D if r['parent_run_id'] in json.loads((OUT.parents[1]/'candidate_figures/data/alignment_population.json').read_text())]
panels=[('3a',disc,'rho','strength'),('3b',disc,'rho','modulus'),('3c',disc,'integral','strength'),('3d',disc,'end_strain','strength'),('4a',disc,'rho','strength'),('4b',disc,'rho','modulus'),('5a',por,'rho','strength'),('5b',por,'min_section','strength'),('6a',[BY[r['name']] for r in N['A']['figure_numbers']['slit_sweep']],'angle','strength'),('6c',[BY[r['name']] for r in N['A']['figure_numbers']['slit_controls']],'porosity','strength'),('8b',mesh,'alignment','strength'),('8d',mesh,'alignment','integral')]
def value(r,s,k):return r['params']['angle_deg'] if k=='angle' else r[k] if k in ['rho','modulus','min_section','alignment','porosity'] else (r[s] or r['A'])[k]
points=[];summ=[]
for panel,rr,x,y in panels:
 for r in rr:
  q=dict(original_panel=panel,parent_run_id=r['parent_run_id'],name=r['name'],source_phase=r['source_phase'],family=r['family'],x_quantity=x,y_quantity=y,rerun=bool(r['C']),residual_flag=r['residual_flag'])
  for s in ['A','B','C']:q.update({s+'_x':value(r,s,x),s+'_y':value(r,s,y),s+'_endpoint_censored':not (r[s] or r['A'])['failure_observed']})
  for a,b in [('A','B'),('B','C'),('A','C')]:
   for k in ['x','y']:
    q[f'{a}_{b}_{k}_delta']=q[b+'_'+k]-q[a+'_'+k]
    q[f'{a}_{b}_{k}_percent']=100*(q[b+'_'+k]/q[a+'_'+k]-1) if abs(q[a+'_'+k])>1e-14 else None
  points.append(q)
 for k in [x,y]:
  ac=[100*(value(r,'C',k)/value(r,'A',k)-1) if value(r,'A',k) else 0 for r in rr];bc=[100*(value(r,'C',k)/value(r,'B',k)-1) if value(r,'B',k) else 0 for r in rr]
  summ.append(dict(original_panel=panel,quantity=k,population=len(rr),targeted_parents=sum(bool(r['C']) for r in rr),A_C_changed_over_1_percent=sum(abs(v)>1 for v in ac),B_C_changed_over_1_percent=sum(abs(v)>1 for v in bc),largest_A_C_percent_by_absolute_magnitude=max(ac,key=abs),largest_B_C_percent_by_absolute_magnitude=max(bc,key=abs)))
for fn,rows in [('one_to_one_point_changes.csv',points),('panel_change_summary.csv',summ)]:
 with (DATA/fn).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Indexed',len(points),'point/panel memberships and',len(summ),'panel quantities')
