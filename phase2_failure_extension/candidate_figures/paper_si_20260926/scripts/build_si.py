"""%SEPT 26 EDIT: restyle verified original-panel comparisons and generate SI assets."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
from scipy.stats import spearmanr
import fitz
from matplotlib.font_manager import findfont,FontProperties
OUT=Path(__file__).resolve().parents[1]; STUDY=OUT.parents[1]; BASE=STUDY.parent; PAPER=BASE/'generated/loading_si_exports'
SRC=STUDY/'candidate_figures/one_to_one'; F=OUT/'figures'; D=OUT/'data'; T=OUT/'tex'
for p in [F,D,T,OUT/'qa']:p.mkdir(exist_ok=True)
font=str(findfont(FontProperties(family='Arial')));bold=str(findfont(FontProperties(family='Arial',weight='bold')))
designs=json.loads((SRC/'data/designs.json').read_text()); by={r['name']:r for r in designs}; q=json.loads((SRC/'data/quantitative_summary.json').read_text()); clips=json.loads((SRC/'data/panel_clips.json').read_text())
def clipfor(name,ids):
    rr=[fitz.Rect(clips[name][s][i]) for s in ['A','C'] for i in ids]
    bb=rr[0]
    for r in rr[1:]:bb=bb|r
    return bb+(-4,-4,4,4)
specs=[
 dict(id='endpoints',source='fig2_design_space',clip=[2,3],kind='stack',mixed=False),
 dict(id='slits',source='fig3_alignment_cliff',kind='stack',mixed=True),
 dict(id='alignment',source='fig4_hierarchy_alignment',kind='split',mixed=True),
 dict(id='costs',source='fig5_costs',kind='stack',mixed=False),
 dict(id='progression',source='fig7_progression',kind='split',mixed=False),
 dict(id='mechanisms',source='fig_mechanisms',kind='split',mixed=True),
 dict(id='ashby',source='fig2_design_space',kind='split',mixed=False),
 dict(id='boundaries',source='fig2_design_space',kind='overlay',mixed=False),
 dict(id='density',source='fig1_rule_of_mixtures',kind='stack',mixed=False),
 dict(id='loadpaths',source='fig2_load_paths',kind='stack',mixed=False)]
WIDTH=504; HEAD=21; GAP=19
def head(pg,y,letter,label):
    pg.insert_font(fontname='Arial',fontfile=font);pg.insert_font(fontname='ArialBold',fontfile=bold)
    if letter:pg.insert_text((2,y+11),letter,fontsize=11,fontname='ArialBold')
    pg.insert_text((18 if letter else 2,y+10),label,fontsize=9,fontname='Arial')
def clean_source(doc):
    # %SEPT 26 NEW MECHANISMS: keep the former disorder footer in its caption.
    for pg in doc:
        for b in pg.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                text=''.join(span['text'] for span in line['spans'])
                if text.startswith('gray: ordered fine mesh at the same porosity'):
                    pg.add_redact_annot(fitz.Rect(line['bbox']),fill=(1,1,1))
        pg.apply_redactions(images=0,graphics=0)
    return doc
def render(src,dst,letter,label,clip=None):
    doc=clean_source(fitz.open(src));rect=clip or doc[0].rect;h=WIDTH*rect.height/rect.width
    out=fitz.open();pg=out.new_page(width=WIDTH,height=HEAD+h);head(pg,0,letter,label)
    pg.show_pdf_page(fitz.Rect(0,HEAD,WIDTH,HEAD+h),doc,0,clip=clip)
    out.save(dst,garbage=4,deflate=True);doc.close();out.close()
manifest=[]
for sp in specs:
    name=sp['source'];clip=clipfor(name,sp['clip']) if sp.get('clip') else None
    labela='Phase 1 single shot and original Phase 2' if sp['mixed'] else 'Phase 1 single shot'
    labelb='Phase 2 longer loading'
    if sp['kind']=='overlay':
        fn='figS_extension_boundaries.pdf';render(SRC/'figures'/f'{name}__overlay.pdf',F/fn,'','Phase 1 single shot (dashed) and Phase 2 longer loading (shaded)',clip)
        sp['files']=[fn]
    elif sp['kind']=='split':
        sp['files']=[]
        for source,letter,label in [('A','A',labela),('C','B',labelb)]:
            fn=f"figS_extension_{sp['id']}_{letter}.pdf";render(SRC/'figures'/f'{name}__{source}.pdf',F/fn,letter,label,clip);sp['files'].append(fn)
    else:
        da=clean_source(fitz.open(SRC/'figures'/f'{name}__A.pdf'));db=clean_source(fitz.open(SRC/'figures'/f'{name}__C.pdf'));rect=clip or da[0].rect
        assert da[0].rect==db[0].rect;h=WIDTH*rect.height/rect.width;half=HEAD+h
        out=fitz.open();pg=out.new_page(width=WIDTH,height=2*half+GAP)
        for doc,y,letter,label in [(da,0,'A',labela),(db,half+GAP,'B',labelb)]:
            head(pg,y,letter,label);pg.show_pdf_page(fitz.Rect(0,y+HEAD,WIDTH,y+half),doc,0,clip=clip)
        fn=f"figS_extension_{sp['id']}.pdf";out.save(F/fn,garbage=4,deflate=True);sp['files']=[fn];out.close();da.close();db.close()
    for fn in sp['files']:
        doc=fitz.open(F/fn);text=doc[0].get_text()
        assert 'ARCHIVED DATA' not in text and 'PRIMARY EXTENSION DATA' not in text and '|' not in text
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(F/Path(fn).with_suffix('.png').name)
        (F/Path(fn).with_suffix('.svg').name).write_text(doc[0].get_svg_image(text_as_path=False))
        manifest.append(dict(file=fn,source=name,source_clip=list(clip) if clip else None,kind=sp['kind'],header_original=labela,header_extended=labelb,sha256=hashlib.sha256((F/fn).read_bytes()).hexdigest()))
(OUT/'figure_manifest.json').write_text(json.dumps(specs,indent=2)+'\n');(D/'asset_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ['primary_A_B_C_effects.csv','plotted_design_coordinates.csv','boundary_vertices.csv','boundary_vertices.json','one_to_one_point_changes.csv','panel_change_summary.csv','designs.json']:
    shutil.copy2(SRC/'data'/name,D/name)
rr=[r for r in designs if r['C']];disc=[r for r in designs if r['campaign']=='discovery'];target=[r for r in disc if r['C']]
macro={'extNPrimary':'34','extNDiscovery':'20','extNSweeps':'14','extNControls':'2','extNResolution':'4','extNAdditional':'5','extNTotal':'45','extNNoNewPeak':str(sum(abs(r['C']['strength']-r['B']['strength'])<1e-8 for r in rr)),'extNResidual':str(sum(r['residual_flag'] for r in rr)),'extNCensored':str(sum(not r['C']['failure_observed'] for r in rr))}
for key,nm in [('Twenty','S7_slit_20deg'),('Fortyfive','S2_slit_45deg'),('Thirtyfive','P1_slit_35deg'),('Ellipse','S5_H2_D40_W8_ellipseX')]:
    r=by[nm]
    for state,suffix in [('A','Original'),('B','Reproduced'),('C','Extended')]:
        for metric,mn,fmt in [('strength','Strength','.2f'),('integral','Integral','.2f'),('end_strain','EndStrain','.5f')]:macro[f'ext{key}{mn}{suffix}']=format(r[state][metric],fmt)
    macro[f'ext{key}PeakLoadingChange']=f"{100*(r['C']['strength']/r['B']['strength']-1):.2f}"
    macro[f'ext{key}PeakReproductionChange']=f"{100*(r['B']['strength']/r['A']['strength']-1):.2f}"
for s,suf in [('A','Original'),('C','Extended')]:
    macro['extAlignR'+suf]=f"{q['alignment'][s]['all']['pearson']:.4f}";macro['extPremium'+suf]=f"{q['alignment'][s]['premium_mean']:.3f}"
    macro['extIntegralR'+suf]=f"{q['alignment'][s]['integral_correlation']['pearson']:.3f}"
ranks=[]
for cohort,subset in [('96 discovery designs',disc),('20 rerun discovery designs',target)]:
    vals=[]
    for k in ['strength','integral','end_strain']:
        rho=float(spearmanr([r['A'][k] for r in subset],[(r['C'] or r['A'])[k] for r in subset]).statistic);vals.append(rho)
        if cohort.startswith('96'):macro[{'strength':'extRankStrength','integral':'extRankIntegral','end_strain':'extRankStrain'}[k]]=f'{rho:.3f}'
    ranks.append(dict(population=cohort,n=len(subset),strength=vals[0],integral=vals[1],ending_strain=vals[2]))
(D/'rank_comparisons.json').write_text(json.dumps(ranks,indent=2)+'\n')
(T/'loading_extension_numbers.tex').write_text('%SEPT 26 EDIT BEGIN: Generated loading-extension comparison numbers\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macro.items())+'%SEPT 26 EDIT END\n')
rows=[]
for name,label in [('S7_slit_20deg',r'$20^\circ$ slit'),('S2_slit_45deg',r'$45^\circ$ slit'),('P1_slit_30deg',r'$30^\circ$ slit$^{*}$'),('P1_slit_35deg',r'$35^\circ$ slit'),('S5_H2_D40_W8_ellipseX','Square-vein ellipse-X mesh')]:
    r=by[name]
    for metric,ml in [('strength',r'$\sigma_{\max}$ (\Nm)'),('integral',r'$W_{\mathrm{end}}$ (\Jm)'),('end_strain',r'$\varepsilon_{\mathrm{end}}$')]:
        rows.append(label+' & '+ml+' & '+' & '.join(f'{r[s][metric]:.4f}' for s in ['A','B','C'])+r' \\')
table=r"""\begin{table}[H]\centering\footnotesize
\caption{Selected paired measurements. Original denotes the original single-shot or earlier phase-2 record; reproduced stop and extended are two endpoints of the same new trajectory. Their difference isolates added loading, whereas original-to-reproduced differences measure reproduction. The $30^\circ$ design belongs to the earlier phase-2 sweep; its extended endpoint remains censored.}
\label{tab:si-extension-pairs}
\begin{tabular}{@{}llrrr@{}}\toprule
Design & Quantity & Original & Reproduced stop & Extended\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
"""
(T/'loading_extension_pairs.tex').write_text('%SEPT 26 EDIT BEGIN: Generated paired measurements\n'+table+'%SEPT 26 EDIT END\n')
rows=[]
for suffix,label in [('S7_slit_0deg_armchair','Armchair slit'),('S5_H2_D40_W8_ellipseX','Square-vein ellipse-X mesh'),('S2_slit_45deg',r'$45^\circ$ slit'),('S7_slit_20deg',r'$20^\circ$ slit')]:
    obj=json.loads((STUDY/'candidate_figures/data/frozen'/f'resolution_comparison_res__{suffix}.json').read_text())
    cc=obj['comparisons']['C_extended']
    vals=[cc[k]['relative_change_percent'] for k in ['maximum_recorded_stress_N_m','end_strain','stress_strain_integral_J_m2']]
    rows.append(label+' & '+' & '.join(f'{x:+.2f}' for x in vals)+r' \\')
sens=r"""\begin{table}[H]\centering\footnotesize
\caption{Sensitivity to halving the post-damage strain increment from 0.005 to 0.0025, expressed as percent change relative to the primary extension. The elastic increment (0.01) and minimum increment (0.00125) are unchanged. The checks restart from the same initial geometry, so differences combine stepping sensitivity and reproduction variability. All four retain residual flags; this is not a converged zero-step limit.}
\label{tab:si-extension-resolution}
\begin{tabular}{@{}lrrr@{}}\toprule
Design & Peak stress (\%) & Ending strain (\%) & Integral (\%)\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
"""
(T/'loading_extension_resolution.tex').write_text('%SEPT 26 EDIT BEGIN: Generated resolution checks\n'+sens+'%SEPT 26 EDIT END\n')
dest=PAPER/'figures/si/loading_extension';dest.mkdir(parents=True,exist_ok=True)
for p in F.glob('*.pdf'):shutil.copy2(p,dest/p.name)
for name in ['loading_extension_numbers.tex','loading_extension_pairs.tex','loading_extension_resolution.tex']:shutil.copy2(T/name,PAPER/'figures'/name)
print('Created',len(manifest),'SI figure assets, generated numbers and two tables.')
