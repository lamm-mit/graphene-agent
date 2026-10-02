"""Aligned population comparisons and held-out descriptor coverage diagnostics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hybrid_geometry import COLORS
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output'
def main():
 r=json.loads((A/'manifest.json').read_text());z=np.load(A/'latent_space.npz');xyz=z['display_xyz'];colors=np.array([COLORS[v['theme_index']] for v in r])/255
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,3,figsize=(12.0,4.25),subplot_kw={'projection':'3d'},facecolor='#05080d')
 for a,ids,title in zip(ax,[np.arange(64000),np.arange(64000,256000),np.arange(256000)],['Original 64,000','Added 192,000','Combined 256,000']):
  a.set_facecolor('#05080d');a.scatter(*xyz[ids].T,c=colors[ids],s=.23,alpha=.6,rasterized=True,depthshade=False);a.set(xlim=(xyz[:,0].min(),xyz[:,0].max()),ylim=(xyz[:,1].min(),xyz[:,1].max()),zlim=(xyz[:,2].min(),xyz[:,2].max()));a.set_box_aspect(np.ptp(xyz,axis=0));a.view_init(20,-60);a.set_axis_off();a.set_title(title,color='#dddddd',fontsize=12,pad=0)
 fig.subplots_adjust(left=0,right=1,bottom=.02,top=.93,wspace=0)
 fig.savefig(OUT/'population_comparison.png',dpi=200,facecolor=fig.get_facecolor());fig.savefig(OUT/'population_comparison.pdf',facecolor=fig.get_facecolor());plt.close(fig)
 q=np.load(A/'bridge_coverage.npz');old=q['old_distance'];new=q['new_distance'];fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
 for values,label,color in [(old,'Original 64,000','#617e92'),(new,'Combined 256,000','#14a8a3')]:
  ax[0].plot(np.sort(values),np.arange(1,len(values)+1)/len(values),label=label,color=color,lw=1.7)
 ax[0].set(xlabel='Distance to nearest design',ylabel='Fraction of held-out queries');ax[0].legend(frameon=False,fontsize=8)
 ax[1].scatter(old,new,s=4,color='#14a8a3',alpha=.3,rasterized=True);m=max(old.max(),new.max());ax[1].plot([0,m],[0,m],color='gray',lw=.8,ls='--');ax[1].set(xlabel='Original nearest distance',ylabel='Expanded nearest distance');fig.tight_layout()
 fig.savefig(OUT/'descriptor_coverage.pdf');fig.savefig(OUT/'descriptor_coverage.svg');fig.savefig(OUT/'descriptor_coverage.png',dpi=200);plt.close(fig)
 print('Coverage figures ready',flush=True)
if __name__=='__main__':main()
