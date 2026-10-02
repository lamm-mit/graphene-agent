"""Text-free 4K cinematic exploration of 64000 actual graphene geometries.
No font, label or text rendering API exists in this renderer.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,subprocess,math,shutil
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=200_000_000
from generate_atlas import COLORS
ROOT=Path(__file__).resolve().parent;A=ROOT/'assets';OUT=ROOT/'output';DURATION=120
def ease(x):x=np.clip(x,0,1);return x*x*(3-2*x)
def mix(a,b,u):return np.asarray(a)*(1-u)+np.asarray(b)*u

class Movie:
 def __init__(self,width=3840):
  self.W=width;self.H=width*9//16;self.s=width/1920
  self.r=json.loads((A/'manifest.json').read_text());self.xyz=np.load(A/'latent_space.npz')['display_xyz'];self.colors=np.array([COLORS[r['theme_index']] for r in self.r],np.uint8)
  self.sprites=np.load(A/'sprites.npy');self.atlas=np.array(Image.open(A/'atlas_64000.png').convert('RGB'));self.tour=json.loads((A/'tour.json').read_text())
  self.order=np.array(json.loads((A/'gallery_order.json').read_text()));slot=np.argsort(self.order);row,col=np.divmod(slot,320);self.grid=np.c_[(col-159.5)*80/320,(99.5-row)*80/320,np.zeros(len(row))]
  self.priority=((np.arange(64000,dtype=np.uint64)*2654435761)%4294967296).astype(float)/4294967296
  self.hero={int(p.stem.split('_')[1]):np.array(Image.open(p)) for p in A.glob('hero_*.png')};self.stats={}
 @lru_cache(maxsize=12000)
 def tile(self,id,size):
  g=self.hero[id] if id in self.hero and size>120*self.s else self.sprites[id]
  g=cv2.resize(g,(size,size),interpolation=cv2.INTER_AREA if size<g.shape[0] else cv2.INTER_CUBIC)
  # Slight gamma lift preserves delicate atomic lines at social-media resolution.
  g=np.power(g.astype(np.float32)/255,.80)
  return np.rint(g[:,:,None]*self.colors[id]).astype(np.uint8)
 def paste(self,im,px,py,alpha=1):
  h,w=im.shape[:2];x0,y0=max(0,px),max(0,py);x1,y1=min(self.W,px+w),min(self.H,py+h)
  if x1<=x0 or y1<=y0:return
  crop=im[y0-py:y1-py,x0-px:x1-px]
  if alpha<.999:crop=(crop*alpha).astype(np.uint8)
  # Additive foreground on black: no opaque rectangles conceal neighbouring points.
  self.im[y0:y1,x0:x1]=cv2.max(self.im[y0:y1,x0:x1],crop)
 def gallery(self,t):
  sh,sw=self.atlas.shape[:2];zoom=np.exp((1-ease(t/11))*np.log(10));base=max(sw/(self.W*.96),sh/(self.H*.90))/zoom
  center=np.array([sw*.5,sh*.5]);mat=np.array([[base,0,center[0]-base*self.W/2],[0,base,center[1]-base*self.H/2]],np.float32)
  self.im=cv2.warpAffine(self.atlas,mat,(self.W,self.H),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)
 def state(self,t):
  if t<24:
   u=ease((t-12)/12);return mix(self.grid,self.xyz,u),np.zeros(3),mix(98.765432,112,u),.38*u,.22*u,None,0
  if t<38:
   u=(t-24)/14;return self.xyz,np.zeros(3),112,.38+u*.95,.22+.13*np.sin(u*np.pi),None,0
  if t<56:
   u=ease((t-38)/13);target=np.array(self.tour[0]['xyz']);return self.xyz,target*u,float(np.exp(mix(np.log(112),np.log(13),u))),1.33+.5*(t-38)/18,.25,self.tour[0],ease((t-46)/4)*ease((56-t)/1.25)
  for k,(lo,hi) in enumerate([(56,72),(72,88),(88,104)],start=1):
   if t<hi:
    u=ease((t-lo)/11);target=mix(self.tour[k-1]['xyz'],self.tour[k]['xyz'],u);distance=13+44*np.sin(np.pi*u)**2
    return self.xyz,target,distance,1.83+.55*(k-1)+.55*(t-lo)/16,.25+.08*np.sin(np.pi*u),self.tour[k],ease((t-lo-8)/4)*ease((hi-t)/1.25)
  u=ease((t-104)/13);return self.xyz,mix(self.tour[3]['xyz'],np.zeros(3),u),float(np.exp(mix(np.log(13),np.log(180),u))),3.48+(t-104)*.045,.25+.18*u,None,0
 def project(self,P,target,distance,yaw,pitch):
  forward=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
  right=np.cross([0,1,0],forward);right/=np.linalg.norm(right);up=np.cross(forward,right);Q=P-target
  # Explicit three-term products avoid backend floating-point-status artefacts.
  depth=distance-np.sum(Q*forward,axis=1)
  fac=self.W/np.maximum(depth,.1);xy=np.c_[self.W/2+np.sum(Q*right,axis=1)*fac,self.H/2-np.sum(Q*up,axis=1)*fac]
  assert np.isfinite(xy).all() and np.isfinite(depth).all()
  return xy,depth,fac
 def cloud(self,t):
  P,target,distance,yaw,pitch,focus,focus_alpha=self.state(t);xy,depth,fac=self.project(P,target,distance,yaw,pitch)
  visible=(depth>1)&(xy[:,0]>-100*self.s)&(xy[:,0]<self.W+100*self.s)&(xy[:,1]>-100*self.s)&(xy[:,1]<self.H+100*self.s)
  ids=np.flatnonzero(visible);brightness=np.clip(.50+(distance-depth)/100,.22,.83);col=(self.colors*brightness[:,None]).astype(np.uint8)
  pixels=np.rint(xy[ids]).astype(int);radius=max(1,round(self.s*.65))
  for dy in range(-radius,radius+1):
   for dx in range(-radius,radius+1):
    if dx*dx+dy*dy>radius*radius:continue
    xx=pixels[:,0]+dx;yy=pixels[:,1]+dy;inside=(xx>=0)&(yy>=0)&(xx<self.W)&(yy<self.H)
    self.im[yy[inside],xx[inside]]=col[ids[inside]]
  # Keep every design as a point; select spatially dispersed thumbnail glyphs.
  # Their centres remain at the exact projected embedding coordinates.
  binsize=23*self.s
  order=ids[np.argsort(depth[ids]/max(distance,1)*.3+self.priority[ids]*.7)]
  bins=np.floor(xy[order]/binsize).astype(np.int32)
  _,first=np.unique(bins,axis=0,return_index=True)
  sprite_ids=order[np.sort(first)].tolist()
  if len(sprite_ids)>1900:
   sprite_ids=sorted(sprite_ids,key=lambda x:self.priority[x])[:1900]
  sizes=np.minimum(46*self.s,np.maximum(12*self.s,.62*fac))
  for id in sorted(sprite_ids,key=lambda x:depth[x],reverse=True):
   size=max(8,round(sizes[id]/2)*2);alpha=float(np.clip(brightness[id]+.25,.3,1));self.paste(self.tile(id,size),round(xy[id,0]-size/2),round(xy[id,1]-size/2),alpha)
  # Magnified callouts make actual neighbours readable. Thin leaders retain their
  # true projected map anchors; callout positions are editorial, not coordinates.
  if focus is not None and focus_alpha>0:
   hero=focus['id'];near=[focus['nearest_feature_neighbors'][j] for j in [0,2,4,7,12,20]];origin=xy[hero]
   self.im=(self.im*(1-.70*focus_alpha)).astype(np.uint8)
   positions=[]
   for j,id in enumerate(near):
    angle=2*np.pi*j/6;callout=np.array([self.W/2+630*self.s*np.cos(angle),self.H/2+315*self.s*np.sin(angle)])
    anchor=xy[id] if depth[id]>1 else origin
    point=mix(anchor,callout,focus_alpha);positions.append((id,point))
    a,b,c=np.rint([origin,anchor,point]).astype(np.int32)
    cv2.line(self.im,tuple(a),tuple(b),tuple(int(x*focus_alpha) for x in [80,97,111]),max(1,round(self.s*.7)),cv2.LINE_AA)
    cv2.line(self.im,tuple(b),tuple(c),tuple(int(x*focus_alpha) for x in [96,119,135]),max(1,round(self.s*.9)),cv2.LINE_AA)
   for id,point in [(hero,origin)]+positions:
    size=round((325 if id==hero else 182)*self.s*(.5+.5*focus_alpha));pad=round(8*self.s);x,y=np.rint(point-size/2).astype(int)
    x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(self.W,x+size+pad),min(self.H,y+size+pad)
    if x1>x0 and y1>y0:self.im[y0:y1,x0:x1]=(self.im[y0:y1,x0:x1]*(1-.94*focus_alpha)).astype(np.uint8)
    self.paste(self.tile(id,size),x,y,focus_alpha)
  self.stats=dict(time=t,points_in_view=int(visible.sum()),thumbnail_glyphs=len(sprite_ids),focus_id=None if focus is None else focus['id'])
 def frame(self,t):
  self.im=np.zeros((self.H,self.W,3),np.uint8)
  if t<12:self.gallery(t)
  else:self.cloud(t)
  # Brief sequential cut between atlas texture and the 3D representation.
  fade=min(ease(t/.6),ease((120-t)/.8))
  if 11.8<t<12:fade*=ease((12-t)/.2)
  if 12<=t<12.2:fade*=ease((t-12)/.2)
  return cv2.convertScaleAbs(self.im,alpha=fade) if fade<.999 else self.im

def main():
 p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=3840);p.add_argument('--preview',action='store_true');args=p.parse_args();m=Movie(args.width)
 if args.preview:
  times=[1,6,11,16,22,30,37,44,52,61,69,77,85,93,101,108,114,118];tiles=[]
  for t in times:
   im=m.frame(t);Image.fromarray(im).save(OUT/f'preview_{t:03d}s.jpg',quality=95);tiles.append(cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA))
  Image.fromarray(np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)])).save(OUT/'contact_sheet.jpg',quality=95);print('Previews ready');return
 final=OUT/f'Graphene_64000_Design_Universe_NoText_{m.W}x{m.H}.mp4';part=final.with_suffix('.rendering.mp4')
 if final.exists() or part.exists():raise FileExistsError(final)
 with (OUT/f'encode_{m.W}.log').open('w') as log:
  process=subprocess.Popen([(shutil.which('ffmpeg') or 'ffmpeg'),'-hide_banner','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{m.W}x{m.H}','-r','30','-i','-','-an','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-preset','fast','-crf','18','-threads','8','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(part)],stdin=subprocess.PIPE,stderr=log)
  try:
   for k in range(120*30):
    process.stdin.write(m.frame(k/30).tobytes())
    if k%300==0:print('Rendered',k/30,'seconds',m.stats,flush=True)
   process.stdin.close();assert process.wait()==0
  finally:
   if process.poll() is None:process.terminate();process.wait()
 part.rename(final);print(final,flush=True)
if __name__=='__main__':main()
