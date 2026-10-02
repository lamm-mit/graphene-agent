"""Sharing copy, complete decode, encoded visual samples and delivery audit."""
from pathlib import Path
import subprocess,json,ast,shutil
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output';A=ROOT/'assets';FF=(shutil.which('ffmpeg') or 'ffmpeg');FP=(shutil.which('ffprobe') or 'ffprobe')
def main():
 source=OUT/'Graphene_64000_Design_Universe_NoText_3840x2160.mp4';share=OUT/'Graphene_64000_Design_Universe_NoText_1920x1080.mp4';assert source.exists()
 if not share.exists():
  part=share.with_suffix('.rendering.mp4');subprocess.run([FF,'-hide_banner','-loglevel','error','-n','-i',str(source),'-vf','scale=1920:1080:flags=lanczos','-c:v','libx264','-preset','fast','-crf','19','-threads','8','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(part)],check=True);part.rename(share)
 print('Sharing copy ready',flush=True);results=[]
 for p in [source,share]:
  subprocess.run([FF,'-v','error','-i',str(p),'-f','null','-'],check=True)
  q=json.loads(subprocess.check_output([FP,'-v','error','-show_streams','-show_format','-of','json',str(p)]));s=q['streams'];assert len(s)==1 and s[0]['codec_type']=='video';v=s[0]
  assert int(v['nb_frames'])==3600 and v['avg_frame_rate']=='30/1' and abs(float(v['duration'])-120)<.01
  results.append(dict(file=p.name,width=v['width'],height=v['height'],frames=int(v['nb_frames']),duration_s=float(v['duration']),fps=v['avg_frame_rate'],full_decode_passed=True,subtitle_tracks=0,audio_tracks=0,bytes=p.stat().st_size));print('Decoded',p.name,flush=True)
 times=[1,6,11,16,22,30,37,44,52,61,69,77,85,93,101,108,114,118];cap=cv2.VideoCapture(str(source));tiles=[]
 for t in times:
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=cap.read();assert ok;im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);Image.fromarray(im).save(OUT/f'encoded_{t:03d}s.jpg',quality=95);tiles.append(cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA))
 cap.release();Image.fromarray(np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)])).save(OUT/'encoded_contact_sheet.jpg',quality=95)
 tree=ast.parse((ROOT/'render_latent.py').read_text());calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)];assert not any(x in calls for x in ['text','putText','truetype','load_default','drawtext'])
 r=json.loads((A/'manifest.json').read_text());e=np.load(A/'latent_space.npz');assert len(r)==len(e['display_xyz'])==64000;assert len({v['geometry_digest'] for v in r})==64000;assert np.isfinite(e['embedding']).all();assert sorted(json.loads((A/'gallery_order.json').read_text()))==list(range(64000))
 (OUT/'delivery_qa.json').write_text(json.dumps(dict(movies=results,text_rendering_calls=0,encoded_samples=times,unique_geometries=64000,embedding_finite=True,gallery_contains_each_design_once=True),indent=2));print('Delivery verified',flush=True)
if __name__=='__main__':main()
