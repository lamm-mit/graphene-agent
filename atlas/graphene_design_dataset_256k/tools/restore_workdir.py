"""Restore the local assets required by the public plotting/movie scripts.

Usage: python restore_workdir.py /path/to/new/workdir
Then: cd /path/to/new/workdir
      python prepare_tour.py
      python render_latent.py --preview
      python render_latent.py

Does not run molecular dynamics. Requires a new or empty destination; existing working files are not overwritten.
"""
from pathlib import Path
import argparse,gzip,json,shutil,sys,tarfile
from huggingface_hub import snapshot_download
from PIL import Image
REPO='lamm-mit/graphene-design-universe-256k'
def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--revision',default='v1.0.0');args=p.parse_args()
 if args.directory.exists() and any(args.directory.iterdir()):
  raise FileExistsError('Choose a new or empty destination; existing working files will not be overwritten')
 root=Path(snapshot_download(REPO,repo_type='dataset',revision=args.revision,allow_patterns=['code/**','source_masks/**','embedding/*.npz','provenance/manifest.json.gz']))
 dest=args.directory;dest.mkdir(parents=True,exist_ok=True)
 shutil.copytree(root/'code',dest,dirs_exist_ok=True)
 assets=dest/'assets';designs=assets/'designs';designs.mkdir(parents=True,exist_ok=True);(dest/'output').mkdir(exist_ok=True)
 manifest=gzip.decompress((root/'provenance/manifest.json.gz').read_bytes());(assets/'manifest.json').write_bytes(manifest)
 for shard in sorted((root/'source_masks').glob('*.tar')):
  with tarfile.open(shard) as tar:
   for member in tar:
    name=Path(member.name).name
    if name!=member.name or not name.endswith('.npz'):raise ValueError('Unexpected archive member')
    target=designs/name
    if not target.exists():target.write_bytes(tar.extractfile(member).read())
 for name in ['latent_space.npz','geometric_descriptors.npz']:shutil.copy2(root/'embedding'/name,assets/name)
 sys.path.insert(0,str(dest));import fast_geometry as F
 from generate_atlas import thumbnail
 records=json.loads(manifest)
 for r in records:
  stem=designs/f'{r["id"]:04d}';stem.with_suffix('.json').write_text(json.dumps(r,indent=2))
  if not stem.with_suffix('.png').exists():
   P,L,ng,keep=F.load_design(stem.with_suffix('.npz'));Image.fromarray(thumbnail(P,L,ng,keep)).save(stem.with_suffix('.png'))
 import numpy as np,cv2
 sprites=np.lib.format.open_memmap(assets/'sprites.npy',mode='w+',dtype=np.uint8,shape=(len(records),72,72))
 for rec in records:
  sprites[rec['id']]=cv2.resize(np.array(Image.open(designs/f"{rec['id']:04d}.png")),(72,72),interpolation=cv2.INTER_AREA)
 sprites.flush()
 print('Saved masks, metadata, PNGs and embedding restored. The movie sprites are restored. Run python prepare_tour.py and python render_latent.py.')
if __name__=='__main__':main()
