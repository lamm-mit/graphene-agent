"""Verify and restore the local release data bundles, without overwriting files.

python scripts/restore_release_data.py --bundle-dir ../data --all
python scripts/restore_release_data.py --bundle-dir ../data --bundle loading-data
No network access and no automatic simulation execution.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, shutil, tarfile, tempfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def restore(archive, info, root):
    root=Path(root).resolve()
    if digest(archive)!=info['sha256']: raise ValueError(f'Archive checksum mismatch: {archive}')
    expected={r['path']:r for r in info['files']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
            raise ValueError('Archive member inventory mismatch')
        for m in members:
            rel=PurePosixPath(m.name)
            if rel.is_absolute() or '..' in rel.parts or not m.isfile(): raise ValueError('Unsafe archive member')
            target=root/str(rel)
            if not target.resolve().is_relative_to(root): raise ValueError('Destination escapes release')
            if m.size != expected[m.name]['bytes']: raise ValueError('Member size mismatch')
            if target.exists() and digest(target)!=expected[m.name]['sha256']:
                raise FileExistsError(f'Existing file differs; nothing overwritten: {target}')
        restored=0
        for m in members:
            target=root/m.name
            if target.exists(): continue
            target.parent.mkdir(parents=True,exist_ok=True)
            # Write to a temporary file, verify, then publish without replacing.
            fd, tmp=tempfile.mkstemp(prefix='.restore-',dir=target.parent)
            try:
                with os.fdopen(fd,'wb') as out, tar.extractfile(m) as source: shutil.copyfileobj(source,out)
                if digest(tmp)!=expected[m.name]['sha256']: raise ValueError(f'Member checksum mismatch: {m.name}')
                os.link(tmp,target); restored+=1
            finally:
                Path(tmp).unlink(missing_ok=True)
        return restored

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--all',action='store_true');g.add_argument('--bundle',action='append')
    a=p.parse_args();index=json.loads((ROOT/'release/data_bundles.json').read_text())
    names=list(index) if a.all else [n if n.endswith('.tar.gz') else n+'.tar.gz' for n in a.bundle]
    for name in names:
        if name not in index: p.error(f'Unknown bundle {name}')
        print(name,restore(a.bundle_dir/name,index[name],ROOT),'files restored',flush=True)
if __name__=='__main__':main()
