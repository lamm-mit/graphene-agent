"""Release data integrity and overwrite protection, using independent fixtures."""
import hashlib, importlib.util, io, tarfile
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('restore',Path(__file__).resolve().parents[1]/'scripts/restore_release_data.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def bundle(tmp_path,name='input/atoms.extxyz'):
    data=b'1\nLattice="2 0 0 0 2 0 0 0 20" pbc="T T F"\nC 0 0 10\n'
    p=tmp_path/'data.tar.gz'
    with tarfile.open(p,'w:gz') as t:
        member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
    info={'sha256':m.digest(p),'files':[{'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}]}
    return p,info,data

def test_restore_idempotent_and_refuses_overwrite(tmp_path):
    p,info,data=bundle(tmp_path);root=tmp_path/'repo';root.mkdir()
    assert m.restore(p,info,root)==1
    assert (root/'input/atoms.extxyz').read_bytes()==data
    assert m.restore(p,info,root)==0
    (root/'input/atoms.extxyz').write_text('working changes')
    with pytest.raises(FileExistsError):m.restore(p,info,root)
    assert (root/'input/atoms.extxyz').read_text()=='working changes'

def test_bad_archive_hash(tmp_path):
    p,info,_=bundle(tmp_path);info['sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'):m.restore(p,info,tmp_path/'repo')

def test_path_traversal(tmp_path):
    p,info,_=bundle(tmp_path,'../escaped')
    with pytest.raises(ValueError,match='Unsafe'):m.restore(p,info,tmp_path/'repo')
    assert not (tmp_path/'escaped').exists()
