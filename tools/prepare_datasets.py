#!/usr/bin/env python3
"""Safely unpack shared archives and link course data into assignment folders."""
from pathlib import Path
import os,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets'
def safe(dest,names):
 for name in names:
  if not (dest/name).resolve().is_relative_to(dest.resolve()):raise ValueError(f'Unsafe archive path: {name}')
def main():
 cifar=DATA/'cifar-10-python.tar.gz'
 if cifar.exists() and not (DATA/'cifar-10-batches-py').exists():
  with tarfile.open(cifar) as f:
   members=f.getmembers();safe(DATA,[m.name for m in members])
   if any(m.issym() or m.islnk() for m in members):raise ValueError('Unexpected links in archive')
   f.extractall(DATA,filter='data')
 coco=DATA/'coco_captioning.zip'
 if coco.exists() and not (DATA/'coco_captioning').exists():
  with zipfile.ZipFile(coco) as f:
   safe(DATA,f.namelist());f.extractall(DATA)
 for i in [1,2,3]:
  dest=ROOT/f'assignments/assignment{i}/assignment{i}/cs231n/datasets'
  for name in ['cifar-10-batches-py','coco_captioning','imagenet_val_25.npz']:
   if i==1 and name=='coco_captioning':continue
   target=DATA/name;link=dest/name
   if target.exists() and not link.exists() and not link.is_symlink():
    link.symlink_to(os.path.relpath(target,dest),target_is_directory=target.is_dir());print('LINK',link.relative_to(ROOT),'->',os.readlink(link))
 print('Shared dataset preparation finished. Missing archives are skipped.')
if __name__=='__main__':main()
