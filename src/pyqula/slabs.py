from . import multicell
import numpy as np

def slab(h,nz=1,nx=1,ny=1,sparse=False,ncut=3):
  """Create slab"""
  hout = multicell.supercell(h,nsuper=[nx,ny,nz],sparse=sparse,ncut=ncut)
  hout.dimensionality = 2
  hout.geometry.dimensionality = 2
  hopping = [] # empty list
  for t in hout.hopping:
    d = t.dir
    if np.abs(d[2])<0.1: # not in the z direction
      hopping.append(t)
  hout.hopping = hopping
  return hout

