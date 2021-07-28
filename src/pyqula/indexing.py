from __future__ import print_function
import numpy as np
from numba import jit

def bulk1d(g,fac=0.3):
  """Return the indexes of the bulk sites"""
  ymin = np.min(g.y)
  ymax = np.max(g.y)
  out = np.zeros(len(g.r)) # output array
  for i in range(len(g.r)):
    if fac<(g.y[i]-ymin)/(ymax-ymin)<(1.0-fac): out[i] = 1.0
  return out


def bulk0d(g0,fac=0.3):
  """Return the indexes of the bulk sites"""
  g = g0.copy()
  g.center()
  r2 = np.sqrt(g.x**2 + g.y**2 + g.z**2)
  rmax = np.max(r2)
  out = np.zeros(r2.shape) # initialize
  out[r2<(fac*rmax)] = 1.0
  return out


def bulk(g,**kwargs):
    if g.dimensionality==0:  return bulk0d(g,**kwargs)
    elif g.dimensionality==1:  return bulk1d(g,**kwargs)


def get_index(g,r,replicas=False):
    """Given a certain position, return the index of it in the geometry"""
    if replicas: # check the replicas
      ds = g.neighbor_directions()
      rset = [g.replicas(d) for d in ds]  # all sets of replicas
    else:
      rset = [g.r] # list of positions
    for rs in rset: # loop over set of sites
      out = get_index_jit(r,np.array(rs)) # index
      if out>0: 
          return out # valid index
    return None # not found


@jit
def get_index_jit(r0,rs):
    for i in range(len(rs)): # loop
        dr = r0 - rs[i] # different
        dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
        if dr2<1e-5: return i
    return -1
