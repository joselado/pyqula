import numpy as np
from scipy.sparse import issparse,coo_matrix,csc_matrix


def spin_mixing_part(m):
    """Return the spin mixing part of a matrix"""
    m = coo_matrix(m) # convert
    n = m.shape[0]//2
    d1,r1,c1 = [],[],[]
    d0,r0,c0 = m.data,m.row,m.col
    for i in range(len(d0)):
        if r0[i]%2==0 and c0[i]%2==1:
            d1.append(d0[i])
            r1.append(r0[i]//2)
            c1.append(c0[i]//2)
    mo = csc_matrix((d1,(r1,c1)),shape=(n,n),dtype=np.complex)
    if issparse(m): return np.array(mo.todense())
    else: return mo



def full2profile(h,profile,check=True):
  """Resums a certain profile to show only the spatial dependence"""
  n = len(profile)
  if check:
    if len(profile)!=h.intra.shape[0]: raise # inconsistency
  if h.has_spin == False and h.has_eh==False: out = np.array(profile)
  elif h.has_spin == True and h.has_eh==False:
    out = np.array([profile[2*i]+profile[2*i+1] for i in range(n//2)])
  elif h.has_spin == False and h.has_eh==True:
    out = np.array([profile[2*i]+profile[2*i+1] for i in range(n//2)])
  elif h.has_spin == True and h.has_eh==True:
    out = np.array([profile[4*i]+profile[4*i+1]+profile[4*i+2]+profile[4*i+3] for i in range(n//4)])
  else: raise # unknown
  if check:
    if len(out)!=len(h.geometry.r): raise # mistmach in the dimensions
  return out



