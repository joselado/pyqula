from __future__ import print_function, division
import numpy as np
from numba import jit
from . import parallel

try:
  from . import density_matrixf90
  use_fortran = True
except:
  use_fortran = False
#  print("Fortran routines not working in densitymatrix.py")
 
use_fortran = False

def full_dm(h,use_fortran=True,nk=10,fermi=0.0,delta=1e-2,ds=None):
  if h.dimensionality == 0: fac = 1.
  elif h.dimensionality == 1: fac = 1./nk
  elif h.dimensionality == 2: fac = 1./nk**2
  elif h.dimensionality == 3: fac = 1./nk**3
  else: raise
  if ds is None: # no directions required
    es,vs = h.get_eigenvectors(nk=nk) # get eigenvectors
    es = es - fermi # shift by the Fermi energy
    if use_fortran:
      dm = density_matrixf90.density_matrix(np.array(es),np.array(vs),delta)
      return dm*fac
    else:
      return np.matrix(full_dm_python(h.intra.shape[0],es,np.array(vs)))*fac # call hte function
  else: # directions required
    es,vs,ks = h.get_eigenvectors(nk=nk,kpoints=True) # get eigenvectors
    es = es - fermi # shift by the Fermi energy
    ks = np.array(ks) # to array
    n = h.intra.shape[0] # dimensionality
    out = parallel.pcall(lambda x: full_dm_python_d(n,es,vs,ks,x)*fac,ds)
#    out = [full_dm_python_d(n,es,vs,ks,d)*fac for d in ds] # compute all the DM
    outd = dict() # dictionary
    for i in range(len(ds)): outd[tuple(ds[i])] = out[i] # as dictionary
    return outd
#    return out # return all the density matrices



def full_dm_python(n,es,vs):
  """Calculate the density matrix"""
  dm = np.zeros((n,n),dtype=np.complex)
  return full_dm_python_jit(n,es,vs,dm)


def full_dm_python_d(n,es,vs,ks,d):
  """Calculate the density matrix"""
  dm = np.zeros((n,n),dtype=np.complex)
  return full_dm_python_d_jit(n,es,vs,ks,np.array(d),dm)



@jit(nopython=True)
def full_dm_python_jit(n,es,vs,dm):
  """Auxiliary function to compute the density matrix"""
  for ie in range(len(es)): # loop
    if es[ie]<0.: # if below Fermi energy
      for i in range(n):
        for j in range(n): 
          dm[i,j] = dm[i,j] + vs[ie][i].conjugate()*vs[ie][j] # add contribution
  return dm


@jit(nopython=True)
def full_dm_python_d_jit(n,es,vs,ks,d,dm):
  """Auxiliary function to compute the density matrix"""
  for ie in range(len(es)): # loop
    k = ks[ie] # get kpoint
    kd = k[0]*d[0] + k[1]*d[1] + k[2]*d[2] # compute scalar product
    phi = np.exp(1j*np.pi*kd*2) # compute phase
#    phi = 0.0
    if es[ie]<0.: # if below Fermi energy
      for i in range(n):
        for j in range(n):
          dm[i,j] = dm[i,j] + phi*vs[ie][i].conjugate()*vs[ie][j] # add contribution
  return dm













def restricted_dm(h,use_fortran=True,mode="KPM",pairs=[],
                   scale=10.0,npol=400,ne=None):
  """Calculate certain elements of the density matrix"""
  if h.dimensionality != 0 : raise
  if mode=="full": # full inversion and then select
    dm = full_dm(h,use_fortran=use_fortran) # Full DM
    outm = np.array([dm[j,i] for (i,j) in pairs]) # get the desired ones
    return outm # return elements
  elif mode=="KPM": # use Kernel polynomial method
    if ne is None: ne = npol*4
    from . import kpm
    xin = np.linspace(-.99*scale,0.0,ne) # input x array
    out = np.zeros(len(pairs),dtype=np.complex)
    ii = 0
    for (i,j) in pairs: # loop over inputs
      (x,y) = kpm.dm_ij_energy(h.intra,i=i,j=j,scale=scale,npol=npol,
                      ne=ne,x=xin)
      out[ii] = np.trapz(y,x=x)/np.pi # pi is here so it normalizes to 0.5
      ii += 1
    return out
  else: raise
       
from . import algebra

def occupied_projector(m,delta=0.0):
    """Return a projector onto the occupied states"""
    (es,vs) = algebra.eigh(m) # diagonalize
    vs = vs.T # transpose
#    vs = vs[es<0.0] # occupied states
    if use_fortran:
      dm = density_matrixf90.density_matrix(np.array(es),np.array(vs),delta)
      return np.array(dm)
    else:
      return np.array(full_dm_python(m.shape[0],es,np.array(vs)))

