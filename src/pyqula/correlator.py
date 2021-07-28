from __future__ import print_function
import scipy.linalg as lg
import scipy.sparse.linalg as slg
from scipy.sparse import csc_matrix,eye
import numpy as np

def correlator0d(m,energies=np.linspace(-10.,10.,400),i=0,j=0,delta=0.07):
  """Calculate a certain correlator"""
  iden = np.identity(m.shape[0],dtype=np.complex)
  zs = np.zeros(energies.shape[0],dtype=np.complex)
  for (ie,e) in zip(range(len(energies)),energies):
    m0 = ((e+1j*delta)*iden - m).I # inverse 
    zs[ie] = m0[i,j]
  np.savetxt("CORRELATOR.OUT",np.matrix([energies,zs.real,-zs.imag]).T)
  print("Saved correlator in CORRELATOR.OUT")
  return (energies,zs.real,-zs.imag)



def gs_correlator(m,i=0,j=0):
    """Compute the integrated correlator using wavefunctions"""
    (es,vs) = lg.eigh(m) # diagonalize
    vs = vs.transpose() # transpose matrix
    c = 0.0
    for (e,v) in zip(es,vs):
        if e<0.0:
            c += v[i]*np.conjugate(v[j])
    return c.real


def dm_ij_energy(m,ne=500,scale=10.,i=0,j=0,delta=0.07):
  """Calculate a certain correlator"""
  energies = np.linspace(-scale,scale,ne)
  iden = np.identity(m.shape[0],dtype=np.complex)
  zs0 = np.zeros(energies.shape[0],dtype=np.complex)
  zs1 = np.zeros(energies.shape[0],dtype=np.complex)
  for (ie,e) in zip(range(len(energies)),energies):
    m0 = ((e+1j*delta)*iden - m).I # inverse 
    m1 = ((e-1j*delta)*iden - m).I # inverse 
    zs0[ie] = m0[i,j]
    zs1[ie] = m1[i,j]
  z = (zs1 - zs0)/2. # both contributions
  return (energies,-z*1j)






def gij(m,i=0,delta=0.01,e=0.0):
  """Calculate a single row of the Green function"""
  v0 = np.zeros(m.shape[0])
  v0[i] = 1.
  iden = eye(v0.shape[0]) # identity matrix
  g = iden*(e+1j*delta) - csc_matrix(m) # matrix to invert
#  print(type(g)) ; exit()
  (b,info) = slg.lgmres(g,v0) # solve the equation  
  go = (b*np.conjugate(b)).real
  return go



