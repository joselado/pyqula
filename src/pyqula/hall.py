from . import ribbonizate
from . import multicell
from . import hamiltonians
import numpy as np


def bar_generator(h,n=50,sparse=True):
  """Return a generator of quantum Hall bars, input is magnetic field"""
# create hamiltonian
  hr = multicell.bulk2ribbon(h,n=n,sparse=sparse,nxt=6) 
  def fun(b):
    """Return Hamiltonian of the bar"""
    ho = hr.copy()
    ho.add_peierls(b) # add magnetic field
    return ho
  return fun # return function







def bulk2ribbon(h,mag_field=0.0,n=10,sparse=True,check=True):
  """ Generate the Hamiltonian of a ribbon with magnetic field"""
  if not h.is_multicell: # dense hamiltonian
    ho = ribbonizate.hamiltonian_bulk2ribbon(h,n=n,sparse=sparse,check=check) # create hamiltonian
  else: # sparse hamiltonian
    ho = multicell.bulk2ribbon(h,n=n,sparse=sparse,nxt=6) # create hamiltonian
  ho.geometry.center()
  if np.abs(mag_field)>0.0000001: ho.add_peierls(mag_field) # add peierls phase
  return ho


def landau_levels(h,mag_field=0.01,k=0.0,nl=10,rfactor=1.5):
  """Return the energies of the Landau levels for a certain magnetic
  field"""
  nc = int(round(rfactor*h.geometry.a2[1]/mag_field)) # number of replicas
  hr = bulk2ribbon(h,mag_field=mag_field,n=nc,sparse=True) # generate ribbon
  from . import operators
  # now get the 
  operator = operators.bulk1d(hin,p=0.7)
  hkgen = hin.get_hk_gen()
  hk = hkgen(k) # get hamiltonian
  # eigenvalues and eigenvectors
  eig,eigvec = lg.eigsh(hk,k=nbands,which="LM",sigma=0.0) 
  eigvec = eigvec.transpose() # tranpose the matrix
  eout = [] # output energies
  for (e,v) in zip(eig,eigvec): # loop over eigenvectors
    v = csc_matrix(v)
    a = np.conjugate(v) * operator * v.T
    a = a.data[0]
    a = a.real # real part
    if np.abs(a)<0.1: eout.append(e)
  return eout # return energies of landau levels
