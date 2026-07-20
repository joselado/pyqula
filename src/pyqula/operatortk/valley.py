import numpy as np
from .. import algebra
from .sharpen import get_sharpen


def get_valley(h,delta=None,**kwargs):
  """Return a callable that calculates the valley expectation value
  using the modified Haldane coupling"""
  if h.dimensionality==0: projector = True # zero dimensional
  ho = h.copy() # copy Hamiltonian
  ho.turn_multicell()
  ho.clean() # set to zero
  ho.add_modified_haldane(1.0/4.5) # add modified Haldane coupling
  hkgen = ho.get_hk_gen() # get generator for the hk Hamiltonian
  sharpen = get_sharpen(delta=delta) # renormalize eigenvalues to +-1
  def fun(m=None,k=None):
      if h.dimensionality>0 and k is None: raise # requires a kpoint
      hk = hkgen(k) # evaluate Hamiltonian
      hk = sharpen(hk) # sharpen the valley
      if m is None: return hk # just return the valley operator
      else: return hk@m # return the projector
  if h.dimensionality==0: return fun() # return a matrix
  return fun # return function

