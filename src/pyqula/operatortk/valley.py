import numpy as np
from .. import algebra


def get_valley(h,delta=None,**kwargs):
  """Return a callable that calculates the valley expectation value
  using the modified Haldane coupling"""
  if h.dimensionality==0: projector = True # zero dimensional
  ho = h.copy() # copy Hamiltonian
  ho.turn_multicell()
  ho.clean() # set to zero
  ho.add_modified_haldane(1.0/4.5) # add modified Haldane coupling
  hkgen = ho.get_hk_gen() # get generator for the hk Hamiltonian
  def sharpen(m):
    """Sharpen the eigenvalues of a matrix"""
#    return m
    if delta is None: return m # do nothing
    if algebra.issparse(m): return m # temporal workaround
    (es,vs) = algebra.eigh(m) # diagonalize
    es = es/(np.abs(es)+delta) # renormalize the valley eigenvalues
    vs = np.matrix(vs) # convert
    m0 = np.matrix(np.diag(es)) # build new hamiltonian
    return vs@m0@vs.H # return renormalized operator
  def fun(m=None,k=None):
      if h.dimensionality>0 and k is None: raise # requires a kpoint
      hk = hkgen(k) # evaluate Hamiltonian
      hk = sharpen(hk) # sharpen the valley
      if m is None: return hk # just return the valley operator
      else: return hk@m # return the projector
  if h.dimensionality==0: return fun() # return a matrix
  return fun # return function

