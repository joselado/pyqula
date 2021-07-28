from scipy.linalg import eigvalsh
import numpy as np

def eigenvalues(h,nk):
    """Return all the eigenvalues of a Hamiltonian"""
    import klist
    h.turn_dense()
    ks = klist.kmesh(h.dimensionality,nk=nk) # get grid
    hkgen = h.get_hk_gen() # get generator
    e0 = 0.0
    import timing
    est = timing.Testimator(maxite=len(ks))
    for k in ks: # loop
      est.iterate()
      es = eigvalsh(hkgen(k)).tolist() # add
      e0 += np.sum(es[es<0.]) # add contribution
    return e0/len(ks) # return the ground state energy

#
#def gsenergy(h,nk=10):
#  """Calculate the ground state energy"""
#  if h.dimensionality!=2: raise  # continue if two dimensional
#  hk_gen = h.get_hk_gen() # gets the function to generate h(k)
#  kxs = np.linspace(0.,1.,nk)  # generate kx
#  kys = np.linspace(0.,1.,nk)  # generate ky
#  e0 = 0.0 # initalize
#  for x in kxs:
#    for y in kys:
#      hk = hk_gen([x,y]) # get hamiltonian
#      es = eigvalsh(hk) # eigenvalues
#      e0 += np.sum(es[es<0.]) # add contribution
#  return e0/nk**2
