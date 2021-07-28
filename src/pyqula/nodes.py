
from __future__ import print_function
import numpy as np
import scipy.linalg as lg



def degenerate_points(h,n=0):
    """Return the points in the Brillouin zone that have a node
    in the bandstructure"""
    from scipy.optimize import differential_evolution
    bounds = [(0.,1.) for i in range(h.dimensionality)]
    hk_gen = h.get_hk_gen() # generator
    def get_point(x0):
      def f(k): # conduction band eigenvalues
        hk = hk_gen(k) # Hamiltonian
        es = lg.eigvalsh(hk) # get eigenvalues
        return abs(es[n]-es[n+1]) # gap
      res = differential_evolution(f,bounds=bounds) # minimize
      return res.x
    x0 = np.random.random(h.dimensionality) # inital vector
    return get_point(x0) # get the k-point


def dirac_points(h,n=0,dk=0.01):
    """Look for Dirac points in a Hamiltonian"""
    k = degenerate_points(h,n=n)
    from .topology import berry_phase
    kpath = [[np.cos(i), np.sin(i)] for i in np.pi*np.linspace(0.,2.,10)]
    kpath = [k+dk*np.array(ik) for ik in kpath] # loop
    b = berry_phase(h,kpath=kpath)/np.pi # normalize
    print(k,b)

