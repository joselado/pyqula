import numpy as np
import scipy.sparse as sp
from scipy.sparse import bmat
from .reorder import reorder
from .. import algebra

def hopping2deltaud(H,T):
    """Given a hopping object T, return H plus the up-down pairing

        sum_ij t_ij c^dag_{i,up} c^dag_{j,dn} + h.c.

    with t_ij the (spinless) hoppings of T, which can be any matrix. The
    spin-singlet part of this pairing is the symmetric part of t and its
    triplet part, with the d-vector along z, the antisymmetric part, so a
    real symmetric t gives an extended s-wave, a real antisymmetric t (for
    instance 1j times a Haldane hopping) a pure triplet, and a complex
    Hermitian t (Haldane, Peierls) a mixed singlet-triplet pairing. This can
    be fast for very large systems"""
    H.turn_nambu() # turn the Nambu spinor
    H.turn_multicell() # multicell mode
    T = T.copy() # make a dummy copy
    T.remove_spin() # remove the spin degree of freedom
    T.turn_multicell() # multicell mode
    n = len(H.geometry.r) # number of sites
    def t2h(mud,mdu): # pairing matrix from its up-dn and dn-up site blocks
      pout = [[None for i in range(n)] for j in range(n)] # initialize
      for i in range(n): pout[i][i] = sp.identity(2)*0.
      for i in range(n): # loop over sites
        for j in range(n): # loop over sites
            if np.abs(mud[i,j])<1e-6 and np.abs(mdu[i,j])<1e-6: continue
            pout[i][j] = sp.csc_matrix([[mud[i,j],0.],[0.,mdu[i,j]]])
      diag = sp.identity(2*n)*0. # zero matrix
      pout = bmat(pout) # convert to block matrix
      mout = [[diag,pout],[None,diag]] # output matrix
      mout = bmat(mout) # return full matrix
      return reorder(mout) # reorder the entries properly
    def neg(R): return tuple(-np.array(R,dtype=int))
    td = dict() # hoppings of T, R -> t_R
    for (R,m) in T.get_multihopping().get_dict().items():
        td[tuple(np.array(R,dtype=int))] = algebra.todense(m)
    zero = 0.*td[(0,0,0)]
    # In the Nambu spinor (c_up, c_dn, c_dn^dag, -c_up^dag) the up-dn block
    # D00_R = t_R gives c^dag_{i,up} t c^dag_{j,dn}, while the dn-up block
    # gives D11_R[i,j] c^dag_{j,up} c^dag_{i,dn}, so the same operator needs
    # D11_R = (t_{-R})^T, for any t (conj(t_R) for a Hermitian one). Using
    # t_R for both, as this routine used to, is right only for a symmetric t:
    # for any other the BdG matrix breaks Fermi antisymmetry.
    keys = set(td) | set(neg(R) for R in td)
    P = dict() # electron-hole blocks
    for R in keys:
        P[R] = t2h(td.get(R,zero),np.transpose(td.get(neg(R),zero)))
    out = dict() # electron-hole blocks plus their hole-electron partners
    for R in keys: out[R] = P[R] + algebra.dagger(P[neg(R)])
    from ..multihopping import MultiHopping
    Hout = H.copy()
    Hout.set_multihopping(H.get_multihopping() + MultiHopping(out))
    return Hout
