import numpy as np
from .. import algebra
import scipy.sparse.linalg as slg
from scipy.sparse import csc_matrix


arpack_tol = algebra.arpack_tol
arpack_maxiter = algebra.arpack_maxiter


def states_generator(h,filt=None,max_waves=None):
    hkgen = h.get_hk_gen() # get hamiltonian generator
    def agen(k):   
        hk = hkgen(k) # get hamiltonian
        if max_waves is None: 
            es,wfs = algebra.eigh(hk) # diagonalize all waves
        else:  
            es,wfs = algebra.arpack_eigh(csc_matrix(hk),k=max_waves,
                        which="SA",sigma=0.0,tol=arpack_tol,
                        maxiter=arpack_maxiter) # orthonormal states
#        es,wfs = algebra.eigh(hk) # diagonalize all waves
        wfs = np.conjugate(wfs).T
        # now filter positive and negative energies
        if filt is not None: wfs = filt(wfs,k=k) # filter wavefunctions
        return wfs
    return agen


def positive_wf(es,wfs):
    """Return eigenfunctions with positive eigenvalues"""
    es0 = []
    wfs0 = []
    for (e,wf) in zip(es,wfs):
        if e>0.0: # store
            es0.append(e)
            wfs0.append(wf)
    wfs1 = [wf for (e,wf) in sorted(es0,wfs0)] # sorted wavefunctions
    es1 = sorted(es) # sorted energies
    return np.array(wfs1)






def occ_states_generator(h,k,**kwargs):
    """Return a function that generates the occupied wavefunctions"""
    if not h.os_gen:
        hk_gen = h.get_hk_gen() # get hamiltonian generator
        return lambda k: occupied_states(hk_gen,k,**kwargs)
    else: return h.os_gen




def occupied_states(hkgen,k,window=None,max_waves=None,nocc=None):
    """ Returns the WF of the occupied states in a 2d hamiltonian: the
    ones below zero energy, the ones inside window, or the lowest nocc"""
    hk = hkgen(k) # get hamiltonian
    if max_waves is None: es,wfs = algebra.eigh(hk) # diagonalize all waves
    else:  es,wfs = algebra.arpack_eigh(csc_matrix(hk),k=max_waves,
                        which="SA",sigma=0.0,tol=arpack_tol,
                        maxiter=arpack_maxiter) # orthonormal states
    wfs = np.conjugate(wfs.transpose()) # wavefunctions
    if nocc is not None: # the lowest nocc states, whatever their energy
        if not 0<nocc<=len(es):
            raise ValueError("nocc must be between 1 and the "+str(len(es))
              +" states that were computed, got "+str(nocc))
        return np.array(wfs[np.argsort(es)[:nocc]])
    occwf = []
    for (ie,iw) in zip(es,wfs):  # loop over states
      if window is None: # no energy window
        if ie < 0.:  # if below fermi
          occwf.append(iw)  # add to the list
      else: # energy window provided
#        if -np.abs(window)< ie < 0:  # between energy window and fermi
        if window[0] < ie < window[1]:  # between the energy window
            occwf.append(iw)  # add to the list
    return np.array(occwf)


def occ_states2d(h,k):
  """Input is a Hamiltonian"""
  hk_gen = h.get_hk_gen() # get hamiltonian generator
  return occupied_states(hk_gen,k)



class Filter():
    """Class for filtering states"""
    def __init__(self,f):
        if type(f)==Filter: self.f = f.f # store
        if callable(f): self.f = f # store
    def __call__(self,*args,**kwargs):
        return self.f(*args,**kwargs)
    def __mul__(self,a):
        a = Filter(a) # convert
        f2 = lambda *args,**kwargs: self(a.f(*args,**kwargs),**kwargs)
        return Filter(f2) # return new function



def filter_state(opk,accept=lambda r: True, nmax = None):
   """Flter certain states according to their eigenvalues

   The operator has to be Hermitian on the states, since its eigenvalues
   are compared as real numbers; a unitary symmetry with eigenvalues +-i,
   such as a mirror of spinful electrons, goes in as i times it."""
   def filt(wfs,k=None):
       n = wfs[0].shape[0]
       iden = np.identity(n,dtype=np.complex128)
       op = opk(iden,k=k) # evaluate at a kpoint if needed
       wfs = algebra.disentangle_manifold(wfs,op) # disentangle
       ls = algebra.get_representation(wfs,op) # get their eigenvalue
       _check_hermitian(ls)
       ls = [ls[i,i].real for i in range(len(ls))]
       out = []
       for (l,w) in zip(ls,wfs):
           if accept(l): 
               out.append(w)
       return out # return wavefunctions
   return Filter(filt) # return filter



def _check_hermitian(m,tol=1e-6):
    """Raise unless the matrix of an operator on a set of states is
    Hermitian. The states are sorted by the real part of its eigenvalues,
    so an operator with eigenvalues +-i used to put every state at zero,
    and disentangle_manifold diagonalizes it with eigh, which reads one
    triangle only"""
    if len(m)==0: return
    err = np.max(np.abs(m - np.conjugate(m.T)))
    if err>tol*max(1.,np.max(np.abs(m))):
        raise ValueError("the operator is not Hermitian on the occupied "
          +"states (its matrix differs from its conjugate transpose by "
          +str(err)+"), so its eigenvalues are not real and the states "
          +"cannot be sorted by them. A unitary symmetry whose eigenvalues "
          +"are +-i, such as a mirror of spinful electrons, has to be "
          +"passed as i times it")


def _commutation_check(opk,tol=1e-6):
    """Filter that leaves the states as they are, after checking that the
    operator maps their span into itself, which is what an operator that
    commutes with the Hamiltonian does to the occupied states"""
    def filt(wfs,k=None):
        if len(wfs)==0: return wfs
        v = np.conjugate(np.array(wfs)).T # the states, as columns
        op = opk(np.identity(v.shape[0],dtype=np.complex128),k=k)
        ov = op@v
        err = np.max(np.abs(ov - v@(np.conjugate(v.T)@ov)))
        if err>tol*max(1.,np.max(np.abs(op))):
            raise ValueError("the operator does not commute with the "
              +"Hamiltonian: it takes the occupied states out of their own "
              +"span (by "+str(err)+"), so the occupied states have no "
              +"definite eigenvalue of it and no sector to be sorted into. "
              +"topologytk.topologicalsector.get_chern_operator_sign_sector "
              +"splits them by the sign of P O P instead, which is what "
              +"h.get_spin_chern() does with s_z")
        return wfs
    return Filter(filt)


def max_valence_states(h,n=2):
    """Function to filter a maximum number of valence states"""
    opk = h.get_operator("energy") # energy operator
    def filt(wfs,k=None):
       ni = wfs[0].shape[0]
       iden = np.identity(ni,dtype=np.complex128)
       op = opk(iden,k=k) # evaluate at a kpoint if needed
       wfs = algebra.disentangle_manifold(wfs,op) # disentangle
       ls = algebra.get_representation(wfs,op) # get their eigenvalue
       ls = [ls[i,i].real for i in range(len(ls))]
       out = negative_wf(ls,wfs) # stored wavefunctions
       if len(out)>=n: out = out[0:n,:]
       else:
           print("Not enough wavefunctions",len(out),n)
       return out # return wavefunctions
    return Filter(filt)


def negative_wf(es,wfs):
    """Return eigenfunctions with positive eigenvalues"""
    es = np.array(es)
    es0 = []
    wfs0 = []
    for (e,wf) in zip(es,wfs):
        if e<0.0: # store
            es0.append(e)
            wfs0.append(wf)
    es0 = np.array(es0)
    # sorted wavefunctions
    wfs1 = [wf for (e,wf) in sorted(zip(-es0,wfs0),key=lambda x: x[0])] 
    es1 = -np.sort(-es) # sorted energies
    return np.array(wfs1)







def occ_states_sector_generator(H,operator=None,sector=1.0,
        nocc=None,
        tol=1e-3,**kwargs):
    """Return the occupied states generator in a specific sector
    of the Hilbert space given by the operator with value sector"""
    if nocc is None: # no number provided
      focc = filter_state(H.get_operator("energy"),accept= lambda e: e<0.)
    else:
        focc = Filter(lambda wfs,**kwargs: wfs[0:nocc,:]) # lowest nocc
    fop = filter_state(operator,accept= lambda e: np.abs(sector-e)<tol)
    # a non-commuting operator used to give a Chern number of zero, with
    # every state rejected by the tolerance on its eigenvalue
    return states_generator(H,filt=fop*_commutation_check(operator)*focc)








