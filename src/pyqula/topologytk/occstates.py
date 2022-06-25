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
            es,wfs = slg.eigsh(csc_matrix(hk),k=max_waves,which="SA",
                        sigma=0.0,tol=arpack_tol,maxiter=arpack_maxiter)
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




def occupied_states(hkgen,k,window=None,max_waves=None):
    """ Returns the WF of the occupied states in a 2d hamiltonian"""
    hk = hkgen(k) # get hamiltonian
    if max_waves is None: es,wfs = algebra.eigh(hk) # diagonalize all waves
    else:  es,wfs = slg.eigsh(csc_matrix(hk),k=max_waves,which="SA",
                        sigma=0.0,tol=arpack_tol,maxiter=arpack_maxiter)
    wfs = np.conjugate(wfs.transpose()) # wavefunctions
    occwf = []
    for (ie,iw) in zip(es,wfs):  # loop over states
      if window is None: # no energy window
        if ie < 0.:  # if below fermi
          occwf.append(iw)  # add to the list
      else: # energy window provided
        if -abs(window)< ie < 0:  # between energy window and fermi
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
   """Flter certain states according to their eigenvalues"""
   def filt(wfs,k=None):
       n = wfs[0].shape[0]
       iden = np.identity(n,dtype=np.complex)
       op = opk(iden,k=k) # evaluate at a kpoint if needed
       wfs = algebra.disentangle_manifold(wfs,op) # disentangle
       ls = algebra.get_representation(wfs,op) # get their eigenvalue
       ls = [ls[i,i].real for i in range(len(ls))]
       out = []
       for (l,w) in zip(ls,wfs):
           if accept(l): 
               out.append(w)
       return out # return wavefunctions
   return Filter(filt) # return filter



def max_valence_states(h,n=2):
    """Function to filter a maximum number of valence states"""
    opk = h.get_operator("energy") # energy operator
    def filt(wfs,k=None):
       ni = wfs[0].shape[0]
       iden = np.identity(ni,dtype=np.complex)
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


