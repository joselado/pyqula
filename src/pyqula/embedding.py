# library to perform embedding calculations
from __future__ import print_function

from . import green
from . import parallel
from . import algebra
from . import timing
import numpy as np
import scipy.linalg as lg
import os
from .increase_hilbert import full2profile
from . import filesystem as fs
from .hamiltonians import Hamiltonian


class Embedding():
    """Define an embedding object"""
    def __init__(self,h,m=None,nsuper=None,selfenergy=None):
        self.H = h.copy() # Pristine Hamiltonian
        self.geometry = self.H.geometry
        self.dimensionality = self.H.dimensionality
        self.selfenergy = selfenergy 
        self.has_gf_generator = False
        self.has_spin = self.H.has_spin
        self.has_eh = self.H.has_eh
        self.gf_mode = "renormalization" # mode for Green's functions
        self.boundary_embedding_generator = None
        self.mode = "bulk" # mode
        self.nsuper = None # supercell between original Hamiltonian
        if m is not None: 
            if isinstance(m, Hamiltonian):
#                print("Picking intracell in embedding")
                m = m.intra # get the intracell
            self.m = m # provided matrix
            if m.shape[0]!=h.intra.shape[0]: 
                if nsuper is None:
                    print("Dimensions do not match in embedding")
                    raise
                else: 
                    self.nsuper = nsuper # store the supercell
                    hs = h.supercell(nsuper)
                    if m.shape[0]!=hs.intra.shape[0]: 
                        print("Dimensions do not match after supercell")
                        print(m.shape[0],hs.intra.shape[0])
                        raise
            else: pass
        else: self.m = h.intra.copy() # pristine one
    def get_gf(self,**kwargs):
        return get_gf(self,**kwargs)
    def get_density_matrix(self,**kwargs):
        return get_dm(self,**kwargs)
    def get_ldos(self,**kwargs):
        return get_ldos(self,**kwargs)
    def ldos(self,**kwargs): 
        return self.get_ldos(**kwargs)
    def get_dos(self,**kwargs):
        (x,y,d) = self.get_ldos(**kwargs)
        return np.sum(d) # sum the DOS
    def dos(self,**kwargs): 
        return self.get_dos(**kwargs)
    def multidos(self,es=np.linspace(-1.0,1.0,30),
                   energies=None,**kwargs):
        if energies is not None: es = energies # overwrite
        ds = parallel.pcall(lambda e: self.dos(energy=e,**kwargs),es)
        return (es,np.array(ds))
    def multildos(self,**kwargs):
        """Compute the ldos at different energies"""
        from .embeddingtk.ldos import multildos
        return multildos(self,**kwargs)
    def set_multihopping(self,mh):
        """Set a multihopping as the impurity"""
        dd = mh.get_dict() # get the dictionary
        for d in dd:
            if d not in [(0,0,0)]: pass#raise # not implemented
            else:
                self.m = dd[d] # copy intra cell matrix
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def get_energy_ingap_state(EO,**kwargs):
        from .embeddingtk import ingap
        return ingap.energy_ingap_state(EO,**kwargs)
    # dummy methods for compatibility
    def turn_multicell(self): pass
    def get_multicell(self): return self
#    def turn_dense(self): pass
    def get_dense(self): return self
    def get_dict(self): return self.H.get_dict()
    def shift_fermi(self,mu): self.H.shift_fermi(mu)
    def get_total_energy(self,**kwargs): return 0.0
    def get_mean_field_hamiltonian(self,**kwargs):
        from .selfconsistency.embedding import hubbard_mf
        return hubbard_mf(self,**kwargs) # return Hubbard mean-field
    def get_didv(self,**kwargs):
        from .embeddingtk import didv
        return didv.get_didv_single(self,**kwargs)
    def get_didv_all(self,**kwargs):
        from .embeddingtk import didv
        return didv.get_didv(self,**kwargs)
    # for compatibility with kappa functionality
    def didv(self,**kwargs): 
        return self.get_didv(**kwargs) 
    def get_kappa(self,**kwargs):
        from .embeddingtk import kappa
        return kappa.get_kappa(self,**kwargs)






def dos_impurity(h,vc=None,energies=np.linspace(-.5,.5,20),
                   mode="adaptive",delta=0.01,nk=50,silent=True,
                   use_generator=False):
  """ Calculates the green function using the embedding technique"""
  if vc is None: vc = h.intra  # assume perfect
  iden = np.identity(h.intra.shape[0],dtype=np.complex128)
  if use_generator:
    getgreen = green.green_generator(h,nk=nk) # get the function
    def get_green(energy): return getgreen(energy,delta=delta)
  else: # use the conventional method
    def get_green(energy):
      return green.bloch_selfenergy(h,energy=energy,delta=delta,nk=nk,
                                       mode=mode)
  def pfun(energy): # function to parallelize
    g,selfe = get_green(energy) # get Green and selfenergy
    emat = iden*(energy + delta*1j)  # E +i\delta 
    gv = lg.inv(emat - vc -selfe)   # Green function of a vacancy, with selfener
    d = -np.trace(g).imag  # save DOS of the pristine
    dv = -np.trace(gv).imag  # save DOS of the defected
    if not silent: print("Done",energy)
    return [d,dv]
  out = np.array(parallel.pcall(pfun,energies)) # compute
  ds,dsv = out[:,0],out[:,1] # get the different data
  np.savetxt("DOS_PRISTINE.OUT",np.array([energies,ds]).T)
  np.savetxt("DOS_DEFECTIVE.OUT",np.array([energies,dsv]).T)
  return ds,dsv # return object





def bulk_and_surface(h1,nk=100,energies=np.linspace(-1.,1.,100),**kwargs):
  """Get the surface DOS of an interface"""
  from scipy.sparse import csc_matrix,bmat
  if h1.dimensionality==2:
      kpath = [[k,0.,0.] for k in np.linspace(0.,1.,nk)]
  else: raise
  ik = 0
  h1 = h1.get_multicell() # multicell Hamiltonian
  tr = timing.Testimator("DOS") # generate object
  dos_bulk = energies*0.0
  dos_sur = energies*0.0
  for k in kpath:
    tr.remaining(ik,len(kpath)) # generate object
    ik += 1
    outs = green.surface_multienergy(h1,k=k,energies=energies,**kwargs)
    dos_bulk += np.array([-algebra.trace(g[1]).imag for g in outs])
    dos_sur += np.array([-algebra.trace(g[0]).imag for g in outs])
  dos_bulk /= len(kpath)
  dos_sur /= len(kpath)
  np.savetxt("DOS.OUT",np.array([energies,dos_bulk,dos_sur]).T)
  return energies,dos_bulk,dos_sur




def onsite_defective_central(h,m,nsuper):
    return onsite_supercell(h,nsuper,mc=m)



def onsite_supercell_multicell(h,nsuper,mc=None):
    """Compute the onsite matrix of a supercell, with a defect mc"""
    if nsuper==1: hs = h.copy() # just make a copy
    else: hs = h.get_supercell(nsuper) # get a supercell of the Hamiltonian
    # new lets replace the onsite matrix of the defective cell
    ni = h.intra.shape[0] # dimension of minimal cell
    nis = hs.intra.shape[0] # dimension of supercell
    if mc is None: return hs.intra
    ### setup the central (defective) cell
    if h.dimensionality==1: n = nsuper # number of unit cells
    elif h.dimensionality==2: n = nsuper**2 # number of unit cells
    else: raise
    if h.dimensionality==1:
        ic=int(n//2) # central site
    elif h.dimensionality==2:
        if nsuper%2==1: # odd supercell
            ic=int(n//2)
        else: # even supercell
            ic=int(n//2) # central
            ic = ic - int(nsuper//2)
    ii = ic*ni # first index
    jj = (ic+1)*ni # last index
    hs.intra[ii:jj,ii:jj] = mc[:,:] # replace
    return hs.intra # return intracell

# redefine
def onsite_supercell(h0,nsuper,**kwargs):
    """Generic function for supercell"""
    try: # try to use the non multicell method
        h = h0.get_no_multicell()
        multicell_mode = False
    except: # if not possible, try the multicell one (not well tested yet)
        h = h0.copy()
        multicell_mode = True
    if multicell_mode:
        print("WARNING, Multicell function in onsite_supercell")
        return onsite_supercell_multicell(h,nsuper,**kwargs)
    else:
        return onsite_supercell_no_multicell(h,nsuper,**kwargs)



def onsite_supercell_no_multicell(h,nsuper,mc=None):
    """Compute the onsite matrix of a supercell, with a defect mc"""
    if nsuper==1:
        if mc is None: return h.intra # pristine
        else: return mc # defective
    if h.is_multicell: # try to make it multicell
        from .htk.kchain import detect_longest_hopping
        if detect_longest_hopping(h)>1:
            print("This function requires short-range hopping, stopping")
            raise # up to NN
        h = h.get_no_multicell() # redefine
    from .checkclass import is_iterable
    if not is_iterable(nsuper): # just a number
        if h.dimensionality==1: nsuper = [nsuper,1]
        elif h.dimensionality==2: nsuper = [nsuper,nsuper]
        else: raise
    if h.dimensionality>2: raise
    #### this is a dirty workaround ####
    if h.dimensionality==1:
        h = h.copy()
        h.tx = h.inter
        h.ty = h.intra*0.0
        h.txy = h.intra*0.0
        h.txmy = h.intra*0.0
    #### end of the dirty trick ###
    nx,ny = nsuper[0],nsuper[1]
    n = nx*ny # number of cells
    dim = h.intra.shape[0] # orbitals per cell
    dag = algebra.dagger
    intra = algebra.todense(h.intra)
    tx = algebra.todense(h.tx)
    ty = algebra.todense(h.ty)
    txy = algebra.todense(h.txy)
    txmy = algebra.todense(h.txmy)
    mc = intra if mc is None else algebra.todense(mc)
    # place only the O(n) nonzero neighbor blocks directly in the dense
    # output, instead of building an n x n (mostly-None) block list and
    # routing it through scipy.sparse.bmat -- for the fixed, regular
    # nearest-neighbor block structure used here, the sparse machinery
    # (COO/CSC construction, index-dtype inference) costs far more than
    # the O(n) numpy slice-assignments it replaces.
    def kidx(x,y): return x*ny+y # matches the (i,j) row-major order below
    out = np.zeros((n*dim,n*dim),dtype=np.complex128)
    for x in range(nx):
        for y in range(ny):
            k = kidx(x,y)
            out[k*dim:(k+1)*dim,k*dim:(k+1)*dim] = intra # intracell
            if x+1<nx: # (dx,dy) = (1,0) and its dagger
                k2 = kidx(x+1,y)
                out[k*dim:(k+1)*dim,k2*dim:(k2+1)*dim] = tx
                out[k2*dim:(k2+1)*dim,k*dim:(k+1)*dim] = dag(tx)
            if y+1<ny: # (dx,dy) = (0,1) and its dagger
                k2 = kidx(x,y+1)
                out[k*dim:(k+1)*dim,k2*dim:(k2+1)*dim] = ty
                out[k2*dim:(k2+1)*dim,k*dim:(k+1)*dim] = dag(ty)
            if x+1<nx and y+1<ny: # (dx,dy) = (1,1) and its dagger
                k2 = kidx(x+1,y+1)
                out[k*dim:(k+1)*dim,k2*dim:(k2+1)*dim] = txy
                out[k2*dim:(k2+1)*dim,k*dim:(k+1)*dim] = dag(txy)
            if x+1<nx and y-1>=0: # (dx,dy) = (1,-1) and its dagger
                k2 = kidx(x+1,y-1)
                out[k*dim:(k+1)*dim,k2*dim:(k2+1)*dim] = txmy
                out[k2*dim:(k2+1)*dim,k*dim:(k+1)*dim] = dag(txmy)
    ### setup the central (defective) cell
    if h.dimensionality==1:
        ii=int(n//2) # central site
    elif h.dimensionality==2:
        if nsuper[0]%2==1: # odd supercell
            ii=int(n//2)
        else: # even supercell
            ii=int(n//2) # central
            ii = ii - int(nsuper[0]//2)
    else: raise
    out[ii*dim:(ii+1)*dim,ii*dim:(ii+1)*dim] = mc # central onsite
    return out


def get_gf(self,**kwargs):
    """Return the Green's function"""
    if self.selfenergy is None: # no selfenergy given, compute it
        return get_gf_exact(self,**kwargs)
    else: # a selfenergy is given
        from .embeddingtk.boundaryembedding import boundary_embedding_gf
        return boundary_embedding_gf(self,selfenergy=self.selfenergy,**kwargs)


def get_gf_exact(self,energy=0.0,delta=1e-2,
        nsuper=1,nk=100,operator=None,**kwargs):
    """Return the Green's function"""
    h = self.H
    e = energy
    if self.nsuper is None: # old way
        g,selfe = green.supercell_selfenergy(h,e=e,delta=delta,nk=nk,
                nsuper=nsuper,gtype=self.mode,gf_mode=self.gf_mode)
    else: # workaround for supercells
        raise # this does not work yet
        g,selfe = green.supercell_selfenergy(h,e=e,delta=delta,nk=nk,
                nsuper=nsuper*self.nsuper) # compute Green's function
        h = h.supercell(self.nsuper) # and redefine with a supercell
    ms = onsite_defective_central(h,self.m,nsuper)
    ns = ms.shape[0] # dimension of the supercell
    iden = np.identity(ns,dtype=np.complex128) # identity
    emat = iden*(e + delta*1j) # energy matrix
    gv = algebra.inv(emat - ms -selfe)   # Defective Green function
    return gv


def get_onsite(self,nsuper=1,**kwargs):
    h = self.H
    ms = onsite_defective_central(h,self.m,nsuper)
    return ms



from .embeddingtk.embedded import get_dm
from .embeddingtk.embedded import embed_hamiltonian
from .embeddingtk.selfenergies import get_selfenergy_from_potential
from .embeddingtk.ldos import get_ldos











