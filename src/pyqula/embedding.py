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
    def __init__(self,h,m=None,nsuper=None):
        self.H = h.copy() # Pristine Hamiltonian
        self.geometry = self.H.geometry
        self.dimensionality = self.H.dimensionality
        self.has_gf_generator = False
        self.has_spin = self.H.has_spin
        self.has_eh = self.H.has_eh
        self.mode = "bulk" # mode
        self.nsuper = None # supercell between original Hamiltonian
        if m is not None: 
            if type(m)==Hamiltonian: 
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
    def ldos(self,**kwargs): return self.get_ldos(**kwargs)
    def dos(self,**kwargs):
        (x,y,d) = self.ldos(**kwargs)
        return np.sum(d) # sum the DOS
    def multidos(self,es=np.linspace(-1.0,1.0,30),
                   energies=None,**kwargs):
        if energies is not None: es = energies # overwrite
        ds = parallel.pcall(lambda e: self.dos(energy=e,**kwargs),es)
        return (es,np.array(ds))
    def multildos(self,es=np.linspace(-2.,2.,20),**kwargs):
        """Compute the ldos at different energies"""
        fs.rmdir("MULTILDOS")
        fs.mkdir("MULTILDOS")
        ds = [] # total DOS
        fo = open("MULTILDOS/MULTILDOS.TXT","w")
        # parallel execution
        out = parallel.pcall(lambda x: self.ldos(energy=x,**kwargs),es) 
        for (e,o) in zip(es,out):
            (x,y,d) = o # extract
#            (x,y,d) = self.ldos(e=e,**kwargs) # compute LDOS
            ds.append(np.mean(d)) # total DOS
            name0 = "LDOS_"+str(e)+"_.OUT" # name
            name = "MULTILDOS/"+name0
            fo.write(name0+"\n") # name of the file
            np.savetxt(name,np.array([x,y,d]).T) # save data
        np.savetxt("MULTILDOS/DOS.OUT",np.array([es,ds]).T)
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
    def turn_dense(self): pass
    def get_dict(self): return self.H.get_dict()
    def shift_fermi(self,mu): self.H.shift_fermi(mu)
    def get_total_energy(self,**kwargs): return 0.0
    def get_mean_field_hamiltonian(self,**kwargs):
        from .selfconsistency.embedding import hubbard_mf
        return hubbard_mf(self,**kwargs) # return Hubbard mean-field
    def get_didv(self,**kwargs):
        from .embeddingtk import didv
        return didv.get_didv(self,**kwargs)
    # for compatibility with kappa functionality
    def didv(self,**kwargs): self.get_didv(**kwargs) 
    def get_kappa(self,**kwargs):
        from .embeddingtk import kappa
        return kappa.get_kappa(self,**kwargs)






def dos_impurity(h,vc=None,energies=np.linspace(-.5,.5,20),
                   mode="adaptive",delta=0.01,nk=50,silent=True,
                   use_generator=False):
  """ Calculates the green function using the embedding technique"""
  if vc is None: vc = h.intra  # assume perfect
  iden = np.identity(h.intra.shape[0],dtype=np.complex)
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


def onsite_supercell(h,nsuper,mc=None):
    if h.is_multicell: h = h.get_no_multicell() # redefine
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
    inds = []
    k = 0
    n = nsuper[0]*nsuper[1] # number of cells
    intrasuper = [[None for j in range(n)] for i in range(n)]
    for i in range(nsuper[0]):
      for j in range(nsuper[1]):
        inds += [(i,j)]
        k += 1
    from scipy.sparse import bmat
    from scipy.sparse import csc_matrix as csc
    tx = csc(h.tx)
    ty = csc(h.ty)
    txy = csc(h.txy)
    txmy = csc(h.txmy)
    intra = csc(h.intra)
    if mc is None: mc = intra
    else: mc = csc(mc)
    for i in range(n):
      intrasuper[i][i] = intra # intracell
      (x1,y1) = inds[i]
      for j in range(n):
        (x2,y2) = inds[j]
        dx = x2-x1
        dy = y2-y1
        if dx==1 and  dy==0: intrasuper[i][j] = tx
        if dx==-1 and dy==0: intrasuper[i][j] = tx.H
        if dx==0 and  dy==1: intrasuper[i][j] = ty
        if dx==0 and  dy==-1: intrasuper[i][j] = ty.H
        if dx==1 and  dy==1: intrasuper[i][j] = txy
        if dx==-1 and dy==-1: intrasuper[i][j] = txy.H
        if dx==1 and  dy==-1: intrasuper[i][j] = txmy
        if dx==-1 and dy==1: intrasuper[i][j] = txmy.H
    if nsuper[0]%2==1: # central cell
        ii=int(n/2)
        intrasuper[ii][ii] = mc # central onsite
    else:
        ii=int(n/2)
        ii = ii - int(nsuper[0]/2)
        intrasuper[ii][ii] = mc # central onsite
    intrasuper = bmat(intrasuper).todense() # supercell
    return intrasuper






def get_gf(self,energy=0.0,delta=1e-2,nsuper=1,nk=100,operator=None,**kwargs):
    """Return the Green's function"""
    h = self.H
    e = energy
    if self.nsuper is None: # old way
        g,selfe = green.supercell_selfenergy(h,e=e,delta=delta,nk=nk,
                nsuper=nsuper,gtype=self.mode)
    else: # workaround for supercells
        raise # this does not work yet
        g,selfe = green.supercell_selfenergy(h,e=e,delta=delta,nk=nk,
                nsuper=nsuper*self.nsuper) # compute Green's function
        h = h.supercell(self.nsuper) # and redefine with a supercell
    ms = onsite_supercell(h,nsuper)
    n = self.m.shape[0]
    ms = onsite_defective_central(h,self.m,nsuper)
    ns = ms.shape[0] # dimension of the supercell
    iden = np.identity(ns,dtype=np.complex) # identity
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



def get_ldos(self,energy=0.0,delta=1e-2,nsuper=1,nk=100,
                    write = True,
                    operator=None,**kwargs):
    """Compute the local density of states"""
    h = self.H
    # get the Green's function
    gv = self.get_gf(energy=energy,delta=delta,nsuper=nsuper,nk=nk)
    if operator is not None:
        operator = h.get_operator(operator) # overwrite
        gv = operator*gv # multiply
    ds = [-gv[i,i].imag/np.pi for i in range(gv.shape[0])] # LDOS
    ds = full2profile(h,ds,check=False) # resum if necessary
    ds = np.array(ds) # convert to array
    gs = h.geometry.supercell(nsuper)
    x,y = gs.x,gs.y
    if write: np.savetxt("LDOS.OUT",np.array([x,y,ds]).T)
    return x,y,ds








