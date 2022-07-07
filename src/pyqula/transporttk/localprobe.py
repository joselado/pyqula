import numpy as np
from .. import algebra
from ..green import green_renormalization
from .. import green
from copy import deepcopy

delta_smatrix = 1e-12
dagger = algebra.dagger
gfmode = "adaptive"

# library to perform transport calculations using a local probe


class LocalProbe():
    def __init__(self,h,lead=None,delta=1e-5,i=0,T=1.0,**kwargs):
        self.H = h.copy() # store Hamiltonian
        self.H.turn_dense() # dense Hamiltonian
        self.has_eh = self.H.has_eh # electron-hole
        self.delta = delta
        self.mode = "bulk"
        self.reuse_gf = False # reuse the Green's function
        self.gf = None
        self.bulk_delta = delta
        self.frozen_lead = True
        self.i = i # this site
        if lead is None:
            from ..geometry import chain 
            lead = chain().get_hamiltonian(has_spin=False) # create a chain
            if self.H.has_spin: lead.turn_spinful()
            if self.H.has_eh: lead.turn_nambu()
        lead = lead.get_no_multicell() # no multicell
        self.lead = lead.copy() # store
        self.get_eh_sector = self.lead.get_eh_sector # if it has electron-hole
        self.T = T # transparency
    def get_selfenergy(self,energy,lead=0,**kwargs):
        """Return the selfenergies"""
        if lead==0: # use the probe
            return lead_selfenergy(self,energy=energy,**kwargs)
        elif lead==1: # use the system
            g = generate_gf(self,energy=energy,
                               **kwargs) # generate the Green's function
            return local_selfenergy(self.H,g,i=self.i,
                                energy=energy,**kwargs)
        else: raise
    def get_central_gmatrix(self,**kwargs):
        return get_central_gmatrix(self,**kwargs)
    def get_reflection_normal_lead(self,s):
        return get_reflection_normal_lead(self,s)
    def didv(self,T=None,**kwargs):
        from .didv import didv
        return didv(self,**kwargs)
    def copy(self): return deepcopy(self)
    def set_coupling(self,c):
        self.T = c # set the coupling
    def remove_pairing(self):
        self.H.remove_pairing()
        self.lead.remove_pairing()
    def get_kappa(self,T=None,**kwargs):
        from .kappa import get_kappa_ratio
        if T is None: T = self.T 
        return get_kappa_ratio(self,T=T,**kwargs)




def generate_gf(self,energy=0.0,**kwargs):
    """Generate the specific Green's function"""
    mode = self.mode 
    # just a trick to reuse the GF if needed
    if self.reuse_gf and self.gf is not None: return self.gf 
    else:
        gf = self.H.get_gf(energy=energy,delta=self.bulk_delta,
                             mode=gfmode,
                             gtype=mode)
        if self.reuse_gf: self.gf = gf # overwrite
        return gf


def lead_selfenergy(self,energy=0.0,**kwargs):
     """Return the selfenergy of the lead"""
     if self.frozen_lead: energy = 0.0 # set as zero energy
     delta = self.delta
     intra = self.lead.intra
     inter = dagger(self.lead.inter)
     cou = inter
     ggg,g = green_renormalization(intra,inter,
                                     energy=energy,
                                     delta=delta)
     sigma = cou@g@dagger(cou) # selfenergy
     return sigma

from ..htk.extract import local_hamiltonian

def local_selfenergy(h,g,energy=0.0,i=0,delta=1e-5,**kwargs):
    """Given a certain Hamiltonian and Green's function, extract
    the local selfenergy"""
    M = get_intra(h) # get intracell matrix
    gi = local_hamiltonian(h,g,i=i) # local Green's function
    oi = local_hamiltonian(h,M,i=i) # local Hamiltonian
    iden = np.identity(gi.shape[0],dtype=np.complex)
    out = algebra.inv(gi) - (energy+1j*delta)*iden + oi # local selfenergy
    return -out



def get_central_gmatrix(P,selfl=None,selfr=None,energy=0.0):
    """Return the central Green's function"""
    delta = P.delta # imaginary part
    if selfl is None: selfl = P.get_selfenergy(lead=0,energy=energy)
    if selfr is None: selfr = P.get_selfenergy(lead=1,energy=energy)
    if delta>delta_smatrix: delta = delta_smatrix # small delta is critical!
    iden = np.identity(selfl.shape[0],dtype=complex)*(energy +1j*delta)
    if P.frozen_lead:
        idenl = np.identity(selfl.shape[0],dtype=complex)*1j*delta
    else: idenl = iden
    hlist = [[None for i in range(2)] for j in range(2)] # list of matrices
    M = get_intra(P.H) # intracell matrix
    oi = local_hamiltonian(P.H,M,i=P.i) # local Hamiltonian
    # set up the different elements
    # first the intra terms
    hlist[0][0] = idenl - P.lead.intra - selfl
    hlist[1][1] = iden - oi - selfr
    # now the inter cell
    hlist[0][1] = -P.lead.inter*P.T # coupling times transparency
    hlist[1][0] = dagger(hlist[0][1]) # Hermitian conjugate
    return hlist


def get_reflection_normal_lead(P,s):
    return s[0][0]




def get_intra(H):
    """Function to extract the intra-matrix from an object
    depending on its type"""
    from ..hamiltonians import Hamiltonian
    from ..embedding import Embedding
    if type(H)==Hamiltonian: return H.intra
    elif type(H)==Embedding: return H.m
    else: raise



def Hamiltonian_didv(self,**kwargs):
   """Wrapper to compute the didv for a Hamiltonian"""
   lp = LocalProbe(self,**kwargs) # create the object
   return lp.didv(**kwargs)




