import numpy as np

from .. import algebra

# define an embedded object based on a Hamiltonian

class Embedded_Hamiltonian():
    def __init__(self,H,delta=1e-4,selfenergy=None):
        self.H = H.copy() # copy Hamiltonian
        self.selfenergy = object2selfenergy(selfenergy,H)
        self.delta = delta
    def get_density_matrix(self,**kwargs): 
        return get_dm(self,delta=self.delta,**kwargs)
    def get_gf(self,**kwargs):
        # Green's function of the Hamiltonian
        gf0 = self.H.get_gf(**kwargs) 
        selfe = self.selfenergy(**kwargs) # store selfenergy
        gf = algebra.inv(algebra.inv(gf0) - selfe) # full Green's function
        return gf # return full Green's function
    def set_multihopping(self,*args): 
        self.H.set_multihopping(*args)
    def get_mean_field_hamiltonian(self,**kwargs):
        from ..selfconsistency.embedding import hubbard_mf
        return hubbard_mf(self,**kwargs) # return Hubbard mean-field
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def get_ldos(self,**kwargs): 
        A = get_A(self,**kwargs) # spectral function
        r = [A[i,i].real for i in range(A.shape[0])]
        r = self.H.full2profile(r) # resum components
        self.H.geometry.write_profile(r,name="LDOS.OUT")
    def get_kdos(self,**kwargs): return get_kdos(self,**kwargs)



def object2selfenergy(self,H,delta=1e-4,**kwargs):
    if self is None: # no selfenergy provided 
        return lambda **kwargs: 0.
    elif algebra.ismatrix(self): # matrix provided
      if H.intra.shape[0]==self.shape[0]: 
        def f(delta=delta,**kwargs):
            if delta>0.: return self
            else: return np.conjugate(self)
        return f
      else: raise # unrecognized
    elif callable(self): return self # assume that it returns a matrix
    else: 
        print("Selfenergy is not compatible with Hamiltonian")
        raise


def embed_hamiltonian(self,**kwargs):
   EB = Embedded_Hamiltonian(self,**kwargs)
   return EB



def get_dm(self,delta=1e-2,emin=-10.,**kwargs):
    """Get the density matrix"""
    fa = lambda e: self.get_gf(energy=e,delta=delta,**kwargs) # advanced
    fr = lambda e: self.get_gf(energy=e,delta=-delta,**kwargs) # retarded
    from ..integration import complex_contour
    Ra = complex_contour(fa,xmin=emin,xmax=0.,mode="upper") # return the integral
    Rr = complex_contour(fr,xmin=emin,xmax=0.,mode="lower") # return the integral
    return 1j*(Ra-Rr)/(2.*np.pi) # return the density matrix



def get_A(self,delta=1e-3,**kwargs):
    Ra = self.get_gf(delta=delta,**kwargs) # advanced
    Rr = self.get_gf(delta=-delta,**kwargs) # retarded
    return 1j*(Ra-Rr)/(2.*np.pi) # return the spectral fucntion



def get_kdos(self,energies=None,kpath=None,**kwargs):
    """Compute k-resolved DOS"""
    def f(e,k): # function to evaluate
        gf0 = self.H.get_gk_gen(**kwargs)(e=e,k=k) # get Green's function
        selfe = self.selfenergy(energy=e,**kwargs) # selfenergy
        gf = algebra.inv(algebra.inv(gf0) - selfe) # full Green's function
        return -np.trace(gf).imag # return full Green's function
    if energies is None: energies = np.linspace(-1.0,1.0,100)
    kpath = self.H.geometry.get_kpath(kpath=kpath) # get the kpath
    fo = open("KDOS.OUT","w")
    for ik in range(len(kpath)):
        print("Doing",ik)
        for ie in energies:
            d = f(ie,kpath[ik])
            fo.write(str(ik)+" ")
            fo.write(str(ie)+" ")
            fo.write(str(d)+"\n")
    fo.close()



