import numpy as np
from .. import filewrite

from .. import algebra

# define an embedded object based on a Hamiltonian

class Embedded_Hamiltonian():
    def __init__(self,H,delta=1e-4,selfenergy=None):
        self.H = H.copy() # copy Hamiltonian
        self.selfenergy = object2selfenergy(selfenergy,H)
        self.delta = delta
    def get_density_matrix(self,delta=None,**kwargs):
        """Return the density matrix, in the index convention of
        Hamiltonian.get_density_matrix (see get_dm). It needs a selfenergy
        analytic in the upper half plane, see get_dm"""
        if delta is None: delta = self.delta # default broadening
        return get_dm(self,delta=delta,**kwargs)
    def get_gf(self,**kwargs):
        # Green's function of the Hamiltonian
        gf0 = self.H.get_gf(**kwargs) 
        selfe = self.selfenergy(**kwargs) # store selfenergy
        gf = algebra.inv(algebra.inv(gf0) - selfe) # full Green's function
        return gf # return full Green's function
    def set_multihopping(self,*args):
        self.H.set_multihopping(*args)
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def get_ldos(self,write=None,**kwargs):
        """LDOS of the embedded cell, returned as (positions, LDOS) and
        written to LDOS.OUT unless write=False"""
        write = filewrite.resolve(write,True) # the call, else the global switch
        A = get_A(self,**kwargs) # spectral function
        r = np.diag(A).real
        r = self.H.full2profile(r) # resum components
        if write: self.H.geometry.write_profile(r,name="LDOS.OUT")
        return self.H.geometry.r,r # return LDOS
    def get_kdos(self,**kwargs): return get_kdos(self,**kwargs)



def _dagger(s):
    """Hermitian conjugate of a matrix, sparse or dense, or of a number"""
    if np.isscalar(s): return np.conjugate(s)
    return s.conj().T


def object2selfenergy(self,H,delta=1e-4,**kwargs):
    """The selfenergy as a function of energy= and delta=. A matrix or a
    callable is the retarded selfenergy, and for delta<0 the advanced one
    is returned, its continuation Sigma(z) = Sigma(z^*)^dagger, the same
    contract as the boundary embedding (embeddingtk/boundaryembedding.py):
    the Hermitian conjugate, not the element-wise one, whatever the
    symmetries of the Hamiltonian. A callable is evaluated at complex
    energies by the density matrix, so it has to be analytic in the upper
    half plane, as a causal selfenergy is; one that is not carries
    analytic=False (embeddingtk.selfenergies.semimetal) and get_dm refuses
    it (bug_audit_6.md #13)"""
    if self is None: # no selfenergy provided
        return lambda **kwargs: 0.
    elif algebra.ismatrix(self): # matrix provided
      if H.intra.shape[0]==self.shape[0]:
        def f(delta=delta,**kwargs):
            if delta<0.: return _dagger(self) # advanced selfenergy
            return self # retarded selfenergy
        return f
      else: # unrecognized
          raise ValueError("the selfenergy matrix does not have the same "
                  "dimension as the Hamiltonian")
    elif callable(self): # the retarded selfenergy as a function of energy
        def f(energy=0.,delta=delta,**kwargs):
            if delta<0.: # advanced, the continuation below the real axis
                return _dagger(self(energy=np.conjugate(energy),
                    delta=-delta,**kwargs))
            return self(energy=energy,delta=delta,**kwargs)
        f.analytic = getattr(self,"analytic",True)
        return f
    else: 
        raise TypeError("the selfenergy must be a matrix of the same "
                "dimension as the Hamiltonian, or a callable returning one")


def embed_hamiltonian(self,**kwargs):
   EB = Embedded_Hamiltonian(self,**kwargs)
   return EB



def get_dm(self,delta=1e-2,emin=-10.,eps=1e-4,**kwargs):
    """Get the density matrix

    It is returned in the index convention of densitymatrix.full_dm,
    dm[i,j] = sum_occ conj(psi_i) psi_j, the transpose of the usual rho,
    so that sum(dm*A) = Tr(dm.T@A) gives <A> here as it does for
    Hamiltonian.get_density_matrix. The resolvent integral below gives
    rho itself, hence the final transpose.

    eps: absolute tolerance of the adaptive contour integral, kept apart
        from the keywords that go to get_gf

    The integral runs on a contour of complex energies, which needs a
    selfenergy analytic in the upper half plane. A constant matrix is, and
    so is a causal callable; one marked analytic=False, such as the
    semimetal selfenergy -i Gamma |E|, gave entries off by 0.3 against the
    real-axis integral, and is refused (bug_audit_6.md #13). This holds for
    Embedded_Hamiltonian and for the boundary embedding of Embedding.
    """
    if getattr(getattr(self,"selfenergy",None),"analytic",True) is False:
        raise NotImplementedError("the density matrix of an embedded "
                "system is a contour integral over complex energies, which "
                "needs a selfenergy analytic in the upper half plane, and "
                "this one is marked analytic=False (the semimetal "
                "selfenergy -i Gamma |E| has no analytic continuation). "
                "Use get_ldos and get_gf, which work at real energies, or "
                "a selfenergy analytic in the upper half plane")
    fa = lambda e: self.get_gf(energy=e,delta=delta,**kwargs) # advanced
    fr = lambda e: self.get_gf(energy=e,delta=-delta,**kwargs) # retarded
    from ..integration import complex128contour
    Ra = complex128contour(fa,xmin=emin,xmax=0.,eps=eps,mode="upper") # return the integral
    Rr = complex128contour(fr,xmin=emin,xmax=0.,eps=eps,mode="lower") # return the integral
    rho = 1j*(Ra-Rr)/(2.*np.pi) # usual density matrix
    return np.transpose(rho) # full_dm convention



def get_A(self,delta=1e-3,**kwargs):
    Ra = self.get_gf(delta=delta,**kwargs) # advanced
    Rr = self.get_gf(delta=-delta,**kwargs) # retarded
    return 1j*(Ra-Rr)/(2.*np.pi) # return the spectral fucntion



def get_kdos(self,energies=None,kpath=None,write=None,**kwargs):
    """Compute k-resolved DOS. Returns the k index, the energy and the
    k-resolved DOS as three columns, and writes them to KDOS.OUT unless
    write=False"""
    write = filewrite.resolve(write,True) # the call, else the global switch
    def f(e,k): # function to evaluate
        gf0 = self.H.get_gk_gen(**kwargs)(e=e,k=k) # get Green's function
        selfe = self.selfenergy(energy=e,**kwargs) # selfenergy
        gf = algebra.inv(algebra.inv(gf0) - selfe) # full Green's function
        return -np.trace(gf).imag # return full Green's function
    if energies is None: energies = np.linspace(-1.0,1.0,100)
    kpath = self.H.geometry.get_kpath(kpath=kpath,write=write) # get the kpath
    out = [] # rows of k index, energy, kdos
    for ik in range(len(kpath)):
        print("Doing",ik)
        for ie in energies:
            out.append([ik,ie,f(ie,kpath[ik])])
    out = np.array(out).T # three columns
    if write:
        fo = open("KDOS.OUT","w")
        for (ik,ie,d) in out.T:
            fo.write(str(int(ik))+" ")
            fo.write(str(ie)+" ")
            fo.write(str(d)+"\n")
        fo.close()
    return out[0],out[1],out[2]







