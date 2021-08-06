# specialized routine to perform an SCF, taking as starting point an
# attractive local interaction in a spinless Hamiltonian

from .. import inout
import numpy as np
import time
import os
from .. import densitymatrix
from copy import deepcopy
from numba import jit
from .. import utilities
from ..multihopping import MultiHopping
from .. import algebra

class Interaction():
    def __init__(self,h=None):
        self.dimensionality = 0
        if h is not None: self.dimensionality = h.dimensionality
        self.v_dict = dict() # store dictionary
    def __mult__(self,a):
        """Function to multiply"""
        out = 0
        for key in self: out = out + self[key]*a[key]
        return out



def normal_term(v,dm):
    """Return the normal term of the mean field"""
    out = dm*0.0 # initialize
    return normal_term_jit(v,dm,out) # return the normal term



def normal_term_ii(v,dm):
    """Return the normal term of the mean field"""
    out = dm*0.0 # initialize
    return normal_term_ii_jit(v,dm,out) # return the normal term


def normal_term_jj(v,dm):
    """Return the normal term of the mean field"""
    out = dm*0.0 # initialize
    return normal_term_jj_jit(v,dm,out) # return the normal term


def normal_term_ij(v,dm):
    """Return the normal term of the mean field"""
    out = dm*0.0 # initialize
    return normal_term_ij_jit(v,dm,out) # return the normal term


def normal_term_ji(v,dm):
    """Return the normal term of the mean field"""
    out = dm*0.0 # initialize
    return normal_term_ji_jit(v,dm,out) # return the normal term

@jit(nopython=True)
def normal_term_jit(v,dm,out):
    """Return the normal terms, jit function"""
    n = len(v[0])
    for i in range(n): # loop
      for j in range(n): # loop
        out[i,j] = out[i,j] - v[i,j]*dm[j,i]
        out[j,i] = out[j,i] - v[i,j]*dm[i,j]
        out[i,i] = out[i,i] + v[i,j]*dm[j,j]
        out[j,j] = out[j,j] + v[i,j]*dm[i,i]
    return out


@jit(nopython=True)
def normal_term_ii_jit(v,dm,out):
    """Return the normal terms, jit function"""
    n = len(v[0])
    for i in range(n): # loop
      for j in range(n): # loop
        out[i,i] = out[i,i] + v[i,j]*dm[j,j]
    return out

@jit(nopython=True)
def normal_term_jj_jit(v,dm,out):
    """Return the normal terms, jit function"""
    n = len(v[0])
    for i in range(n): # loop
      for j in range(n): # loop
        out[j,j] = out[j,j] + v[i,j]*dm[i,i]
    return out


@jit(nopython=True)
def normal_term_ij_jit(v,dm,out):
    """Return the normal terms, jit function"""
    n = len(v[0])
    for i in range(n): # loop
      for j in range(n): # loop
        out[i,j] = out[i,j] - v[i,j]*dm[j,i]
    return out


@jit(nopython=True)
def normal_term_ji_jit(v,dm,out):
    """Return the normal terms, jit function"""
    n = len(v[0])
    for i in range(n): # loop
      for j in range(n): # loop
        out[j,i] = out[j,i] - v[i,j]*dm[i,j]
    return out



def update_hamiltonian(tdict,mf):
    """Update the hoppings with the mean field"""
    return (MultiHopping(tdict) + MultiHopping(mf)).get_dict()
#    out = deepcopy(tdict) # copy
#    for key in mf:
#        if key in tdict: out[key] = tdict[key] + mf[key] # add contribution
#        else: out[key] = mf[key]
#    return out # return dictionary


def mix_mf(mf,mf0,mix=0.8):
    """Mix mean fields"""
    return ((1-mix)*MultiHopping(mf0) + mix*MultiHopping(mf)).get_dict()
#    out = dict() # initialize
#    for key in mf: # loop
#        if key not in mf0: out[key] = mf[key]
#        else:
#            #v0 = np.tanh(mf0[key]/mix)
#            #v1 = np.tanh(mf[key]/mix)
#            #out[key] = np.arctanh((v0+v1)/2.)*mix
#            out[key] = mf0[key]*(1.-mix) + mf[key]*mix # add contribution
#        #out[key] = mf0[key]*(1.-mix) + mf[key]*mix # add contribution
#    return out



def diff_mf(mf0,mf):
    """Difference mean fields"""
    out = 0.0 # initialize
    for key in mf: # loop
        if key not in mf0: out += np.mean(np.abs(mf[key]))
        else: out += np.mean(np.abs(mf0[key] - mf[key])) # add contribution
        #out += np.mean(np.abs(mf0[key] - mf[key])) # add contribution
    return out # return


def hamiltonian2dict(h):
    return h.get_dict() # return dictionary
#    out = dict() # create dictionary
#    if not h.is_multicell: raise
#    out[(0,0,0)] = h.intra
#    for t in h.hopping: out[tuple(t.dir)] = t.m # store
#    return out


def set_hoppings(h,hop):
    """Add the hoppings to the Hamiltonian"""
    h.set_multihopping(MultiHopping(hop))

def get_dm(h,v,nk=1):
    """Get the density matrix"""
    ds = [(0,0,0)] # directions
    if h.dimensionality>0:
        for key in v: ds.append(key) # store
    dms = h.get_density_matrix(ds=ds,nk=nk) # get all the density matrices
    return dms # return dictionary



def get_mf(v,dm,has_eh=False,compute_anomalous=True,
        compute_normal=True,**kwargs):
    """Get the mean field matrix"""
    if has_eh:
        # let us assume that it is a full Nambu spinor
        # (this may not be general, but good enough in the meantime)
        from .. import superconductivity
        dme = dict() # dictionary
        dma01 = dict() # dictionary
        dma10 = dict() # dictionary
        ns = v[(0,0,0)].shape[0]//2 # number of spinless sites
#        op = superconductivity.nambu_anomalous_reordering(ns)
#        op = op@op # comment this to go back to the previous version
        for key in dm: # extract the electron part 
#            m = op.T@dm[key]@op # transform to the new basis
            m = dm[key] # transform to the new basis
            dme[key] = superconductivity.get_eh_sector(m,i=0,j=0)
            # this is a workaround for the reordering of Nambu spinors
            dma10[key] = superconductivity.get_eh_sector(m,i=0,j=1)
        mfe = get_mf_normal(v,dme,**kwargs) # electron part of the mean field
        # anomalous part
        #dma01,dma10 = enforce_eh_symmetry_anomalous(dma01,dma10)
        mfa01 = get_mf_anomalous(v,dma10) 
        mfa01,mfa10 = enforce_eh_symmetry_anomalous(mfa01)
        ##############################################
        # now rebuild the Hamiltonian
        mf = dict()
        for key in v:
            if not compute_normal: mfe[key] = mfe[key]*0.0
            if compute_anomalous:
                m = superconductivity.build_nambu_matrix(mfe[key],
                    c12 = mfa10[key],c21=mfa01[key])
            else:
                m = superconductivity.build_nambu_matrix(mfe[key])
            mf[key] = m # store this matrix
    #        print(key)
    #        print(np.round(m,2))
            #print(np.unique(np.round(m,2)))
        if not MultiHopping(mf).is_hermitian(): # just a sanity check
            print("Non-Hermitian mean field")
            print(np.round(mf[(0,0,0)],2))
            exit()
   #     exit()
        # enforce electron-hole symmetry
   #     mf = superconductivity.enforce_eh_symmetry(mf)
        return mf # return mean field matrix
    else: return get_mf_normal(v,dm,**kwargs) # no BdG Hamiltonian




def get_mf_normal(v,dm,compute_dd=True,add_dagger=True,
        compute_cross=True):
    """Get the mean field"""
    zero = dm[(0,0,0)]*0. # zero
    mf = dict()
    for d in v: mf[d] = zero.copy()  # initialize
    # compute the contribution to the mean field
    # onsite term
#    mf[(0,0,0)] = normal_term(v[(0,0,0)],dm[(0,0,0)]) 
    def dag(m): return m.T.conjugate()
    for d in v: # loop over directions
        d2 = (-d[0],-d[1],-d[2]) # minus this direction
        # add the normal terms
        if compute_cross: # only density density terms
            m = normal_term_ij(v[d],dm[d2]) # get matrix
            mf[d] = mf[d] + m # add normal term
            if add_dagger:
                mf[d2] = mf[d2] + dag(m) # add normal term
        if compute_dd: # density density terms
            m = normal_term_ii(v[d],dm[(0,0,0)]) # get matrix
            mf[(0,0,0)] = mf[(0,0,0)] + m # add normal term
            m = normal_term_jj(v[d2],dm[(0,0,0)]) # get matrix
            mf[(0,0,0)] = mf[(0,0,0)] + m # add normal term
    return mf



# anomalous part of the mean field
from .superscf import get_mf_anomalous
from .superscf import enforce_eh_symmetry_anomalous






def get_dc_energy(v,dm):
    """Compute double counting energy"""
    out = 0.0
    for d in v: # loop over interactions
        d2 = (-d[0],-d[1],-d[2]) # minus this direction
        n = v[d].shape[0] # shape
        for i in range(n): # loop
          for j in range(n): # loop
              out -= v[d][i,j]*dm[(0,0,0)][i,i]*dm[(0,0,0)][j,j]
              c = dm[d][i,j] # cross term
              out += v[d][i,j]*c*np.conjugate(c) # add contribution
#    print("DC energy",out.real)
    return out.real



from .mfconstrains import obj2mf

mf_file = "MF.pkl" 

def generic_densitydensity(h0,mf=None,mix=0.1,v=None,nk=8,solver="plain",
        maxerror=1e-5,filling=None,callback_mf=None,callback_dm=None,
        load_mf=True,compute_cross=True,compute_dd=True,verbose=1,
        compute_anomalous=True,compute_normal=True,info=False,
        callback_h=None,**kwargs):
    """Perform the SCF mean field"""
    if verbose>1: info=True
#    if not h0.check_mode("spinless"): raise # sanity check
    h1 = h0.copy() # initial Hamiltonian
    h1.turn_dense()
    h1.nk = nk # store the number of kpoints
    if mf is None: # no mean field given
      try: 
          if load_mf: 
              mf = inout.load(mf_file) # load the file
              MultiHopping(h0.get_dict()) + MultiHopping(mf) # see if compatible
          else: raise
      except: 
          mf = dict()
          for d in v: mf[d] = np.exp(1j*np.random.random(h1.intra.shape))
          mf[(0,0,0)] = mf[(0,0,0)] + mf[(0,0,0)].T.conjugate()
    elif type(mf)==str:
        from ..meanfield import guess
        mf = guess(h0,mode=mf) # overwrite
    else: pass # assume that it is a valid mf
    mf = obj2mf(mf) # convert to MF
    ii = 0
    os.system("rm -f STOP") # remove stop file
    hop0 = hamiltonian2dict(h1) # create dictionary
    def f(mf,h=h1):
      """Function to minimize"""
#      print("Iteration #",ii) # Iteration
      mf0 = deepcopy(mf) # copy
      h = h1.copy()
      hop = update_hamiltonian(hop0,mf) # add the mean field to the Hamiltonian
      set_hoppings(h,hop) # set the new hoppings in the Hamiltonian
      if callback_h is not None:
          h = callback_h(h) # callback for the Hamiltonian
      t0 = time.perf_counter() # time
      dm = get_dm(h,v,nk=nk) # get the density matrix
      if callback_dm is not None:
          dm = callback_dm(dm) # callback for the density matrix
      t1 = time.perf_counter() # time
      # return the mean field
      mf = get_mf(v,dm,compute_cross=compute_cross,compute_dd=compute_dd,
              has_eh=h0.has_eh,compute_anomalous=compute_anomalous,
              compute_normal=compute_normal) 
      if callback_mf is not None:
          mf = callback_mf(mf) # callback for the mean field
      t2 = time.perf_counter() # time
      if verbose>1: print("Time in density matrix = ",t1-t0) # Difference
      if verbose>1: print("Time in the normal term = ",t2-t1) # Difference
      scf = SCF() # create object
      scf.hamiltonian = h # store
#      h.check() # check the Hamiltonian
      scf.hamiltonian0 = h0 # store
      scf.mf = mf # store mean field
      if os.path.exists("STOP"): scf.mf = mf0 # use the guess
      scf.dm = dm # store density matrix
      scf.v = v # store interaction
      scf.tol = maxerror # maximum error
      return scf
    if solver=="plain":
      do_scf = True
#      from .mixing import Mixing
#      Mxg = Mixing() # initialize
      while do_scf:
        scf = f(mf) # new vector
        mfnew = scf.mf # new vector
        t0 = time.perf_counter() # time
        diff = diff_mf(mfnew,mf) # mix mean field
#        mix = Mxg.get_mix(diff) # add error
#        print("Mixing",mix)
        mf = mix_mf(mfnew,mf,mix=mix) # mix mean field
        if callback_mf is not None: # redefine mean-field if necessary
            mf = callback_mf(mf) # callback for the mean field
        t1 = time.perf_counter() # time
        if verbose>1: print("Time in mixing",t1-t0)
        if verbose>0: print("ERROR in the SCF cycle",diff)
        #print("Mixing",dmix)
        if diff<maxerror: 
            scf = f(mfnew) # last iteration, with the unmixed mean field
            inout.save(scf.mf,mf_file) # save the mean field
         #   scf.hamiltonian.check(tol=100*maxerror) # perform some sanity checks
            return scf
    else: # use different solvers
        scf = f(mf) # perform one iteration
        fmf2a = get_mf2array(scf) # convert MF to array
        fa2mf = get_array2mf(scf) # convert array to MF
        def fsol(x): # define the function to solve
            mf1 = fa2mf(x) # convert to a MF
            scf1 = f(mf1) # compute function
            xn = fmf2a(scf1.mf) # new vector
            diff = x - xn # difference vector
            print("ERROR",np.max(np.abs(diff)))
            print()
            return x - xn # return vector
        x0 = fmf2a(scf.mf) # initial guess
        # these methods do seem too efficient, but lets have them anyway
        if solver=="krylov":
            from scipy.optimize import newton_krylov
            x = newton_krylov(fsol,x0,rdiff=1e-3) # use the solver
        elif solver=="anderson":
            from scipy.optimize import anderson
            x = anderson(fsol,x0) # use the solver
        elif solver=="broyden1":
            from scipy.optimize import broyden1
            x = broyden1(fsol,x0,f_tol=maxerror*100) # use the solver
        elif solver=="linear":
            from scipy.optimize import linearmixing
            x = linearmixing(fsol,x0,f_tol=maxerror*100) # use the solver
        else: raise # unrecognised solver
        mf = fa2mf(x) # transform to MF
        scf = f(mf) # compute the SCF with the solution
        scf.error = maxerror # store the error
        inout.save(scf.mf,mf_file) # save the mean field
        return scf # return the mean field


def get_mf2array(scf):
    """Function to transform the mean field in an array"""
    nt = len(scf.mf) # number of terms in the dictionary
    n = scf.mf[(0,0,0)].shape[0]
    def fmf2a(mf):
        #print(mf[(0,0,0)].real)
        out = [mf[key].real for key in mf] # to plain array
        out += [mf[key].imag for key in mf] # to plain array
        out = np.array(out)
#        print(out.shape)
        out = out.reshape(nt*n*n*2) # reshape
        return out
    return fmf2a # return function

def get_array2mf(scf):
    """Function to transform an array into a mean field"""
    ds = [key for key in scf.mf] # store keys
    nt = len(scf.mf) # number of terms in the dictionary
    n = scf.mf[(0,0,0)].shape[0] # size
    def fa2mf(a):
        a = a.copy().reshape((2*nt,n*n)) # reshape array
        mf =  dict()
        #print(a.shape)
        for i in range(len(ds)):
            d = ds[i]
            m = a[i,:] + 1j*a[i+nt,:] # get matrix
            mf[d] = m.reshape((n,n)) # store
        return mf
    return fa2mf # return function


def densitydensity(h,filling=0.5,mu=None,verbose=0,**kwargs):
    """Function for density-density interactions"""
    if h.has_eh: 
        if not h.has_spin: return NotImplemented # only for spinful
    h = h.get_multicell()
    h.turn_dense()
    def callback_h(h):
        """Set the filling"""
        if mu is None:
          fermi = h.get_fermi4filling(filling,nk=h.nk) # get the filling
          if verbose>1: print("Fermi energy",fermi)
          h.fermi = fermi
          h.shift_fermi(-fermi) # shift by the fermi energy
        else: h.shift_fermi(-mu) # shift by mu
        return h
#    callback_h = None
    scf = generic_densitydensity(h,callback_h=callback_h,verbose=verbose,
            **kwargs)
    # Now compute the total energy
    h = scf.hamiltonian
    etot = h.get_total_energy(nk=h.nk)
    if mu is None: 
        etot += h.fermi*h.intra.shape[0]*filling # add the Fermi energy
    #print("Occupied energies",etot)
    etot += get_dc_energy(scf.v,scf.dm) # add the double counting energy
    etot = etot.real
    scf.total_energy = etot
    if verbose>1:
      print("##################")
      print("Total energy",etot)
      print("##################")
    return scf




def hubbard(h,U=1.0,constrains=[],**kwargs):
    """Wrapper to perform a Hubbard model calculation"""
    h = h.copy() # copy Hamiltonian
    h.turn_multicell() # multicell Hamiltonian
    U = obj2geometryarray(U,h.geometry) # redefine as array 
    n = len(h.geometry.r) # number of spinless sites
    if h.has_spin:
      zero = np.zeros((2*n,2*n),dtype=np.complex)
      for i in range(n): zero[2*i,2*i+1] = U[i] # Hubbard interaction
    else: 
      zero = np.zeros((n,n),dtype=np.complex)
      n = len(h.geometry.r) # number of spinless sites
      for i in range(n): zero[i,i] = U[i] # Hubbard interaction
    v = dict() # dictionary
    v[(0,0,0)] = zero 
    from . import mfconstrains
    def callback_mf(mf):
        """Put the constrains in the mean field if necessary"""
        mf = mfconstrains.enforce_constrains(mf,h,constrains)
        return mf
    if h.has_spin:
      return densitydensity(h,v=v,callback_mf=callback_mf,**kwargs)
    else:
      return densitydensity(h,v=v,compute_cross=False,
              callback_mf=callback_mf,**kwargs)


def Vinteraction(h,V1=0.0,V2=0.0,V3=0.0,U=0.0,
        constrains=[],Vr=None,**kwargs):
    """Perform a mean-field calculation with density-density interactions
    - U, local Hubbard interaction
    - V1, first neighbor interaction
    - V2, second neighbor interaction
    """
    h = h.get_multicell() # multicell Hamiltonian
    h.turn_dense()
    # define the function
    nd = h.geometry.neighbor_distances() # distance to first neighbors
    from .. import specialhopping
    mgenerator = specialhopping.distance_hopping_matrix([V1/2.,V2/2.,V3/2.],nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False,is_multicell=True,
            mgenerator=mgenerator) 
    if Vr is not None:
      hv1 = h.geometry.get_hamiltonian(has_spin=False,is_multicell=True,
              fun=Vr)
      hv = hv + hv1 # add the two Hamiltonians
    v = hv.get_hopping_dict() # hopping dictionary
    U = obj2geometryarray(U,h.geometry) # convert to array
    if h.has_spin: #raise # not implemented
        for d in v: # loop
            m = v[d] ; n = m.shape[0]
    #        print(m)
            m1 = np.zeros((2*n,2*n),dtype=np.complex)
            for i in range(n):
              for j in range(n): 
                  m1[2*i,2*j] = m[i,j]
                  m1[2*i+1,2*j] = m[i,j]
                  m1[2*i,2*j+1] = m[i,j]
                  m1[2*i+1,2*j+1] = m[i,j]
            v[d] = m1 # store
        for i in range(n):
            v[(0,0,0)][2*i,2*i+1] += U[i]/2. # add
            v[(0,0,0)][2*i+1,2*i] += U[i]/2. # add
    # Now put the constrains if necessary
    from . import mfconstrains
    def callback_mf(mf):
        """Put the constrains in the mean field if necessary"""
        mf = mfconstrains.enforce_constrains(mf,h,constrains)
        return mf
    return densitydensity(h,v=v,callback_mf=callback_mf,**kwargs)



from ..meanfield import identify_symmetry_breaking

class SCF():
    def identify_symmetry_breaking(self,**kwargs):
        return identify_symmetry_breaking(self.hamiltonian,self.hamiltonian0,
                tol=10*self.tol,**kwargs)
    def order_parameter(self,name):
        from ..meanfield import order_parameter
        return order_parameter(self,name)





def obj2geometryarray(U,g):
    """Convert an object to an array"""
    if algebra.isnumber(U):
        return np.array([U for ir in g.r]) # same for all
    elif callable(U):
        return np.array([U(ir) for ir in g.r]) # call for each
    else: raise
