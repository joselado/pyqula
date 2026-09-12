from __future__ import print_function
from scipy.sparse import csc_matrix,bmat
import numpy as np
from .multihopping import MultiHopping
from. import superconductivity
#from .scftypes import selfconsistency

dup = csc_matrix(([1.0],[[0],[0]]),shape=(2,2),dtype=np.complex128)
ddn = csc_matrix(([1.0],[[1],[1]]),shape=(2,2),dtype=np.complex128)
sp = csc_matrix(([1.0],[[1],[0]]),shape=(2,2),dtype=np.complex128)
sm = csc_matrix(([1.0],[[0],[1]]),shape=(2,2),dtype=np.complex128)
def zero(d): return csc_matrix(([],[[],[]]),shape=(d,d),dtype=np.complex128)


def element(i,n,p,d=2,j=None):
  if j is None: j=i
  o = csc_matrix(([1.0],[[d*i+p[0]],[d*j+p[1]]]),
                   shape=(n*d,n*d),dtype=np.complex128)  
  return o
  o = csc_matrix(([1.0],[[p[0]],[p[1]]]),shape=(d,d),dtype=np.complex128)
  m = [[None for i1 in range(n)] for i2 in range(n)]
  for i1 in range(n): m[i1][i1] = zero(d)
  m[i][j] = o.copy()
  return bmat(m) # return matrix



class interaction(): 
  def __init__(self):
    self.contribution = "AB" # assume that both operators contribute
    self.dir = [0,0,0] # unit cell to which to hop
    self.dhop = [0,0,0] # matrix that is affected

def hubbard_density(i,n,g=1.0):
  """Return pair of operators for a Hubbard mean field"""
  v = interaction()
  v.a = element(i,n,[0,0]) # value of the coupling
  v.b = element(i,n,[1,1]) # value of the coupling
  v.dir = [0,0,0] # direction of the interaction 
  v.g = g
  v.i = i
  v.j = i
  return v



def spinless_hubbard_density(i,n,g=1.0):
  """Return pair of operators for a Hubbard mean field"""
  v = interaction()
  v.a = element(i,n,[0,0],d=1) # value of the coupling
  v.b = element(i,n,[0,0],d=1) # value of the coupling
  v.dir = [0,0,0] # direction of the interaction 
  v.g = g
  v.i = i
  v.j = i
  return v







def hubbard_exchange(i,n,g=1.0):
  """Return pair of operators for a Hubbard mean field"""
  v = interaction()
  v.a = element(i,n,[0,1]) # value of the coupling
  v.b = element(i,n,[1,0]) # value of the coupling
  v.dir = [0,0,0] # direction of the interaction 
  v.g = -g # minus from fermion
  v.i = i
  v.j = i
  return v




def hubbard_pairing_ud(i,n,g=1.0):
  """Return pair of operators for a Hubbard mean field"""
  v = interaction()
  v.a = element(i,n,[0,2],d=4) # cc
  v.b = element(i,n,[2,0],d=4) # cdcd
  v.dir = [0,0,0] # direction of the interaction 
  v.g = g
  v.i = i
  v.j = i
  return v


def hubbard_pairing_du(i,n,g=1.0):
  """Return pair of operators for a Hubbard mean field"""
  v = interaction()
  v.a = element(i,n,[1,3],d=4) # cc
  v.b = element(i,n,[3,1],d=4) # cdcd
  v.dir = [0,0,0] # direction of the interaction 
  v.g = g
  v.i = i
  v.j = i
  return v




def v_pairing_uu(i,j,n,g=1.0,d=[0,0,0],channel="ee"):
  """Return pair of operators for a Hubbard mean field"""
  v = interaction()
  if channel=="ee": # ee channel
    v.a = element(i,n,[0,3],d=4,j=j) # cc
    v.b = element(j,n,[3,0],d=4,j=i) # cdcd
  elif channel=="hh":
    v.a = element(i,n,[3,0],d=4,j=j) # cc
    v.b = element(j,n,[0,3],d=4,j=i) # cdcd
  else:
      raise ValueError("unknown channel; the accepted ones are 'ee' and 'hh'")
  v.dir = d # direction of the interaction 
  v.g = g
  v.contribution = "A"
  v.i = i
  v.j = j
  return v


def v_pairing_dd(i,j,n,g=1.0,d=[0,0,0],channel="ee"):
  """Return pair of operators for a V mean field"""
  v = interaction()
  if channel=="ee": # ee channel
    v.a = element(i,n,[1,2],d=4,j=j) # cc
    v.b = element(j,n,[2,1],d=4,j=i) # cdcd
  elif channel=="hh":
    v.a = element(i,n,[2,1],d=4,j=j) # cc
    v.b = element(j,n,[1,2],d=4,j=i) # cdcd
  else:
      raise ValueError("unknown channel; the accepted ones are 'ee' and 'hh'")
  v.dir = d # direction of the interaction 
  v.g = g
  v.contribution = "A"
  v.i = i
  v.j = j
  return v


def v_pairing_du(i,j,n,g=1.0,d=[0,0,0]):
  """Return pair of operators for a V mean field"""
  v = interaction()
  v.a = element(i,n,[1,3],d=4,j=j) # cc
  v.b = element(j,n,[3,1],d=4,j=i) # cdcd
  v.dir = d # direction of the interaction 
  v.g = g
  v.i = i
  v.j = j
  return v




def v_ij(i,j,n,g=1.0,d=[0,0,0],spini=0,spinj=0):
  """Return pair of operators for a V mean field"""
  v = interaction()
  v.a = element(i,n,[spini,spini],d=2,j=j) # cc
  v.b = element(j,n,[spinj,spinj],d=2,j=i) # cdc
  v.dir = d # direction of the interaction 
  v.g = -g # this minus comes from commutation relations
  v.contribution = "A"
  v.i = i
  v.j = j
  return v


def v_ij_spinless(i,j,n,g=1.0,d=[0,0,0]):
  """Return pair of operators for a V mean field"""
  v = interaction()
  v.a = csc_matrix(([1.0],[[i],[j]]),shape=(n,n),dtype=np.complex128) # cc
  v.b = csc_matrix(([1.0],[[j],[i]]),shape=(n,n),dtype=np.complex128) # cdc
  v.dir = d # direction of the interaction 
  v.dhop = d # direction of the interaction 
  v.g = -g # this minus comes from commutation relations
  v.contribution = "A"
  v.i = i
  v.j = j
  return v


def v_ij_density_spinless(i,j,n,g=1.0,d=[0,0,0],contribution="AB"):
  """Return pair of operators for a V mean field"""
  v = interaction()
  v.a = csc_matrix(([1.0],[[i],[i]]),shape=(n,n),dtype=np.complex128) # cc
  v.b = csc_matrix(([1.0],[[j],[j]]),shape=(n,n),dtype=np.complex128) # cdc
  v.dir = d # direction of the neighbor
  v.dhop = [0,0,0] # hopping that is affected
  v.g = g 
  v.contribution = contribution
  v.i = i
  v.j = j
  return v


def v_ij_fast_coulomb(i,jvs,n,vcut=1e-3):
  """Return the interaction part for the fast Coulomb trick"""
  v = interaction()
  v.a = csc_matrix(([1.0],[[i],[i]]),shape=(n,n),dtype=np.complex128) # cc
  jj = range(n) # indexes
  if len(jvs)!=n: # something wrong
      raise ValueError("the fast-Coulomb interaction needs one potential "
              "value per site")
  v.b = csc_matrix((jvs,[jj,jj]),shape=(n,n),dtype=np.complex128) # cdc
  v.b.eliminate_zeros()
  v.dir = [0,0,0] # direction of the neighbor
  v.dhop = [0,0,0] # hopping that is affected
  v.g = 1.0 # default value
  v.contribution = "A"
  return v



def v_ij_fast_coulomb_spinful(i,jvs,n,channel="up"):
  """Return the interaction part for the fast Coulomb trick"""
  jvs2 = np.zeros(2*n) # initialize
  if channel=="up":
      for jj in range(n): jvs2[2*jj] = jvs[jj]
      ii = 2*i+1
  elif channel=="down":
      for jj in range(n): jvs2[2*jj+1] = jvs[jj]
      ii = 2*i
  else:
      raise ValueError("unknown channel; the accepted ones are 'up' and "
              "'down'")
  return v_ij_fast_coulomb(ii,jvs2,2*n)



symmetry_breaking = ["magnetic","antiferro","ferroX","ferroY",
        "ferroZ","Charge density wave","Haldane",
        "kanemele","rashba","s-wave superconductivity"]

spinful_guesses = ["Fully random","dimerization"]

spinful_guesses += symmetry_breaking


def require_nambu(h,mode):
  """Complain if a superconducting guess is asked of a Hamiltonian without
  the electron-hole (Nambu) degree of freedom"""
  from .check import require_nambu as _require
  _require(h,"mean-field guess mode '"+str(mode)+"' is a superconducting "
    +"order parameter, so it")


def require_spin(h,mode):
  """Complain if a spinful guess is asked of a spinless Hamiltonian"""
  from .check import require_spin as _require
  _require(h,"mean-field guess mode '"+str(mode)+"'")


def require_sublattice(h,mode):
  """Complain if a sublattice-modulated guess is asked of a geometry
  that carries no sublattice label"""
  from .check import require_sublattice as _require
  _require(h,"mean-field guess mode '"+str(mode)+"' seeds the order with "
    +"the sublattice, so it")


# The mean-field guesses live in a registry (mode -> builder) rather than in
# an if/elif chain, so known_guesses below is derived from the dispatch
# instead of being a second list kept in sync by hand, and adding a guess is
# one entry. Every builder takes the Hamiltonian, a cleaned multicell copy of
# it, the amplitude and the mode name, and returns the guess: either the
# intracell matrix or a hopping dictionary, whichever the channel needs.


def _guess_ferro(h,h0,fun,mode):
    require_spin(h,mode) ; h0.add_zeeman(fun) ; return h0.intra

def _guess_magnetic(h,h0,fun,mode):
    require_spin(h,mode)
    h0.add_zeeman(lambda x: np.random.random(3)*fun)
    return h0.intra

def _guess_ferro_axis(axis):
    """Builder for a ferromagnetic guess along one Cartesian axis"""
    def f(h,h0,fun,mode):
        m = [0.,0.,0.] ; m[axis] = fun
        require_spin(h,mode) ; h0.add_zeeman(m) ; return h0.intra
    return f

def _guess_randomXY(h,h0,fun,mode):
    def f(r):
        m = [np.random.random()-0.5,np.random.random()-0.5,0.]
        m = np.array(m)
        return m/np.sqrt(m.dot(m))
    require_spin(h,mode) ; h0.add_zeeman(f)
    return h0.get_hopping_dict()

def _guess_random(h,h0,fun,mode):
    """A fully random Hermitian guess, built from h itself rather than
    from the cleaned copy: every hopping of the dictionary is replaced"""
    dd = h.get_dict()
    for key in dd:
        n = dd[key].shape[0]
        dd[key] = np.random.random((n,n))-.5 + 1j*(np.random.random((n,n))-.5)
    dd = MultiHopping(dd)
    dd = dd + dd.get_dagger()
    return dd.get_dict()

def _guess_dimerization(h,h0,fun,mode):
    return guess(h,mode="random",fun=fun)

def _guess_kekule(h,h0,fun,mode):
    h0.turn_multicell()
    h0.add_kekule(fun)
    return h0.get_hopping_dict()

def _guess_haldane(h,h0,fun,mode):
    h0.add_haldane(fun) ; return h0.get_hopping_dict()

def _guess_rashba(h,h0,fun,mode):
    require_spin(h,mode) ; h0.add_rashba(fun)
    return h0.get_hopping_dict()

def _guess_kanemele(h,h0,fun,mode):
    require_spin(h,mode) ; h0.add_kane_mele(fun)
    return h0.get_hopping_dict()

def _guess_antihaldane(h,h0,fun,mode):
    """Built from a fresh copy of h rather than from h0, which has already
    been turned multicell -- add_antihaldane wants the original form"""
    h = h.copy() ; h.clean() ; h.add_antihaldane(fun)
    return h.get_hopping_dict()

def _guess_CDW(h,h0,fun,mode):
    require_sublattice(h,mode)
    h0.add_onsite(h.geometry.sublattice) ; return h0.intra

def _guess_potential(h,h0,fun,mode):
    h0.add_onsite(fun) ; return h0.intra

def _guess_antiferro(h,h0,fun,mode):
    require_spin(h,mode) ; h0.add_antiferromagnetism(fun) ; return h0.intra

def _guess_imbalance(h,h0,fun,mode):
    h0.add_sublattice_imbalance(fun) ; return h0.intra

def _guess_swave(h,h0,fun,mode):
    require_nambu(h,mode) ; h0.add_swave(fun) ; return h0.intra

def _guess_pwave(h,h0,fun,mode):
    require_nambu(h,mode)
    for t in h0.hopping: t.m *= 0. # clean
    h0.add_pairing(delta=fun,mode="pwave") # px+ipy pairing
    hop = dict()
    hop[(0,0,0)] = h0.intra
    for t in h0.hopping: hop[tuple(t.dir)] = t.m
    return hop


# mode -> builder(h,h0,fun,mode); several names are aliases of one builder
_guesses = {
  "ferro": _guess_ferro,
  "magnetic": _guess_magnetic,
  "ferroX": _guess_ferro_axis(0),
  "ferroY": _guess_ferro_axis(1),
  "ferroZ": _guess_ferro_axis(2),
  "randomXY": _guess_randomXY,
  "XY": _guess_randomXY,
  "random": _guess_random,
  "Fully random": _guess_random,
  "dimerization": _guess_dimerization,
  "kekule": _guess_kekule,
  "Haldane": _guess_haldane,
  "rashba": _guess_rashba,
  "kanemele": _guess_kanemele,
  "antihaldane": _guess_antihaldane,
  "valley": _guess_antihaldane,
  "CDW": _guess_CDW,
  "Charge density wave": _guess_CDW,
  "potential": _guess_potential,
  "antiferro": _guess_antiferro,
  "imbalance": _guess_imbalance,
  "swave": _guess_swave,
  "s-wave superconductivity": _guess_swave,
  "pwave": _guess_pwave,
  }


# every mode understood by guess(), derived from the dispatch above
known_guesses = list(_guesses)


def get_guess_names():
    """Return every mean-field initialization that guess() accepts"""
    return list(_guesses)


def guess(h,mode="ferro",fun=1e-1):
  """Return a mean field matrix guess given a certain Hamiltonian"""
  if mode not in _guesses:
      raise ValueError("Unrecognized mean-field initialization '"+str(mode)
        +"'. Known modes: "+str(sorted(set(known_guesses))))
  h0 = h.copy() # copy Hamiltonian
  h0 = h0.get_multicell() # multicell
  h0.clean() # clean the Hamiltonian
  return _guesses[mode](h,h0,fun,mode)


from .algebra import braket_wAw
#from numba import jit

#@jit
def expectation_value(wfs,A,phis):
  """Return the expectation value of a set of wavevectors"""
  out = 0.0j
  for (p,w) in zip(phis,wfs):
    out += braket_wAw(w,A)*p
  return np.conjugate(out) # return value




def enforce_pwave(mf):
  """Enforce pwave symmetry in a mean field Hamiltonian"""
  for key in mf: mf[key] = np.array(mf[key]) # dense matrix
  for key in mf:
#    print(mf[key],type(mf[key]))
    n = mf[key].shape[0]//4 # number of sites 
    dm = tuple([-di for di in key]) # the opposite matrix
    for i in range(n): # loop over positions
      for j in range(n): # loop over positions
        for (ii,jj) in [(1,2),(0,3),(2,1),(3,0)]:
          mf[key][4*i+ii,4*j+jj] = (mf[key][4*i+ii,4*j+jj] - mf[dm][4*j+ii,4*i+jj])/2.
  return mf


def enforce_eh(h,mf):
  """Enforce eh symmetry in a mean field Hamiltonian"""
  from .superconductivity import eh_operator
  eh = eh_operator(h.intra) # get the function
  for key in mf: mf[key] = np.array(mf[key]) # dense matrix
  mfout = dict()
  for key in mf:
    mkey = (-key[0],-key[1],-key[2])
    mfout[key] = (mf[key] - eh(mf[mkey].conj().T))/2.
  return mfout



def broyden_solver(scf):
    """Broyden solver for selfconsistency"""
    scf.mixing = 1.0 # perfect mixing
    scf.iterate() # do one iteration
    x0 = scf.cij # get the array with expectation values
    def fun(x): # function to solve
        scf.cij = x # impose those expectation values
        scf.cij2v() # update mean field
        scf.iterate() # perform an iteration
        return scf.cij - x # difference with respect to input
    from scipy.optimize import broyden1,broyden2,anderson
#    x = anderson(fun,x0)
    x = broyden1(fun,x0,f_tol=1e-7) # broyden solver
    scf.cij = x # impose those expectation values
    scf.cij2v() # update mean field
    scf.iterate() # perform an iteration
    return scf # return scf




def coulomb_interaction(g,vc=1.0,vcut=1e-4,vfun=None,has_spin=False,**kwargs):
    """Return a list with the Coulomb interaction terms"""
    interactions = [] # empty list
    nat = len(g.r) # number of atoms
    lat = np.sqrt(g.a1.dot(g.a1)) # size of the unit cell
    rcut = 4.0
    g.ncells = max([int(2*rcut/lat),1]) # number of unit cells to consider
    ri = g.r # positions
    if vfun is None: # no function provided
        def vfun(dr):
            if dr<1e-4: return 0.0
            if dr>rcut: return 0.0
            return vc/dr*np.exp(-dr/rcut)
    for d in g.neighbor_directions(): # loop
      rj = np.array(g.replicas(d)) # positions
      for i in range(nat): # loop over atoms
        for j in range(nat): # loop over atoms
          dx = rj[j,0] - ri[i,0]
          dy = rj[j,1] - ri[i,1]
          dz = rj[j,2] - ri[i,2]
          dr = dx*dx + dy*dy + dz*dz
          dr = np.sqrt(dr) # square root
          v = vfun(dr) # interaction
          if v>1e-3: # sizeble interaction
            if has_spin:
              interactions.append(v_ij_density_spinless(2*i,2*j+1,2*nat,
              g=v,d=[0,0,0]))
            else:
              interactions.append(v_ij_density_spinless(i,j,nat,
              g=v,d=[0,0,0]))
    return interactions


def coulomb_interaction_spinless(g,**kwargs):
    return coulomb_interaction(g,has_spin=False,**kwargs)


def coulomb_interaction_spinful(g,**kwargs):
    return coulomb_interaction(g,has_spin=True,**kwargs)



def fast_coulomb_interaction(g,vc=1.0,vcut=1e-4,vfun=None,has_spin=False,**kwargs):
    """Return a list with the Coulomb interaction terms, summed over sites"""
    interactions = [] # empty list
    nat = len(g.r) # number of atoms
    lat = np.sqrt(g.a1.dot(g.a1)) # size of the unit cell
    rcut = 4.0
    g.ncells = max([int(2*rcut/lat),1]) # number of unit cells to consider
    ri = g.r # positions
    if vfun is None: # no function provided
        def vfun(dr):
            if dr<1e-4: return 0.0
            if dr>rcut: return 0.0
            return vc/dr*np.exp(-dr/rcut)
    for i in range(nat): # loop over atoms
      vjs = np.zeros(nat) # initialize
      for d in g.neighbor_directions(): # loop
        rj = np.array(g.replicas(d)) # positions
        for j in range(nat): # loop over atoms
          dx = rj[j,0] - ri[i,0]
          dy = rj[j,1] - ri[i,1]
          dz = rj[j,2] - ri[i,2]
          dr = dx*dx + dy*dy + dz*dz
          dr = np.sqrt(dr) # square root
          vt = vfun(dr) # interaction
          vjs[j] += vt # add contribution
      vjs[vjs<1e-4] = 0.0 # discard near zeros
      if np.sum(vjs)>1e-3: # sizable interaction
        print("Total Coulomb term",np.sum(vjs))
        if has_spin:
          interactions.append(
                  v_ij_fast_coulomb_spinful(i,vjs,nat,channel="up")
                  )
          interactions.append(
                  v_ij_fast_coulomb_spinful(i,vjs,nat,channel="down")
                  )
        else:
            raise NotImplementedError("the fast-Coulomb interaction is only "
                    "implemented for spinless and spinful Hamiltonians, not "
                    "for Nambu ones")
    return interactions


def identify_symmetry_breaking(h0,h,as_string=False,tol=1e-3):
    """Given two Hamiltonians, identify what is the symmetry
    breaking between them"""
    dt0 = h0.get_multihopping() # first multihopping
    dt = h.get_multihopping() # second multihopping
    dd = dt0 - dt # difference between Hamiltonians
    out = [] # empty list
    if dd.norm()>tol: # non-zero
        for s in symmetry_breaking: # loop over contributions
            try:
                d0 = MultiHopping(guess(h,s,fun=1.0)) # get this type
                proj = dd.dot(d0) # compute the projection
                if np.abs(proj)>tol: out.append(s)
            except: pass
        h1 = h0.copy() # copy
        h1.set_multihopping(dd) # set the difference
        out += superconductivity.identify_superconductivity(h1,tol=tol) # check SC
    else: out = ["No symmetry breaking"]
    if len(out)==0: out = ["Unidentified symmetry breaking"]
    if as_string: # return as a string
        out2 = ""
        for o in out: out2 += o + ", "
        return out2
    return out


def order_parameter(self,name):
    mf = self.hamiltonian - self.hamiltonian0 # mean field
    if name=="even_SC":
        from .sctk.orderparameter import singlet
        return singlet(mf)
    elif name=="odd_SC":
        from .sctk.orderparameter import triplet
        return triplet(mf)
    else:
        raise ValueError("unknown order parameter; the accepted ones are "
                "'even_SC' and 'odd_SC'")




from .scftk import densitydensity

hubbardscf = densitydensity.hubbard
Vinteraction = densitydensity.Vinteraction

from .scftk import densitydensity_kpm

hubbardscf_kpm = densitydensity_kpm.hubbard_kpm
Vinteraction_kpm = densitydensity_kpm.Vinteraction_kpm

from .scftk.potentials import keldysh

from .scftk import spinspin

SzSz = spinspin.SzSz
SxSx = spinspin.SxSx
SySy = spinspin.SySy
Jinteraction = spinspin.Jinteraction
VJinteraction = spinspin.VJinteraction
