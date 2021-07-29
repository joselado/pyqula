from __future__ import print_function
import scipy.sparse.linalg as lg
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix,bmat
import time
import numpy as np
import os
import scipy.optimize as optimize
from . import klist
from . import inout
from . import algebra
from . import meanfield
from . import groundstate
from . import parallel

from .meanfield import guess # function to calculate guesses
try:
  from . import correlatorsf90
  multicorrelator = correlatorsf90.multicorrelator
  multicorrelator_bloch = correlatorsf90.multicorrelator_bloch
  use_multicorrelator = True
except:
#  print("WARNING, FORTRAN not working in scftypes.py")
  use_multicorrelator = False
#  raise
#  def multicorrelator


timing = True


def load(input_file="SCF.pkl"):
  """Get a selfconsistent calculation"""
  return inout.load(input_file)

class scfclass(): 
  """Class for a selfconsistent calculation"""
  def copy(self):
    from copy import deepcopy
    return deepcopy(self)
  def save(self,output_file="SCF.pkl"):
    inout.save(self,output_file)
  def __init__(self,h):
    self.has_index2vec = False
    self.iteration = 0 # initialize
    self.scfmode = "filling" # do SCF fixing the number of filled states
    self.interactions = [] # empty list
    self.nkgrid = 1 # one kpoint
    self.mf = dict() # dictionary
    self.kfac = 1.0 # normalization factor for the number of kpoints
    self.fermi = 0.0 # fermi energy
    self.error = 0.0 # error in the selfconsistency
    self.silent = False
    self.gap = 0.0 # gap of the system
    self.smearing = None
    self.write_vabv = False # write EV in each iteration
    if use_multicorrelator: # fortran library is present
      self.correlator_mode = "multicorrelator" # multicorrelator mode 
    else:
      self.correlator_mode = "plain" # multicorrelator mode 
#    self.correlator_mode = "1by1" # multicorrelator mode 
    self.bloch_multicorrelator = False # multicorrelator with Bloch phases
    self.enforce_symmetry = "none" # do not enforce any symmetry
    self.hamiltonian0 = h.get_multicell() # store Hamiltonian
    self.hamiltonian = self.hamiltonian0.copy() # store Hamiltonian
    self.has_eh = self.hamiltonian.has_eh # has electron-hole
    self.is_sparse = self.hamiltonian.is_sparse # sparse Hamiltonian
    self.sites = len(self.hamiltonian.geometry.r) # number of sites
    self.iteration = 0 # first iteration
    self.filling = 0.5 # half filling
    self.mixing = 0.9 # initialize the mixing
    self.fermi_shift = 0.0 # shift in the fermi energy
    self.energy_cutoff = 1.0 # energy for the selfconsistency
    self.num_waves = 10 # number of waves to compute
    self.use_weights = False # calculate SCF using weights
  def update_occupied_states(self,fermi_shift=0.0):
    """Get the eigenvectors for a mesh of kpoints"""
    mine = None # minimum energy
    if self.scfmode=="fermi" and self.is_sparse:
      self.hamiltonian.turn_sparse()
      print("WARNING!!! using sparse mode")
      print("Use this mode only if you know what you are doing!!!!\n\n")
      es,ws,ks = self.hamiltonian.get_eigenvectors(nk=self.nkgrid,kpoints=True,
                              sparse=True,numw=self.num_waves)
      if np.max(np.abs(es))*0.9<self.energy_cutoff: 
        print("NOT ENOUGH STATES, recalling with",self.num_waves*2)
        self.num_waves += 5 
        self.update_occupied_states() # call again
#  raise 
    else: # any other, use brute force
      es,ws,ks = self.hamiltonian.get_eigenvectors(nk=self.nkgrid,kpoints=True)
#    self.kfac = float(len(ks))/self.hamiltonian.intra.shape[0] 
    # number of kpoints of the calculation
#    mine = min(es)*0.9 # minimum energy retained
    self.kfac = len(klist.kmesh(self.hamiltonian.dimensionality,nk=self.nkgrid))
    if self.scfmode=="filling": # get the fermi energy if the mode requires it 
      self.fermi = get_fermi_energy(es,self.filling,fermi_shift=fermi_shift) 
    elif self.scfmode=="fermi": pass # do nothing
    self.gap = get_gap(es,self.fermi) # store the gap
    if self.energy_cutoff is not None:
      if self.gap>self.energy_cutoff/2: 
        print("\nEnergy cutoff is to small, halting",self.gap,"\n\n")
        raise
      print("Warning!!!! Performing calculation with an energy cutoff")
    eoccs,voccs,koccs = get_occupied_states(es,ws,ks,self.fermi,
                            mine=self.energy_cutoff,smearing = self.smearing)
    self.wavefunctions = voccs # store wavefunctions
#    print(len(voccs),voccs.shape,len(voccs[0]))
    self.energies = eoccs # store energies
    self.kvectors = koccs # store kvectors
  def update_occupied_states_kpoint(self,k=[0.,0.,0.]):
    """Update the object with the eigenvectors of a single kpoint,
    this function is used for the adaptive method of SCF"""
    if not self.scfmode=="fermi": raise # only for this mode
    es,ws,ks = self.hamiltonian.get_eigenvectors(nk=self.nkgrid,
            kpoints=True,k=k)
    self.kfac = 1
    self.gap = get_gap(es,self.fermi) # store the gap
    eoccs,voccs,koccs = get_occupied_states(es,ws,ks,self.fermi) # occupied states
    self.wavefunctions = voccs # store wavefunctions
    self.energies = eoccs # store energies
    self.kvectors = koccs # store kvectors
  def adaptive_correlator(self):
    """Calculate the correlators using the Simpson's adaptive algorithm"""
    raise # not finished yet
  def update_hamiltonian(self):
    """Updates the total mean field Hamiltonian"""
    self.hamiltonian = self.hamiltonian0.copy() # copy original
    self.hamiltonian.intra += self.mf[(0,0,0)] # add mean field
    if self.hamiltonian.dimensionality>0:
      from . import multicell
      hopping = self.hamiltonian.hopping # empty list
      for key in self.mf: # loop over the mean field
        if key==(0,0,0): continue
        m = self.mf[key] # get this hopping
        hopping.append(multicell.Hopping(d=key,m=m)) # add this one
      self.hamiltonian.hopping = multicell.collect_hopping(self.hamiltonian)
  def setup_interaction(self,mode="Hubbard",g=1.0,**kwargs):
    """Create the operators that will determine the interactions"""
    if timing: t0 = time.perf_counter()
    interactions = [] # empty list
    has_eh = self.hamiltonian.has_eh # has electron hole
    nat = self.sites # number of sites
#    self.correlator_mode = "1by1" # mode to calculate the correlators
    if has_eh: # for superconducting systems
      self.scfmode = "fermi"  # SCF calculation fixing the fermi energy
      self.fermi = 0.0 # set the fermi energy in 0
      if mode=="Hubbard" or mode=="U": # Hubbard model
        print("Adding Hubbard interaction")
        for i in range(nat): 
          interactions.append(meanfield.hubbard_pairing_ud(i,nat,g=g)) 
          interactions.append(meanfield.hubbard_pairing_du(i,nat,g=g)) 
      elif mode=="V": # V interaction
        print("Adding V interaction")
        self.bloch_multicorrelator = True
        from . import neighbor
 #       self.nkgrid = 1 # only implemented in Gamma point
        for d in self.hamiltonian.geometry.neighbor_directions(): # loop
          ri = self.hamiltonian.geometry.r # positions
          rj = self.hamiltonian.geometry.replicas(d) # positions
          ijs = neighbor.find_first_neighbor(ri,rj) # return pairs
          if len(ijs)>0: self.mf[tuple(d)] = self.hamiltonian.intra*0. # initialize
          for (i,j) in ijs: # loop over pairs of sites
#            interactions.append(meanfield.v_pairing_du(i,j,nat,g=g,d=d)) 
#            interactions.append(meanfield.v_pairing_du(j,i,nat,g=g,d=d)) 
            interactions.append(meanfield.v_pairing_dd(i,j,nat,g=g,d=d,channel="ee")) 
            interactions.append(meanfield.v_pairing_dd(i,j,nat,g=g,d=d,channel="hh")) 
            interactions.append(meanfield.v_pairing_uu(i,j,nat,g=g,d=d,channel="ee")) 
            interactions.append(meanfield.v_pairing_uu(i,j,nat,g=g,d=d,channel="hh")) 
    else: # no electron hole symmetry
      if mode=="Hubbard" or mode=="U": # Hubbard model
        if self.hamiltonian.has_spin: # spinful
          print("Adding Spinful Hubbard interaction")
          for i in range(nat): 
            interactions.append(meanfield.hubbard_density(i,nat,g=g)) 
            interactions.append(meanfield.hubbard_exchange(i,nat,g=g)) 
        else:
          print("Adding spinless Hubbard interaction")
          for i in range(nat): 
            interactions.append(meanfield.spinless_hubbard_density(i,nat,g=g)) 
      elif mode=="Hubbard collinear": # Hubbard model
        print("Adding Hubbard collinear interaction")
        for i in range(nat): 
          interactions.append(meanfield.hubbard_density(i,nat,g=g)) 
  # store this interaction
      elif mode=="Coulomb": # Coulomb interaction
        interactions = meanfield.coulomb_interaction(
                self.hamiltonian.geometry,has_spin=self.hamiltonian.has_spin,
                vc=g,**kwargs)
      elif mode=="fastCoulomb": # Coulomb interaction
        interactions = meanfield.fast_coulomb_interaction(
                self.hamiltonian.geometry,has_spin=self.hamiltonian.has_spin,
                vc=g,**kwargs)
      elif mode=="V": # V interaction
#        self.correlator_mode = "1by1" # mode to calculate the correlators
        self.bloch_multicorrelator = True
        from . import neighbor
#        self.nkgrid = 1 # only implemented in Gamma point
        for d in self.hamiltonian.geometry.neighbor_directions(): # loop
          ri = self.hamiltonian.geometry.r # positions
          rj = self.hamiltonian.geometry.replicas(d) # positions
          ijs = neighbor.find_first_neighbor(ri,rj) # return pairs
          if len(ijs)>0: self.mf[tuple(d)] = 0.0j # initialize
          for (i,j) in ijs: # loop over pairs of sites
            if self.hamiltonian.has_spin: # spinful
              interactions.append(meanfield.v_ij(i,j,nat,g=g,d=d,spini=0,spinj=0)) 
              interactions.append(meanfield.v_ij(i,j,nat,g=g,d=d,spini=0,spinj=1)) 
              interactions.append(meanfield.v_ij(i,j,nat,g=g,d=d,spini=1,spinj=0)) 
              interactions.append(meanfield.v_ij(i,j,nat,g=g,d=d,spini=1,spinj=1)) 
            else: # spinless
              # factor 2 in g due to spin degree of freedom
              interactions.append(meanfield.v_ij_spinless(i,j,nat,g=2*g,d=d)) 
      else: raise # ups
    self.interactions += interactions # store list
    if timing: print("Time in creating MF operators",time.perf_counter()-t0)
    self.setup_multicorrelator()
  def setup_multicorrelator(self):
    """Create the neccesary arrays to calculate the different
    correlators efficiently"""
    lamb = [] # empty list
    ijk = []
    dv = [] # direction of the interaction
    k = 0
    for v in self.interactions: # loop
      ac = coo_matrix(v.a)       
      bc = coo_matrix(v.b)       
      for (i,j,d) in zip(ac.row,ac.col,ac.data): 
        ijk.append([i,j,k])
        lamb.append(d) # store data
        dv.append(v.dir) # store direction
      k += 1 # increase counter
      for (i,j,d) in zip(bc.row,bc.col,bc.data): 
        ijk.append([i,j,k])
        lamb.append(d) # store data
        dv.append(-np.array(v.dir)) # store direction, be aware of the sign!!!
      k += 1 # increase counter
    ijk = np.array(ijk,dtype=np.int)
    lamb = np.array(lamb,dtype=np.complex) # data array
    self.ijk = ijk # first array
    self.lamb = lamb # data array
    self.dir = np.array(dv,dtype=np.int) # data array
#    self.tensormf = algebra.sparsetensor.Tensor3(ijk[:,0],
#            ijk[:,1],ijk[:,2],lamb,
#            shape=(v.a.shape[0],v.a.shape[0],k))
  def update_expectation_values(self):
    """Calculate the expectation values of the different operators"""
    # this conjugate comes from being inconsistent
    # in the routines to calculate exectation values
    voccs = np.array(np.conjugate(self.wavefunctions)) # get wavefunctions
    ks = self.kvectors # kpoints
#    self.correlator_mode = "plain"
    mode = self.correlator_mode # 
    if mode=="plain": # conventional mode
        plain_expectation_value(self)
#      for v in self.interactions:
#        v.vav = np.trace(voccs@v.a@np.conjugate(voccs).T)/self.kfac # <vAv>
#        v.vbv = np.trace(voccs@v.b@np.conjugate(voccs).T)/self.kfac # <vBv>
    elif mode=="1by1": # conventional mode
      for v in self.interactions:
        phis = [self.hamiltonian.geometry.bloch_phase(v.dir,k*0.) for k in ks]
        v.vav = meanfield.expectation_value(voccs,v.a,np.conjugate(phis))/self.kfac # <vAv>
        v.vbv = meanfield.expectation_value(voccs,v.b,phis)/self.kfac # <vBv>
      self.v2cij() # update the v vector
    elif mode=="multicorrelator": # multicorrelator mode
      numc = len(self.interactions)*2 # number of correlators
      if self.bloch_multicorrelator:
        cs = multicorrelator_bloch(voccs,ks,self.lamb,self.ijk,self.dir,numc)
      else: cs = multicorrelator(voccs,self.lamb,self.ijk,numc)
      self.cij = cs/self.kfac # store in the object, already normalized
      self.cij2v() # update the expectation values
    else: raise
  def cij2v(self):
    """Update the values of vav and vbv using the values of cij"""
    cs = self.cij # get the array
    k = 0
    for v in self.interactions:
      v.vav = cs[k]
      k += 1
      v.vbv = cs[k]
      k += 1
  def v2cij(self):
    """Update the value of cij with the values of v"""
    cs = np.zeros(len(self.interactions)*2,dtype=np.complex) # get the array
    k = 0
    for v in self.interactions:
      cs[k] = v.vav
      k += 1
      cs[k] = v.vbv 
      k += 1
    self.cij = cs.copy() # store
  def update_mean_field(self,**kwargs):
    """Calculate the expectation values of the different operators"""
    from .scftk.updatescf import update_mean_field
    update_mean_field(self,**kwargs)
  def get_total_energy(self):
    """Return the total energy"""
    eout = np.sum(self.energies)/self.kfac 
    for v in self.interactions:
      if v.contribution=="AB": eout += -v.vav*v.vbv*v.g
      else: eout += -v.vav*v.vbv*v.g/2.
    self.total_energy = eout.real # store the total energy
    return eout.real # return total energy of the system
  def extract(self):
    """Write in file different parts of the Hamiltonian"""
    if self.hamiltonian.has_eh: groundstate.swave(self.hamiltonian) 
  def iterate(self):
    """Perform a single iteration"""
    if not self.silent: print("## Iteration number ",self.iteration)
    mixing = self.mixing 
    self.update_hamiltonian() # update the Hamiltonian
    try: self.hamiltonian.check() # check that nothing weird happened
    except: 
        print("WARNING, hamiltonian is non hermitian in mean field")
        raise
    t1 = time.perf_counter()
    self.update_occupied_states(fermi_shift=self.fermi_shift)
    t2 = time.perf_counter()
    self.update_expectation_values() # calculate expectation values
    t3 = time.perf_counter()
    self.update_mean_field(mixing=mixing) # calculate mean field
    t4 = time.perf_counter()
    self.iteration += 1
    if not self.silent:
      print("Time in diagonalization",t2-t1)
      print("Time in expectation values",t3-t2)
      print("Time in new Hamiltonian",t4-t3)
      print("\n")
  def solve(self):
    """Solve the selfconsistent problem"""
    return
    self.iterate() # first iteration
    def r2z(v): # convert real to complex
      n = v.shape[0]//2
      return v[0:n] + 1j*v[n:2*n]
    def z2r(v): # convert complex to real
      return np.concatenate([v.real,v.imag])

    def fopt(cij): # function to return the error
      self.cij = r2z(cij) # store this vector
      self.cij2v()  # update the v vectors
      self.iterate() # do an iteration
      print(self.cij)
      self.v2cij() # convert the v to cij
      return z2r(self.cij)-cij
    cij = optimize.broyden1(fopt,z2r(self.cij),f_tol=1e-8,max_rank=10)
#    cij = optimize.fsolve(fopt,z2r(self.cij),xtol=1e-8)
#    cij = optimize.anderson(fopt,z2r(self.cij),f_tol=1e-6,w0=0.1)
#    cij = optimize.newton_krylov(fopt,z2r(self.cij),f_tol=1e-6,outer_k=8)
    self.cij = cij
    self.cij2v() # update




def directional_mean_field(vecs):
  """ Creates an initial mean field accoring to certain vectors"""
  from .hamiltonians import sx,sy,sz
  mf = [[None for i in vecs] for j in vecs] # create
  for i in range(len(vecs)):
    v = vecs[i]
    mf[i][i] = v[0]*sx + v[1]*sy + v[2]*sz # add contribution
  return bmat(mf).todense()



def get_occupied_states(es,ws,ks,fermi,smearing=None,mine=None):
  """Return the occupied states"""
  if smearing is None: # no smearing
    voccs = [] # occupied vectors
    eoccs = [] # occupied eigenvalues
    koccs = [] # occupied eigenvalues
    if mine is None: mine = -1000000 # accept all
    else: mine = -np.abs(mine)
    for (e,v,k) in zip(es,ws,ks): # loop over eigenvals,eigenvecs
      if mine<e<fermi:  # if level is filled, add contribution
        voccs.append(v) # store
        eoccs.append(e) # store
        koccs.append(k) # store
    voccs = np.array(voccs)  # as array
    eoccs = np.array(eoccs)  # as array
    koccs = np.array(koccs)  # as array
  else:
    voccs = [] # occupied vectors
    eoccs = [] # occupied eigenvalues
    koccs = [] # occupied eigenvalues
    if mine is None: mine = -1000000 # accept all
    else: mine = -np.abs(mine)
    for (e,v,k) in zip(es,ws,ks): # loop over eigenvals,eigenvecs
      weight = 1./(np.exp((e-fermi)/smearing)+1.0) # occupation
      voccs.append(v*weight) # store
      eoccs.append(e*weight) # store
      koccs.append(k) # store
    voccs = np.matrix(np.array(voccs))  # as array
    eoccs = np.array(eoccs)  # as array
    koccs = np.array(koccs)  # as array
  return eoccs,voccs,koccs


from .spectrum import get_fermi_energy


def get_gap(es,fermi):
  """Return the gap"""
  dife = es - fermi # difference in energy
  gap = np.min(np.abs(dife))*2 # gap
  return gap





from .selfconsistency.hubbard import hubbardscf
from .selfconsistency.coulomb import coulombscf
from .selfconsistency.densitydensity import densitydensity

repulsive_hubbard = hubbardscf
from .selfconsistency.attractive_hubbard_spinless import attractive_hubbard

def get_super_correlator(voccs,weight=None,totkp=1):
  """Get the different correlators for a superconducting system"""
  from . import correlatorsf90
  ndim = voccs.shape[1] # dimension of the matrix
  if weight is not None:
    if len(weight)!=voccs.shape[0]: raise # inconsistent dimensions
  nat = ndim//4 # one fourth
  corrs = [] # empty list with the different correlators
  ops = [] # list with the operators
  for i in nat: # loop over atoms
    corrs.append([4*i,4*i]) # up density
    ops.append(csc_matrix(([1.],[1,1]),dtype=np.complex)) # down operator
    corrs.append([4*i+1,4*i+1]) # up density
    ops.append(csc_matrix(([1.],[0,0]),dtype=np.complex)) # down operator
    corrs.append([4*i,4*i+1]) # up density
    ops.append(csc_matrix(([-1.],[1,0]),dtype=np.complex)) # down operator
    corrs.append([4*i+1,4*i]) # up density
    ops.append(csc_matrix(([-1.],[0,1]),dtype=np.complex)) # down operator
  pdup = np.array([[4*i,4*i] for i in range(nat)]) # up density
  pddn = pdup + 1 # down density
  pxc = np.array([[4*i,4*i+1] for i in range(nat)]) # exchange
  deltadd = np.array([[4*i+1,4*i+2] for i in range(nat)]) # Delta dd
  deltauu = np.array([[4*i,4*i+3] for i in range(nat)]) # Delta uu
  deltaud = np.array([[4*i,4*i+2] for i in range(nat)]) # Delta ud
  deltadu = np.array([[4*i+1,4*i+3] for i in range(nat)]) # Delta du
  if weight is None: # no weight
    vdup = correlatorsf90.correlators(voccs,pdup)/totkp
    vddn = correlatorsf90.correlators(voccs,pddn)/totkp
    vxc = correlatorsf90.correlators(voccs,pxc)/totkp
    vdeltadd = correlatorsf90.correlators(voccs,deltadd)/totkp
    vdeltauu = correlatorsf90.correlators(voccs,deltauu)/totkp
    vdeltadu = correlatorsf90.correlators(voccs,deltadu)/totkp
    vdeltaud = correlatorsf90.correlators(voccs,deltaud)/totkp
  else: # with weight
    raise
    vdup = correlatorsf90.correlators_weighted(voccs,weight,pdup)/totkp
    vddn = correlatorsf90.correlators_weighted(voccs,weight,pddn)/totkp
    vxc = correlatorsf90.correlators_weighted(voccs,pxc)/totkp
  ndn = csc_matrix((vdup,pddn.transpose()),dtype=np.complex,shape=(ndim,ndim))
  nup = csc_matrix((vddn,pdup.transpose()),dtype=np.complex,shape=(ndim,ndim))
  xc = csc_matrix((np.conjugate(vxc),pxc.transpose()),
                           dtype=np.complex,shape=(ndim,ndim))
  return (vdup,vddn,vxc,ndn,nup,xc) # return everything





def write_magnetization(mag):
  """Write magnetization in a file"""
  fo = open("MAGNETIZATION.OUT","w")
  ix = 1
  for m in mag:
    fo.write(str(ix)+"  ")
    fo.write(str(m[0])+"  ")
    fo.write(str(m[1])+"  ")
    fo.write(str(m[2])+"\n")
    ix += 1
  fo.close()






#
#def selfconsistency(h,g=1.0,nkp = 100,filling=0.5,mix=0.2,
#                  maxerror=1e-05,silent=False,mf=None,nk=None,
#                  smearing=None,fermi_shift=0.0,save=True,
#                  mode="Hubbard",energy_cutoff=None,maxite=1000,
#                  broyden=False,callback=None,**kwargs):
#  """ Solve a generalized selfcnsistent problem"""
#  if nk is not None: nkp = nk # redefine
#  mf_file = "MF.pkl"
#  os.system("rm -f STOP") # remove stop file
#  nat = h.intra.shape[0]//2 # number of atoms
#  htmp = h.copy()  # copy hamiltonian
#  htmp.turn_dense() # turn to dense Hamiltonian
#  # generalate the necessary list of correlators
#  if mf is None: # generate initial mean field
#    try:  old_mf = inout.load(mf_file) # load the file
#    except: old_mf = meanfield.guess(h,"random") # random guess
#  else: old_mf = mf # use guess
#  # get the pairs for the correlators
#  ndim = h.intra.shape[0] # dimension
#  totkp = nkp**(h.dimensionality) # total number of kpoints
#  file_etot = open("SCF_ENERGY.OUT","w")
#  file_gap = open("SCF_GAP.OUT","w")
#  file_error = open("SCF_ERROR.OUT","w")
#  ite = 0 # iteration counter
#  scf = scfclass(htmp) # create scf class
#  scf.nkgrid = nkp
#  scf.silent = silent
#  scf.mixing = mix
#  scf.mode = mode # store the mode
#  scf.g = g # coupling
#  scf.smearing = smearing
#  scf.energy_cutoff = energy_cutoff # energy_cutoff
#  scf.filling = filling # filling of the system
##  scf.mf0 = old_mf # initial mean field
#  if type(mode) is type(dict()): # of mode is a dictionary add several interactions
#    print("Adding multiple interactions")
#    for key in mode:
#      scf.setup_interaction(g=mode[key],mode=key,**kwargs) # create the interaction matrices
#  else: # conventional way
#    scf.setup_interaction(g=g,mode=mode,**kwargs) # create the interaction matrices
#  if type(old_mf) is type(dict()): # of mode is a dictionary add several interactions
#    scf.mf = old_mf # initial mean field
#  else:
#    scf.mf[(0,0,0)] = old_mf # initial mean field
##  scf.solve()
#  stop_scf = False # do not stop
#  scf.maxerror = maxerror
##  print("BEGINNING OF BROYDEN")
##  print("END OF BROYDEN")
#  while True: # infinite loop
#    scf.iterate() # do an iteration
#    if scf.hamiltonian.has_spin: 
#        scf.hamiltonian.write_magnetization() # write the magnetization
#    eout = scf.get_total_energy() # total energy
#    if not silent: print("Total energy",eout)
##    etot = np.sum(eoccs)/totkp + edc  # eigenvalues and double counting
#    ite = scf.iteration
#    file_etot.write(str(ite)+"    "+str(scf.get_total_energy())+"\n") 
#    file_error.write(str(ite)+"    "+str(scf.error)+"\n") # write energy in file
#    file_gap.write(str(ite)+"    "+str(scf.gap)+"\n") # write energy in file
##    totcharge = np.sum(charge).real # total charge
##    avcharge = totcharge/nat # average charge
##    ######
#    if callback is not None: callback(scf.hamiltonian) # callback
#    if not silent:
#      print("\n")
#      print("Error in SCF =",scf.error)
#      print("Fermi energy =",scf.fermi)
#      print("Gap =",scf.gap)
#    if scf.error<maxerror or os.path.exists("STOP") or scf.iteration==maxite: 
#      stop_scf = True # stop the calculation after the next iteration
#      scf.mixing = 1.0 # last iteration with mixing one
#      scf.smearing = None # last iteration without smearing
#    if stop_scf: break # stop the calculation
#  if broyden: scf = meanfield.broyden_solver(scf)
#  if save: inout.save(scf.mf,mf_file) # save mean field
#  file_etot.close() # close file
#  file_error.close() # close file
#  file_gap.close() # close file
#  if scf.scfmode=="filling":  scf.hamiltonian.shift_fermi(-scf.fermi)
#  return scf # return mean field




def plain_expectation_value(self):
    """Compute the expectation values using the plain scheme"""
    voccs = np.array(np.conjugate(self.wavefunctions))
    if parallel.cores==1: # single core execution
      for v in self.interactions:
        v.vav = np.trace(voccs@v.a@np.conjugate(voccs).T)/self.kfac # <vAv>
        v.vbv = np.trace(voccs@v.b@np.conjugate(voccs).T)/self.kfac # <vBv>
    else: # multicore execution
        ma = [v.a for v in self.interactions]
        mb = [v.b for v in self.interactions]
        kfac = self.kfac
        def fun(i):
            vav = np.trace(voccs@ma[i]@np.conjugate(voccs).T)/kfac
            vbv = np.trace(voccs@mb[i]@np.conjugate(voccs).T)/kfac
            return [vav,vbv]
        # compute expectation values
        vv = parallel.pcall(fun,range(len(self.interactions)))
        for i in range(len(self.interactions)):
            self.interactions[i].vav = vv[i][0]
            self.interactions[i].vbv = vv[i][1]


# workaround
from .meanfield import Vinteraction as selfconsistency

