from __future__ import print_function
from __future__ import division
from .helptk import get_docstring
import numpy as np
from scipy.sparse import csc_matrix,bmat,csr_matrix
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import numpy as np
from . import ldos
from . import operators
from . import inout
from . import superconductivity
from . import kanemele 
from . import magnetism
from . import checkclass
from . import extract
from . import multicell
from . import spectrum
from . import kekule
from . import algebra
from . import groundstate
from . import rotate_spin
from . import topology
from . import ldos
from . import increase_hilbert
from .meanfield import Vinteraction
from .sctk import dvector
from .algebratk import hamiltonianalgebra
from .bandstructure import get_bands_nd

from scipy.sparse import coo_matrix,bmat,csc_matrix
from .rotate_spin import sx,sy,sz
from .increase_hilbert import get_spinless2full,get_spinful2full
from . import tails
from scipy.sparse import diags as sparse_diag
import pickle
from .htk import mode as hamiltonianmode
from .htk import symmetry

#import data

from .limits import densedimension as maxmatrix
optimal = False

class Hamiltonian():
    """ Class for a hamiltonian """
    def get_tails(self,discard=None):
      """Write the tails of the wavefunctions"""
      if self.dimensionality!=0: raise
      else: return tails.matrix_tails(self.intra,discard=discard)
    def __add__(self,h):  return hamiltonianalgebra.add(self,h)
    def __rmul__(self,h):  return hamiltonianalgebra.rmul(self,h)
    def __mul__(self,h):  return hamiltonianalgebra.mul(self,h)
    def __neg__(self):  return (-1)*self
    def __sub__(self,a):  return self + (-a)
    def spinless2full(self,m,time_reversal=False):
      """Transform a spinless matrix in its full form"""
      return get_spinless2full(self,time_reversal=time_reversal)(m) # return
    def spinful2full(self,m):
      """Transform a spinless matrix in its full form"""
      return get_spinful2full(self)(m) # return
    def kchain(self,k=0.):
      return kchain(self,k)
    def get_fermi_surface(self,**kwargs):
        return spectrum.fermi_surface(self,**kwargs)
    def get_multi_fermi_surface(self,**kwargs):
        return spectrum.multi_fermi_surface(self,**kwargs)
    def get_eigenvectors(self,**kwargs):
        from .htk.eigenvectors import get_eigenvectors
        return get_eigenvectors(self,**kwargs)
    def modify_hamiltonian_matrices(self,f):
        """Modify all the matrices of a Hamiltonian"""
        modify_hamiltonian_matrices(self,f)
    def remove_sites(self,store):
        from . import sculpt
        self.geometry = sculpt.remove_sites(self.geometry,store)
        from .algebratk.matrixcrop import crop_matrix
        if self.has_spin: raise
        f = lambda m: crop_matrix(m,store)
        self.modify_hamiltonian_matrices(f) # modify all the matrices
    def get_filling(self,**kwargs):
        """Get the filling of a Hamiltonian at this energy"""
        return spectrum.get_filling(self,**kwargs) # eigenvalues
    def project_interactions(self,**kwargs):
        """Project interactions"""
        from .interactions.vijkl import Vijkl
        return Vijkl(self,**kwargs)
    def reduce(self):
        return hamiltonianmode.reduce_hamiltonian(self)

    def full2profile(self,x,**kwargs):
        """Transform a 1D array in the full space to the spatial basis"""
        from .htk.matrixcomponent import full2profile
        return full2profile(self,x,**kwargs)

    def get_chern(h,**kwargs):
        return topology.chern(h,**kwargs)

    def get_hopping_dict(self):
        """Return the dictionary with the hoppings"""
        return multicell.get_hopping_dict(self)

    def get_multihopping(self):
        out = multicell.get_hopping_dict(self)
        return multicell.MultiHopping(out) # return the object

    def set_multihopping(self,mh):
        """Set a multihopping as the Hamiltonian"""
        multicell.set_multihopping(self,mh)

    @get_docstring(spectrum.set_filling)
    def set_filling(self,filling,**kwargs):
        spectrum.set_filling(self,filling=filling,**kwargs)

    def __init__(self,geometry=None):
      self.data = dict() # empty dictionary with various data
      self.has_spin = True # has spin degree of freedom
      self.prefix = "" # a string used a prefix for different files
      self.path = "" # a path used for different files
      self.has_eh = False # has electron hole pairs
      self.get_eh_sector = None # no function for getting electrons
      self.fermi_energy = 0.0 # fermi energy at zero
      self.dimensionality = 0 # dimensionality of the Hamiltonian
      self.temperature = 0.0 # temperature of the Hamiltonian
      self.is_sparse = False
      self.is_multicell = False # for hamiltonians with hoppings to several neighbors
      self.hopping_dict = {} # hopping dictonary
      self.has_hopping_dict = False # has hopping dictonary
      self.non_hermitian = False # non hermitian Hamiltonian
      if not geometry is None:
  # dimensionality of the system
        self.dimensionality = geometry.dimensionality 
        self.geometry = geometry # add geometry object
        self.num_orbitals = len(geometry.x)
    def get_hk_gen(self):
        """ Generate kdependent hamiltonian"""
        if self.is_multicell: return multicell.hk_gen(self) # for multicell
        else: return hk_gen(self) # for normal cells

    def has_time_reversal_symmetry(self):
        """Check if a Hamiltonian has time reversal symmetry"""
        from .htk import symmetry
        return symmetry.has_time_reversal_symmetry(self)

    def get_qpi(self,**kwargs):
        from .chitk import qpi
        return qpi.get_qpi(self,**kwargs)

    @get_docstring(ldos.get_ldos)
    def get_ldos(self,**kwargs):
        return ldos.get_ldos(self,**kwargs)
    def get_gk_gen(self,delta=0.05,operator=None,canonical_phase=False):
      """Return the Green function generator"""
      hkgen = self.get_hk_gen() # Hamiltonian generator
      def f(k=[0.,0.,0.],e=0.0):
        hk = hkgen(k) # get matrix
        if canonical_phase: # use a Bloch phase in all the sites
          frac_r = self.geometry.frac_r # fractional coordinates
          # start in zero
          U = np.diag([self.geometry.bloch_phase(k,r) for r in frac_r])
          U = np.matrix(U) # this is without .H
          U = self.spinless2full(U) # increase the space if necessary
          Ud = np.conjugate(U.T) # dagger
          hk = Ud@hk@U
  #        print(csc_matrix(np.angle(hk)))
  #        exit()
        if operator is not None: 
            hk = algebra.dagger(operator)@hk@operator # project
        out = algebra.inv(np.identity(hk.shape[0])*(e+1j*delta) - hk)
        return out
      return f
    def to_canonical_gauge(self,m,k):
        """Return a matrix in the canonical gauge"""
        from . import gauge
        return gauge.to_canonical_gauge(self,m,k) # return the matrix
    def print_hamiltonian(self):
        """Print hamiltonian on screen"""
        print_hamiltonian(self)
    def check_mode(self,n):
        """Verify the type of Hamiltonian"""
        return hamiltonianmode.check_mode(self,n)
    def diagonalize(self,nkpoints=100):
      """Return eigenvalues"""
      return diagonalize(self,nkpoints=nkpoints)
    def get_fermi4filling(self,filling,**kwargs):
        """Return the fermi energy for a certain filling"""
        return spectrum.get_fermi4filling(self,filling,**kwargs)
    def get_dos(self,**kwargs):
        from . import dos
        return dos.get_dos(self,**kwargs)
    def get_bands(self,**kwargs):
      """ Returns a figure with teh bandstructure"""
      return get_bands_nd(self,**kwargs)
    def add_sublattice_imbalance(self,mass):
      """ Adds a sublattice imbalance """
      if self.geometry.has_sublattice and self.geometry.sublattice_number==2:
        add_sublattice_imbalance(self,mass)
      else: pass
    def add_antiferromagnetism(self,mass):
        """ Adds antiferromagnetic imbalanc """
        if self.geometry.has_sublattice:
            if self.geometry.sublattice_number==2:
                magnetism.add_antiferromagnetism(self,mass)
            elif self.geometry.sublattice_number>2:
                magnetism.add_frustrated_antiferromagnetism(self,mass)
            else: raise
        else: return 
    def turn_nambu(self):
        """Add electron hole degree of freedom"""
        self.get_eh_sector = get_eh_sector_odd_even # assign function
        superconductivity.turn_nambu(self)
    def add_swave(self,*args,**kwargs):
        """ Adds swave superconducting pairing"""
        superconductivity.add_swave_to_hamiltonian(self,*args,**kwargs)
    def setup_nambu_spinor(self): self.add_swave(0.0)
    def get_anomalous_hamiltonian(self):
        """Return a Hamiltonian only with the anomalous part"""
        return superconductivity.get_anomalous_hamiltonian(self)
    def add_pairing(self,**kwargs):
      """ Add a general pairing matrix, uu,dd,ud"""
      superconductivity.add_pairing_to_hamiltonian(self,**kwargs)
    def same_hamiltonian(self,*args,**kwargs):
        """Check if two hamiltonians are the same"""
        return hamiltonianmode.same_hamiltonian(self,*args,**kwargs)
    def supercell(self,nsuper):
      """ Creates a supercell of a one dimensional system"""
      if nsuper==1: return self
      if self.dimensionality==0: return self
      try: 
          nsuper[2] 
          ns = nsuper # array as input
      except:
          if self.dimensionality==1: ns = [nsuper,1,1]
          elif self.dimensionality==2: ns = [nsuper,nsuper,1]
          elif self.dimensionality==3: ns = [nsuper,nsuper,nsuper]
          else: raise
      return multicell.supercell_hamiltonian(self,nsuper=ns)
    def set_finite_system(self,periodic=True):
      """ Transforms the system into a finite system"""
      return set_finite_system(self,periodic=periodic) 
    def get_gap(self,**kwargs):
      """Returns the gap of the Hamiltonian"""
      from . import gap
      return gap.indirect_gap(self,**kwargs) # return the gap
    def save(self,output_file="hamiltonian.pkl"):
      """ Write the hamiltonian in a file"""
      inout.save(self,output_file) # write in a file
    write = save # just in case
    def read(self,output_file="hamiltonian.pkl"):
      """ Read the Hamiltonian"""
      return load(output_file) # read Hamiltonian

    def load(self,**kwargs): return self.read(**kwargs)

    @get_docstring(spectrum.total_energy)
    def get_total_energy(self,**kwargs):
      return spectrum.total_energy(self,**kwargs)

    def total_energy(self,**kwargs): return self.get_total_energy(**kwargs)

    def add_zeeman(self,zeeman):
        """Adds zeeman to the matrix """
        self.turn_spinful()
        from .magnetism import add_zeeman
        add_zeeman(self,zeeman=zeeman)
    def add_magnetism(self,m):
        """Adds magnetism, new version of zeeman"""
        self.turn_spinful()
        from .magnetism import add_magnetism
        add_magnetism(self,m)
    def add_exchange(self,m): self.add_magnetism(m)
    def turn_spinful(self,enforce_tr=False):
      """Turn the hamiltonian spinful""" 
      if self.has_spin: return # already spinful
      if self.is_sparse: # sparse Hamiltonian
        self.turn_dense() # dense Hamiltonian
        self.turn_spinful(enforce_tr=enforce_tr) # spinful
        self.turn_sparse()
      else: # dense Hamiltonian
        from .increase_hilbert import spinful
        def fun(m):
            if enforce_tr: return spinful(m,np.conjugate(m))
            else: return spinful(m)
        self.modify_hamiltonian_matrices(fun) # modify the matrices
        self.has_spin = True # set spinful
    def remove_spin(self):
      """Removes spin degree of freedom"""
      if self.check_mode("spinless"): return # do nothing
      elif self.check_mode("spinful"):
          def f(m): return des_spin(m,component=0)
          self.modify_hamiltonian_matrices(f) # modify the matrices
          self.has_spin = False # set to spinless
      else: raise
    def remove_nambu(self):
      if self.check_mode("spinful_nambu"): 
          def f(m):
              return superconductivity.get_eh_sector(m,i=0,j=0)
          self.modify_hamiltonian_matrices(f) # modify the matrices
          self.has_eh = False # set to normal
      elif self.check_mode("spinful"): pass
      elif self.check_mode("spinless"): pass
      else: raise
    def add_onsite(self,fermi):
      """ Move the Fermi energy of the system"""
      shift_fermi(self,fermi)
    def get_topological_invariant(self,**kwargs):
        """Return a topological invariant"""
        if self.dimensionality==0: pass
        elif self.dimensionality==1: pass
        elif self.dimensionality==2: 
            if self.has_time_reversal_symmetry():
                return topology.z2_invariant(self,**kwargs)
            else:
                print("Computing Chern")
                return topology.chern(self,**kwargs)
        else: raise
    def shift_fermi(self,fermi): self.add_onsite(fermi)  
    def first_neighbors(self):
      """ Create first neighbor hopping"""
      if 0<=self.dimensionality<3:
        first_neighborsnd(self)
      elif self.dimensionality == 3:
        from .multicell import first_neighbors as fnm
        fnm(self)
      else: raise
    def add_hopping_matrix(self,fm):
        """
        Add a certain hopping matrix to the Hamiltonian
        """
        if not self.is_multicell: raise # this may not work for multicell
        h = self.geometry.get_hamiltonian(has_spin=self.has_spin,
                is_multicell=self.is_multicell,
                mgenerator=fm) # generate a new Hamiltonian
        self.add_hamiltonian(h) # add this contribution
    def add_hamiltonian(self,h):
        """
        Add the hoppings of another Hamiltonian
        """
        if not self.is_multicell: raise # not implemented
        hd = h.get_dict() # get the dictionary
        self.intra = self.intra + hd[(0,0,0)] # add the matrix
        for i in range(len(self.hopping)):
            d = tuple(self.hopping[i].dir)
            if d in hd:
              self.hopping[i].m = self.hopping[i].m + hd[d]
    def get_dict(self):
        """
        Return the dictionary that yields the hoppings
        """
        if not self.is_multicell: # not implemented
            self = self.get_multicell()
        hop = dict()
        hop[(0,0,0)] = self.intra
        for t in self.hopping: hop[tuple(np.array(t.dir))] = t.m
        for t in self.hopping: hop[tuple(-np.array(t.dir))] = np.conjugate(t.m).T
        return hop # return dictionary

    @get_docstring(ldos.multi_ldos)
    def get_multildos(self,**kwargs):
        return ldos.multi_ldos(self,**kwargs)
    def get_multihopping(self):
        """Return a multihopping object"""
        from .multihopping import MultiHopping
        return MultiHopping(self.get_dict())

    @get_docstring(Vinteraction)
    def get_mean_field_hamiltonian(self,**kwargs):
        return Vinteraction(self,**kwargs).hamiltonian
    def copy(self):
        """
        Return a copy of the hamiltonian
        """
        from copy import deepcopy
        return deepcopy(self)
    def check(self,**kwargs):
        """
        Check if the Hamiltonian is OK
        """
        from . import check
        check.check_hamiltonian(self,**kwargs) # check the Hamiltonian
    def enforce_eh(self):
        """Enforce electron-hole symmetry in the Hamiltonian"""
        self.turn_multicell() # turn to multicell mode
        from superconductivity import eh_operator
        f = eh_operator(self.intra) # electron hole operator
        raise # not implemented
    def turn_sparse(self):
        """
        Transforms the hamiltonian into a sparse hamiltonian
        """
        from scipy.sparse import csc_matrix
        def f(m):
            return csc_matrix(m)
        self.modify_hamiltonian_matrices(f) # modify the matrices
        self.is_sparse = True # sparse flag to true
    def turn_dense(self):
      """ Transforms the hamiltonian into a sparse hamiltonian"""
      def f(m):
          return algebra.todense(m)
      self.modify_hamiltonian_matrices(f) # modify the matrices
      self.is_sparse = False # sparse flag to true
    def add_rashba(self,c):
      """Adds Rashba coupling"""
      from . import rashba
      rashba.add_rashba(self,c)
    def add_kane_mele(self,t,**kwargs):
      """ Adds a Kane-Mele SOC term"""  
      kanemele.add_kane_mele(self,t,**kwargs) # return kane-mele SOC
    def add_haldane(self,t):
      """ Adds a Haldane term"""  
      kanemele.add_haldane(self,t) # return Haldane SOC
    def add_kekule(self,t):
        """
        Add Kekule coupling
        """
        if self.dimensionality==0: # zero dimensional
          m = kekule.kekule_matrix(self.geometry.r,t=t)
          self.intra = self.intra + self.spinless2full(m)
        else: # workaround for higher dimensionality
          r = self.geometry.multireplicas(2) # get many replicas
          fm = kekule.kekule_function(r,t=t)
          self.add_hopping_matrix(fm) # add the Kekule hopping
    def add_chiral_kekule(self,**kwargs):
        """
        Add a chiral kekule hopping
        """
        fun = kekule.chiral_kekule(self.geometry,**kwargs)
        self.add_kekule(fun)
  
    def add_modified_haldane(self,t):
        """
        Adds a Haldane term
        """  
        kanemele.add_modified_haldane(self,t) # return Haldane SOC
    def add_anti_kane_mele(self,t):
        """
        Adds an anti Kane-Mele term
        """  
        kanemele.add_anti_kane_mele(self,t) # return anti kane mele SOC
    def add_antihaldane(self,t): 
        """Add an anti-Haldane term"""
        self.add_modified_haldane(t) # second name
    def add_crystal_field(self,v):
        """Add a crystal field term to the Hamiltonian"""
        from . import crystalfield
        crystalfield.hartree(self,v=v) 
    def add_peierls(self,mag_field,**kwargs):
        """
        Add magnetic field
        """
        from .peierls import add_peierls
        add_peierls(self,mag_field=mag_field,**kwargs)
    def add_inplane_bfield(self,**kwargs):
        """Add in-plane magnetic field"""
        from .peierls import add_inplane_bfield
        add_inplane_bfield(self,**kwargs)
    def align_magnetism(self,vectors):
        """ Rotate the Hamiltonian to have magnetism in the z direction"""
        if self.has_eh: raise
        from .rotate_spin import align_magnetism as align
        self.intra = align(self.intra,vectors)
        self.inter = align(self.inter,vectors)
    def global_spin_rotation(self,**kwargs):
        """ Perform a global spin rotation """
        return rotate_spin.hamiltonian_spin_rotation(self,**kwargs)
    def generate_spin_spiral(self,**kwargs):
        """ Generate a spin spiral antsaz in the Hamiltonian """
        return rotate_spin.generate_spin_spiral(self,**kwargs)
    def get_magnetization(self,**kwargs):
        """ Return the magnetization """
        mx = self.extract(name="mx")
        my = self.extract(name="my")
        mz = self.extract(name="mz")
        return np.array([mx,my,mz]).T # return array
    def compute_vev(self,name="sz",**kwargs):
        """
        Compute a VEV of a spatially resolved operator
        """
        n = len(self.geometry.r) # number of sites
        ops = [operators.index(self,n=[i]) for i in range(n)]
        if name=="sx": op = operators.get_sx(self)
        elif name=="sy": op = operators.get_sy(self)
        elif name=="sz": op = operators.get_sz(self)
        elif name=="density": op = operators.index(self,n=range(n))
        else: raise
        ops = [o@op for o in ops] # define operators
        return spectrum.ev(self,operator=ops,**kwargs).real
    def get_1dh(self,k=0.0):
        """Return a 1d Hamiltonian"""
        if self.is_multicell: # not implemented
            self = self.get_no_multicell() # return the no multicell Hamiltonian
        if not self.dimensionality==2: raise # not implemented
        intra,inter = kchain(self,k) # generate intra and inter
        hout = self.copy() # copy the Hamiltonian
        hout.intra = intra # store
        hout.inter = inter # store
        hout.dimensionality = 1 # one dimensional
        hout.geometry.dimensionality = 1 # one dimensional
        return hout
    def get_multicell(self):
        """Return a multicell Hamiltonian"""
        return multicell.turn_multicell(self)
    def turn_multicell(self):
        """Conver to multicell Hamiltonian"""
        h = multicell.turn_multicell(self)
        self.is_multicell = True
        self.hopping = h.hopping # assign hopping
    def get_no_multicell(self):
        """Return a multicell Hamiltonian"""
        h1 = multicell.turn_no_multicell(self)
        h0 = h1.get_multicell() # turn multicell again
        diff = (self.get_multihopping() - h0.get_multihopping()).norm()
        if diff>1e-6: 
            print("Hamiltonain cannot be made no multicell")
            raise
        else: return h1 # return the Hamiltonian
    def clean(self):
        """Clean a Hamiltonian"""
        from .clean import clean_hamiltonian
        clean_hamiltonian(self)
    def get_operator(self,name,**kwargs):
        """Return a certain operator"""
        from . import operatorlist
        return operators.object2operator(operatorlist.get_operator(self,name,
            **kwargs))
    def extract(self,name): 
        """Extract something from the Hamiltonian"""
        return extract.extract_from_hamiltonian(self,name)
    @get_docstring(dvector.dvector_non_unitarity_map)
    def write_non_unitarity(self,**kwargs):
        dvector.dvector_non_unitarity_map(self,**kwargs)
    def write_magnetization(self,nrep=5):
        """Extract the magnetization and write it in a file"""
        mx = self.extract("mx")
        my = self.extract("my")
        mz = self.extract("mz")
        g = self.geometry
        g.write_profile(mx,name="MX.OUT",normal_order=True,nrep=nrep)
        g.write_profile(my,name="MY.OUT",normal_order=True,nrep=nrep)
        g.write_profile(mz,name="MZ.OUT",normal_order=True,nrep=nrep)
        # this is just a workaround
        m = np.genfromtxt("MX.OUT").transpose()
        (x,y,z,mx) = m[0],m[1],m[2],m[3]
        my = np.genfromtxt("MY.OUT").transpose()[3]
        mz = np.genfromtxt("MZ.OUT").transpose()[3]
        np.savetxt("MAGNETISM.OUT",np.array([x,y,z,mx,my,mz]).T)
        return np.array([mx,my,mz])
    def write_onsite(self,nrep=5,normal_order=False):
        """Extract onsite energy"""
        d = self.extract("density")
        d = d - np.mean(d)
        self.geometry.write_profile(d,name="ONSITE.OUT",
                normal_order=normal_order,nrep=nrep)
    def write_hopping(self,**kwargs):
        groundstate.hopping(self,**kwargs)
    def write_anomalous_hopping(self,**kwargs):
        groundstate.anomalous_hopping(self,**kwargs)
    def write_swave(self,**kwargs):
        """Write the swave pairing"""
        groundstate.swave(self,**kwargs)
    def get_ipr(self,**kwargs):
        """Return the IPR"""
        from . import ipr
        if self.dimensionality==0:
            return ipr.ipr(self.intra,**kwargs) 
        else: raise # not implemented
    @get_docstring(dvector.dvector_non_unitarity)
    def get_dvector_non_unitarity(self,**kwargs):
        return dvector.dvector_non_unitarity(self,**kwargs)
    def get_density_matrix(self,**kwargs):
        """Return the density matrix"""
        from . import densitymatrix
        return densitymatrix.full_dm(self,**kwargs)
    @get_docstring(superconductivity.average_hamiltonian_dvector)
    def get_average_dvector(self,**kwargs):
        return superconductivity.average_hamiltonian_dvector(self,**kwargs)


hamiltonian = Hamiltonian





def get_first_neighbors(r1,r2):
    """Gets the fist neighbors, input are arrays"""
#  if optimal:
    from . import neighbor
    pairs = neighbor.find_first_neighbor(r1,r2)
    return pairs
#  else:
#    from numpy import array
#    n=len(r1)
#    pairs = [] # pairs of neighbors
#    for i in range(n):
#      ri=r1[i]
#      for j in range(n):
#        rj=r2[j]
#        dr = ri - rj ; dr = dr.dot(dr)
#        if .9<dr<1.1 : # check if distance is 1
#          pairs.append([i,j])  # add to the list
#    return pairs # return pairs of first neighbors

















def create_fn_hopping(r1,r2):
  n=len(r1)
  mat=np.matrix([[0.0j for i in range(n)] for j in range(n)])
  pairs = get_first_neighbors(r1,r2) # get pairs of first neighbors
  for p in pairs: # loop over pairs
    mat[p[0],p[1]] = 1.0 
  return mat



# function to calculate the chirality between two vectors (vectorial productc)
def vec_chi(r1,r2):
  """Return clockwise or anticlockwise"""
  z=r1[0]*r2[1]-r1[1]*r2[0]
  zz = r1-r2
  zz = sum(zz*zz)
  if zz>0.01:
    if z>0.01: # clockwise
      return 1.0
    if z<-0.01: # anticlockwise
      return -1.0
  return 0.0




# routine to check if two atoms arein adistance d
def is_neigh(r1,r2,d,tol):
  r=r1-r2
  x=r[0]
  y=r[1]
  dt=abs(d*d-x*x-y*y)
  if dt<tol:
    return True
  return False




#################################3

def diagonalize(h,nkpoints=100):
  """ Diagonalice a hamiltonian """
  import scipy.linalg as lg
  # for one dimensional systems
  if h.dimensionality==1:  # one simensional system
    klist = np.arange(0.0,1.0,1.0/nkpoints)  # create array with the klist
    if h.geometry.shift_kspace:
      klist = np.arange(-0.5,0.5,1.0/nkpoints)  # create array with the klist
    intra = h.intra  # assign intraterm
    inter = h.inter  # assign interterm
    energies = [] # list with the energies
    for k in klist: # loop over kpoints
      bf = np.exp(1j*np.pi*2.*k)  # bloch factor for the intercell terms
      inter_k = inter*bf  # bloch hopping
      hk = intra + inter_k + inter_k.H # k dependent hamiltonian
      energies += [lg.eigvalsh(hk)] # get eigenvalues of the current hamiltonian
    energies = np.array(energies).transpose() # each kpoint in a line
    return (klist,energies) # return the klist and the energies
# for zero dimensional systems system
  elif h.dimensionality==0:  
    intra = h.intra  # assign intraterm
    energies = lg.eigvalsh(intra) # get eigenvalues of the current hamiltonian
    return (range(len(intra)),energies) # return indexes and energies
  else: raise




def diagonalize_hk(k):
  return algebra.eigh(hk(k))




def diagonalize_kpath(h,kpath):
  """Diagonalice in a certain path"""
  energies = [] # empty list with energies
  import scipy.linalg as lg
  ik = 0.
  iks = [] # empty list
  for k in kpath:
    f = h.get_hk_gen() # get Hk generator
    hk = f(k)  # k dependent hamiltonian
    es = (lg.eigvalsh(hk)).tolist() # get eigenvalues for current hamiltonian
    energies += es # append energies 
    iks += [ik for i in es]
    ik += 1.
  iks = np.array(iks)
  iks = iks/max(iks) # normalize path
  return (iks,energies)





def print_hamiltonian(h):
  """ Print the hamilotnian on screen """
  from scipy.sparse import coo_matrix as coo # import sparse matrix
  intra = coo(h.intra) # intracell
  inter = coo(h.inter) # intracell
  print("Intracell matrix")
  print(intra)
  print("Intercell matrix")
  print(inter)
  return



def add_sublattice_imbalance(h,mass):
  """ Adds to the intracell matrix a sublattice imbalance """
  if h.geometry.has_sublattice:  # if has sublattice
    def ab(i): 
      return h.geometry.sublattice[i]
  else: 
    print("WARNING, no sublattice present")
    return 0. # if does not have sublattice
  natoms = len(h.geometry.r) # assume spinpolarized calculation 
  rows = range(natoms)
  if callable(mass):  # if mass is a function
    r = h.geometry.r
    data = [mass(r[i])*ab(i) for i in range(natoms)]
  else: data = [mass*ab(i) for i in range(natoms)]
  massterm = csc_matrix((data,(rows,rows)),shape=(natoms,natoms)) # matrix
  h.intra = h.intra + h.spinless2full(massterm)








def build_eh_nonh(hin,c1=None,c2=None):
  """Creates a electron hole matrix, from an input matrix, coupling couples
     electrons and holes
      - hin is the hamiltonian for electrons, which has the usual common form
      - coupling is the matrix which tells the coupling between electron
        on state i woth holes on state j, for exmaple, with swave pairing
        the non vanishing elments are (0,1),(2,3),(4,5) and so on..."""
  n = len(hin)  # dimension of input
  nn = 2*n  # dimension of output
  hout = np.matrix(np.zeros((nn,nn),dtype=complex))  # output hamiltonian
  for i in range(n):
    for j in range(n):
      hout[2*i,2*j] = hin[i,j]  # electron term
      hout[2*i+1,2*j+1] = -np.conjugate(hin[i,j])  # hole term
  if not c1 is None: # if there is coupling
    for i in range(n):
      for j in range(n):
        # couples electron in i with hole in j
        hout[2*i,2*j+1] = c1[i,j]  # electron hole term
  if not c2 is None: # if there is coupling
    for i in range(n):
      for j in range(n):
        # couples hole in i with electron in j
        hout[2*j+1,2*i] = np.conjugate(c2[i,j])  # hole electron term
  return hout 









def set_finite_system(hin,periodic=True):
  """ Transforms the hamiltonian into a finite system,
  removing the hoppings """
  from copy import deepcopy
  h = hin.copy() # copy Hamiltonian
  h.dimensionality = 0 # put dimensionality = 0
  if periodic: # periodic boundary conditions
    if h.dimensionality == 1:
      h.intra = h.intra + h.inter + h.inter.H 
    if h.dimensionality == 2:
      h.intra = h.intra +  h.tx + h.tx.H 
      h.intra = h.intra +  h.ty + h.ty.H 
      h.intra = h.intra +  h.txy + h.txy.H 
      h.intra = h.intra +  h.txmy + h.txmy.H 
  else: pass
  return h
  
# remove spin degree of freedom
des_spin = increase_hilbert.des_spin


def shift_fermi(h,fermi):
  """ Moves the fermi energy of the system, the new value is at zero"""
  r = h.geometry.r # positions
  n = len(r) # number of sites
  if checkclass.is_iterable(fermi): # iterable
    if len(fermi)==n: # same number of sites
      h.intra = h.intra + h.spinless2full(sparse_diag([fermi],[0]))
    else: raise
  else:
    rc = [i for i in range(n)]  # index
    datatmp = [] # data
    for i in range(n): # loop over positions 
      if callable(fermi): fshift = fermi(r[i]) # fermi shift
      else: fshift = fermi # assume it is a number
      datatmp.append(fshift) # append value
    m = csc_matrix((datatmp,(rc,rc)),shape=(n,n)) # matrix with the shift
    h.intra = h.intra + h.spinless2full(m) # Add matrix 
    return


import numbers

def is_number(s):
    return isinstance(x, numbers.Number)

def is_hermitic(m):
  mh = np.conjugate(m).T
  hh = m - mh
  for i in range(len(hh)):
    for j in range(len(hh)):
      if np.abs(hh[i,j]) > 1e-5:
        print("No hermitic element", i,j,m[i,j],m[j,i])
        return False
  return True
  





def first_neighborsnd(h):
  """ Gets a first neighbor hamiltonian"""
  r = h.geometry.r    # x coordinate 
  g = h.geometry
# first neighbors hopping, all the matrices
  a1, a2 = g.a1, g.a2
  def gett(r1,r2):
    """Return hopping given two sets of positions"""
    from . import neighbor
    pairs = neighbor.find_first_neighbor(r1,r2)
    if len(pairs)==0: rows,cols = [],[]
    else: rows,cols = np.array(pairs).T # transpose
    data = np.array([1. for c in cols])
    n = len(r1)
    m = csc_matrix((data,(rows,cols)),shape=(n,n),dtype=np.complex)
    m = h.spinless2full(m) # add spin degree of freedom if necessary
    if h.is_sparse: return m
    else: return m.todense()
  if h.dimensionality==0:
    h.intra = gett(r,r)
  elif h.dimensionality==1:
    h.intra = gett(r,r)
    h.inter = gett(r,r+a1)
  elif h.dimensionality==2:
    h.intra = gett(r,r)
    h.tx = gett(r,r+a1)
    h.ty = gett(r,r+a2)
    h.txy = gett(r,r+a1+a2)
    h.txmy = gett(r,r+a1-a2)
  else: raise



from .superconductivity import add_swave
from .superconductivity import build_eh
nambu_nonh = build_eh_nonh



from .bandstructure import lowest_bands



def hk_gen(h):
  """ Returns a function that generates a k dependent hamiltonian"""
  if h.dimensionality == 0: return lambda x: h.intra
  elif h.dimensionality == 1: 
    def hk(k):
      """k dependent hamiltonian, k goes from 0 to 1"""
      try: kp = k[0]
      except: kp = k
      tk = h.inter * h.geometry.bloch_phase([1.],kp) # get the bloch phase
      ho = h.intra + tk + algebra.dagger(tk) # hamiltonian
      return ho
    return hk  # return the function
  elif h.dimensionality == 2: 
    def hk(k):
      """k dependent hamiltonian, k goes from (0,0) to (1,1)"""
      if len(k)==3:
        k = np.array([k[0],k[1]]) # redefine for 2d
      k = np.array(k)
      ux = np.array([1.,0.])
      uy = np.array([0.,1.])
      ptk = [[h.tx,ux],[h.ty,uy],[h.txy,ux+uy],[h.txmy,ux-uy]] 
      ho = (h.intra).copy() # intraterm
      for p in ptk: # loop over hoppings
#        tk = p[0]*np.exp(1j*np.pi*2.*(p[1].dot(k)))  # add bloch hopping
        tk = p[0]*h.geometry.bloch_phase(p[1],k)  # add bloch hopping
        ho = ho + tk + algebra.dagger(tk)  # add bloch hopping
      return ho
    return hk
  else: raise



from .neighbor import parametric_hopping
from .neighbor import parametric_hopping_spinful
from .neighbor import generate_parametric_hopping




from .superconductivity import get_nambu_tauz
from .superconductivity import project_electrons
from .superconductivity import project_holes
from .superconductivity import get_eh_sector_odd_even




def kchain(h,k):
  """ Return the kchain Hamiltonian """
  if h.dimensionality != 2: raise
  if h.is_multicell: h = h.get_no_multicell() # redefine
  tky = h.ty*np.exp(1j*np.pi*2.*k)
  tkxy = h.txy*np.exp(1j*np.pi*2.*k)
  tkxmy = h.txmy*np.exp(-1j*np.pi*2.*k)  # notice the minus sign !!!!
  # chain in the x direction
  ons = h.intra + tky + np.conjugate(tky).T  # intra of k dependent chain
  hop = h.tx + tkxy + tkxmy  # hopping of k-dependent chain
  return (ons,hop)



# import the function written in the library
from .kanemele import generalized_kane_mele


def modify_hamiltonian_matrices(self,f):
  """Apply a certain function to all the matrices"""
  self.intra = f(self.intra)
  if self.is_multicell: # for multicell hamiltonians
    for i in range(len(self.hopping)): # loop over hoppings
      self.hopping[i].m = f(self.hopping[i].m) # put in nambu form
  else: # conventional way
    if self.dimensionality==0: pass # one dimensional systems
    elif self.dimensionality==1: # one dimensional systems
      self.inter = f(self.inter)
    elif self.dimensionality==2: # two dimensional systems
      self.tx = f(self.tx)
      self.ty = f(self.ty)
      self.txy = f(self.txy)
      self.txmy = f(self.txmy)
    else: raise



from . import inout

def load(input_file="hamiltonian.pkl"):  return inout.load(input_file)


def print_hopping(h):
    """Print all the hoppings in a user friendly way"""
    from pandas import DataFrame
    def pprint(m): 
        if np.max(np.abs(m.real))>0.00001: print(DataFrame(m.real))
        if np.max(np.abs(m.imag))>0.00001: print(DataFrame(m.imag*1j))
        print("\n")
    print("Onsite")
    pprint(h.intra)
    if h.dimensionality==0: return
    h = h.get_multicell()
    for t in h.hopping:
        print("Hopping",t.dir)
        pprint(t.m)







