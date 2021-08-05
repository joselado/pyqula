from __future__ import print_function
import numpy as np
from scipy.sparse import coo_matrix, bmat, csc_matrix
import scipy.sparse as sp
from scipy.sparse import issparse
from . import algebra



def get_eh_sector_odd_even(m,i=0,j=0):
  """ Return the electron hole sector of a matrix,
  assumming that the matrix is in full nambu form""" 
  if i>1 or j>1: return NotImplemented # meaningless
  m = nambu2block(m) # reorder the matrix
  n = m.shape[0]//2 # number of orbitals
  if i==0 and j==0: return m[0:n,0:n] 
  elif i==1 and j==0: return m[n:2*n,0:n] 
  elif i==0 and j==1: return m[0:n,n:2*n] 
  elif i==1 and j==1: return m[n:2*n,n:2*n] 
  else: raise


get_eh_sector = get_eh_sector_odd_even

def get_electron_sector(m):
    """Return the electron sector"""
    return get_eh_sector_odd_even(m,i=0,j=0)


def get_anomalous_sector(m):
    """Return the anomalous electron-hole sector"""
    return get_eh_sector_odd_even(m,i=0,j=1)


def get_second_anomalous_sector(m):
    """Return the anomalous electron-hole sector"""
    return get_eh_sector_odd_even(m,i=1,j=0)


def get_nambu_tauz(m,has_eh=False):
  """Return the nambu matrix tauz for electron-hole"""
  raise # this function is not consistent with the Nambu notation (see below)
  n = m.shape[0] # number of sites 
  if has_eh: n = n//2 # half
  mout = np.matrix(np.zeros((n*2,n*2)),dtype=np.complex) # define matrix
  for ii in range(n): # loop over index
    mout[2*ii,2*ii] = 1. # assign
    mout[2*ii+1,2*ii+1] = -1. # assign
  return mout # return tauz



def project_electrons(m):
  """Return the nambu matrix tauz for electron-hole"""
  n = m.shape[0] # number of sites 
  mout = m*0.0 # define matrix
  for ii in range(n): # loop over index
    for jj in range(n): # loop over index
      if ii%4<2 and jj%4<2:  mout[ii,jj] = m[ii,jj] # assign
      else: continue
  return mout # return tauz



def project_holes(m):
  """Return the nambu matrix tauz for electron-hole"""
  n = m.shape[0] # number of sites 
  mout = m*0.0 # define matrix
  for ii in range(n): # loop over index
    for jj in range(n): # loop over index
      if ii%4>1 and jj%4>1:  mout[ii,jj] = m[ii,jj] # assign
      else: continue
  return mout # return tauz




def eh_operator(m):
  """Return the electron hole symmetry operator, as a function"""
  n = m.shape[0]//4 # number of sites 
  from .hamiltonians import sy
  msy = [[None for ri in range(n)] for j in range(n)]
  for i in range(n): msy[i][i] = sy # add sy
  msy = bmat(msy) # sy matrix in the electron subspace
  out = [[None,1j*msy],[-1j*msy,None]]
  out = bmat(out) # sparse matrix
  out = reorder(out) # reshufle the matrix
  def ehop(inm):
    """Function that applies electron-hole symmetry to a matrix"""
    return out@np.conjugate(inm)@out # return matrix
  return ehop # return operator
    

def enforce_multihopping_eh_symmetry(MH):
    """Enforce electron-hole symmetry in a multihopping object"""
    raise
    from .multihopping import MultiHopping
    dd = MH.get_dict() # get the dictionary
    out = dict() # dictionary
    feh = eh_operator(dd[(0,0,0)]) # eh operator
    for key in dd: # loop
        m = dd[key] + feh(dd[tuple(-np.array(key))])
        m = m/2. # normalize
        out[key] = m
    return MultiHopping(out) # return object


def enforce_eh_symmetry(cc,cdcd):
    """Enforce electron-hole symmetry, given the two dictionaries
    for the expectation values the anomalous expectation values"""
    outcc = dict() # dictionary
    outcdcd= dict() # dictionary
    for key in cc: # loop over keys
        keym = tuple(-np.array(key)) # the opposite direction
        m = cc[key] # get this correlator (the cc one)
        m2 = cdcd[keym] # get the other matrix
        md = algebra.dagger(m) # get the dagger
        m2d = algebra.dagger(m2) # get the dagger of the matrix
 #       print("MATRICES")
#        print(np.round(m,2))
#        print(np.round(m2,2))
 #       print("MATRICES")
        # this enforces even superconductivity
        outcc[key] = (m + m2d)/2.
        outcdcd[keym] = (m2+md)/2.
    return (outcc,outcdcd) # return the dictionaries





#    raise # this is not ok
#    if type(m)==dict: 
#        from .multihopping import MultiHopping
#        m = MultiHopping(m)
#        return enforce_multihopping_eh_symmetry(m).get_dict()
#    else: raise




def non_unitary_pairing(v):
  """Calculate the vector that defines the non-unitary pairing,
  the input is the three components of the pairing matrix"""
  vc = [np.conjugate(v[i]) for i in range(3)]
  ux = v[1]*vc[2] - v[2]*vc[1]
  uy = v[2]*vc[0] - v[0]*vc[2]
  uz = v[0]*vc[1] - v[1]*vc[0]
  return (1j*np.array([ux,uy,uz])).real



# from now on, the spinors will be
# Psi =
# Psi_up
# Psi_dn
# Psi^dag_dn
# -Psi^dag_up


spair = csc_matrix([[0.,0.,1.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
proje = csc_matrix([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
projh = csc_matrix([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
#pzpair = csc_matrix([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
#pxpair = csc_matrix([[0.,0.,1.,0.],[0.,0.,0.,-1.],[0.,0.,0.,0.],[-0.,0.,0.,0.]])
#pypair = csc_matrix([[0.,0.,-1j,0.],[0.,0.,0.,1j],[0.,0,0.,0.],[0,0.,0.,0.]])
# Definition using Manfred's notes
# This are the operators that give the d-vectors
# Minus sign due to the nambu spinor!
deltauu = csc_matrix([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[-1.,0.,0.,0.]])
# and the rest are simple
deltadd = csc_matrix([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]])
deltauu = deltauu.H
deltadd = deltadd.H
deltax = (deltadd - deltauu)/2.
deltay = (deltadd + deltauu)/2j
# this one is tricky, we only want the antisymmetric part
deltaz = csc_matrix([[0.,0.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,-1.,0.,0.]])/2.

deltaz = deltaz.H



# redefine the functions according to the previous notation

def time_reversal(m):
  """Do the spinful time reversal of this matrix"""
  from .hamiltonians import sy
  n = m.shape[0]//2 # number of spinful blocks
  msy = [[None for ri in range(n)] for j in range(n)]
  for i in range(n): msy[i][i] = sy # add sy
  msy = bmat(msy) # as block matrix
  return msy*np.conjugate(m)*msy # return time reversal




def build_eh(hin,coupling=None,is_sparse=True):
  if coupling is not None:
    c12 = coupling
    c21 = np.conjugate(coupling).T
  else:
    c12 = None
    c21 = None
  return build_nambu_matrix(hin,is_sparse=is_sparse,c12=c12,c21=c21)


def build_nambu_matrix(hin,c12=None,c21=None,is_sparse=True):
    n = hin.shape[0]  # dimension of input
    bout = [[None,None],[None,None]] # initialize None matrix
    bout[0][0] = csc_matrix(hin) # electron part
    bout[1][1] = -time_reversal(csc_matrix(hin)) # hole part
    if c12 is not None:
      bout[0][1] = csc_matrix(c12) # pairing part
    if c21 is not None:
      bout[1][0] = csc_matrix(c21) # pairing part
    bout = bmat(bout) # return matrix
    out = reorder(bout) # reorder matrix
    if issparse(hin): return out
    else: return out.todense()






def add_swave(delta=0.0,is_sparse=False,rs=None):
  """ Adds swave pairing """
  if rs is None: raise # raise error to signal that this is temporal
  n = len(rs) # number of sites
  if callable(delta): # delta is a function
    datar = [delta(ri) for ri in rs] # generate data for the different positions
    data = []
    # the coupling involves up and down
    for dr in datar: # loop over positions
      data.append(dr)  # up e dn h
      data.append(dr)  # dne up h
    iis = range(n*2) # indexes
    coupling = csc_matrix((data,(iis,iis)),dtype=np.complex)  # generate matrix
  else:
    coupling = sp.identity(n*2)*delta # delta matrix
  zero = coupling*0.
  return build_eh(zero,coupling=coupling,is_sparse=is_sparse) # return matrix


def add_swave_to_hamiltonian(self,delta,**kwargs):
    """Add the swave coupling to the Hamiltonian"""
    from .operators import isnumber
#    if isnumber(delta):
#        if delta==0.0: return 
    # spinless Hamiltonian
    if self.check_mode("spinless") or self.check_mode("spinless_nambu"): 
        from .sctk import spinless
        spinless.add_swave_to_hamiltonian(self,delta)
        # spinful Hamiltonian
    elif self.check_mode("spinful") or self.check_mode("spinful_nambu"): 
      self.turn_nambu() # add electron hole
      self.intra = self.intra + add_swave(delta=delta,rs=self.geometry.r,is_sparse=self.is_sparse)
    else: raise









def add_pxipy(delta=0.0,is_sparse=False,r1=None,r2=None):
  """Add px x + py y pairing"""
  def deltafun(r1i,r2i):
    """Function to calculate the pairing"""
    dr = r2i-r1i
    dr2 = dr.dot(dr)
    if 0.9<dr2<1.1: # first neighbor
      dr = delta*dr/np.sqrt(dr2) # unit vector
#      dr = np.cross(dr,np.array([0.,0.,1.]))
      dr = [dr[0],1j*dr[1],0.]      
#      dr = [0.,0.,dr[0]+1j*dr[1]]
#      return dr
      return dvector2deltas(dr) # return delta
    else: return [0.,0.,0.] # zero vector
  return add_pairing(deltas=deltafun,r1=r1,r2=r2)


add_pwave = add_pxipy


# functions for d-vector computations
from .sctk.dvector import dvector2deltas 
from .sctk.dvector import delta2dvector 



def add_pairing(deltas=[[0.,0],[0.,0.]],is_sparse=True,r1=[],r2=[]):
  """ Adds a general pairing in real space"""
  def get_pmatrix(r1i,r2j): # return the different pairings
    if callable(deltas): dv = deltas(r1i,r2j) # get the components
    else: dv = deltas
    duu = dv[0,1] # delta up up
    ddd = dv[1,0] # delta dn dn
    dud = dv[0,0] # delta up dn
    ddu = dv[1,1] # delta up dn
    # be aware of the minus signs coming from the definition of the
    # Nambu spinor!!!!!!!!!!!!!!!!!
    # c_up d_dn d^\dagger_dn -d^\dagger_up
    D = csc_matrix([[dud,duu],[ddd,ddu]]) # SC matrix
#    D = bmat([[None,D],[-D.H,None]]) # the minus sign comes from triplet
    return D
  # superconducting coupling
  n = len(r1)  # number of sites
  pout = [[None for i in range(n)] for j in range(n)] # initialize None matrix
  # zeros in the diagonal
#  for i in range(n): bout[i][i] = csc_matrix(np.zeros((4,4),dtype=np.complex))
  for i in range(n): # loop over sites
    for j in range(n): # loop over sites
      pout[i][j] = get_pmatrix(r1[i],r2[j]) # get this pairing
  diag = csc_matrix(np.zeros((2*len(r1),2*len(r1)),dtype=np.complex)) # diag
  pout = bmat(pout) # convert to block matrix
#  mout = [[diag,pout],[np.conjugate(pout),diag]] # output matrix
  mout = [[diag,pout],[None,diag]] # output matrix
#  mout = [[diag,pout],[pout,diag]] # output matrix
  mout = bmat(mout) # return full matrix
  mout = reorder(mout) # reorder the entries
  return mout






def block2nambu_matrix(m):
  '''Reorder a matrix that has electrons and holes, so that
  the order resembles the Nambu spinor in each site.
  The initial matrix is
  H D
  D H
  The output is a set of block matrices for each site in the 
  Nambu form'''
  R = np.matrix(np.zeros(m.shape)) # zero matrix
  nr = m.shape[0]//4 # number of positions
  for i in range(nr): # electrons
    R[2*i,4*i] = 1.0 # up electron
    R[2*i+1,4*i+1] = 1.0 # down electron
    R[2*i+2*nr,4*i+2] = 1.0 # down holes
    R[2*i+1+2*nr,4*i+3] = 1.0 # up holes
  R = csc_matrix(R) # to sparse
  return R


def block2nambu(m):
    R = block2nambu_matrix(m)
    Rh = np.conjugate(R.T)
    return Rh@m@R

reorder = block2nambu

def nambu2block(m):
    R = block2nambu_matrix(m)
    Rh = np.conjugate(R.T)
    return R@m@Rh


from .sctk.extract import extract_pairing
from .sctk.extract import extract_singlet_pairing
from .sctk.extract import extract_triplet_pairing
from .sctk.pairing import pairing_generator

def add_pairing_to_hamiltonian(self,**kwargs):
    """ Add a general pairing matrix to a Hamiltonian"""
#    self.get_eh_sector = get_eh_sector_odd_even # assign function
    df = pairing_generator(self,**kwargs) # function that outputs a 2x2 matrix
    self.turn_nambu() # add electron hole terms
    r = self.geometry.r # positions 
    m = add_pairing(df,r1=r,r2=r) # intra cell
    self.intra = self.intra + m + m.H
    if self.dimensionality>0:
      if not self.is_multicell: # for multicell hamiltonians
        self.turn_multicell()
#        print("Converting Hamiltonian to multicell")
      from .multicell import Hopping
      for d in self.geometry.neighbor_directions(): # loop over directions
        # this is a workaround to be able to do triplets
        # do it for +k and -k
        if d.dot(d)<0.0001: continue # skip onsite
        r2 = self.geometry.replicas(d) # positions
        m = add_pairing(df,r1=r,r2=r2) # new matrix
#        m2 = add_pairing(df,r1=r,r2=r2) # new matrix, the other way
        m2 = m
        if np.max(np.abs(m))>0.0001: # non zero
            self.hopping.append(Hopping(d=d,m=m)) # add pairing
        if np.max(np.abs(m2))>0.0001: # non zero
            self.hopping.append(Hopping(d=-np.array(d),m=m2.H)) # add pairing
      from .multicell import collect_hopping
      self.hopping = collect_hopping(self)


iden = np.matrix([[1.,0.],[0.,1.]],dtype=np.complex)
taux = np.matrix([[0.,1.],[1.,0.]],dtype=np.complex)
tauz = np.matrix([[1.,0.],[0.,-1.]],dtype=np.complex)




def extract_euphdn(m):
    """Extract electron up hole down sector"""
    n = m.shape[0]//2 # half the dimension
    out = np.matrix(np.zeros((n,n),dtype=np.complex))
    for i in range(n):
      for j in range(n):
        out[i,j] = m[2*i,2*j]
    return out



def superconductivity_type(h):
    """Check the sype of superconductivity"""
    h.check() # check that everything is ok
    hk = h.get_hk_gen() # get geenrator
    k = np.random.random(3) # random kpoint
    h1 = hk(k) # Hamiltonian at +k
    h2 = hk(-k) # Hamiltonian at -k
    (t,t,m1) = extract_pairing(h1)
    (t,t,m2) = extract_pairing(h2)
#    m3 = time_reversal(m1) # time reversal in spin space
    if np.max(np.abs(m1-m2))<1e-5:
        print("Even pairing")
    elif np.max(np.abs(m1+m2))<1e-5:
        print("Odd pairing")


def get_nambu2signless(m):
    """Get the matrix that transforms a SC matrix (with the - sign in the
    spinor), so a basis without - sign"""
    n = m.shape[0]//4 # number of sites
    dd = []
    for i in range(n):
        dd.append(1.)
        dd.append(1.)
        dd.append(1.)
        dd.append(-1.)
    from scipy.sparse import diags
    return diags([dd],[0],shape=(4*n,4*n),dtype=np.complex) # create matrix



def nambu_anomalous_reordering(n):
    """For a matrix of <eu,ed>, return a matrix that reorders
    to the Nambu basis"""
    m = np.zeros((2*n,2*n)) # initialize as zero matrix
    for i in range(n): # loop over sites
        m[2*i,2*i+1] = 1.0 # reorder
        m[2*i+1,2*i] = 1.0 # reorder
    return m # return the matrix




def identify_superconductivity(h,tol=1e-5):
    """Given a Hamiltonian, identify the kind of superconductivity"""
    if not h.has_eh: return [] # empty list
    dd = h.get_multihopping()
    if dd.norm()<tol: return [] # nothing
    out = [] # initialize the list
#    out.append("Superconductivity") # is superconducting
    dv = h.get_average_dvector() # get the average d-vector
    if dv[0]>tol: out.append("dx SC")
    if dv[1]>tol: out.append("dy SC")
    if dv[2]>tol: out.append("dz SC")
    (uum,ddm,udm) = dict2absdeltas(dd.get_dict()) # extract absolute value of deltas
    if np.max(uum)>tol: out.append("up-up pairing")
    if np.max(ddm)>tol: out.append("down-down pairing")
    if np.max(udm)>tol: out.append("up-down pairing")
    if np.sqrt(np.sum(h.get_average_dvector(non_unitarity=True)))>tol:
        out.append("Non-unitary superconductivity")
    if h.dimensionality==0: return out
    k = np.random.random(3) # random k-vector
    m1 = h.get_hk_gen()(k) # get the Bloch hamiltonian
    m2 = h.get_hk_gen()(-k) # get the Bloch hamiltonian
    # now check if it has some symmetry
    singletp = np.sum(np.abs(extract_singlet_pairing(m1))) # singlet
    tripletp = np.sum(np.abs(extract_triplet_pairing(m1))) # triplet
    if singletp>tol: out.append("Spin-singlet superconductivity")
    if tripletp>tol: out.append("Spin-triplet superconductivity")
#    print(np.round(d1-d2,2),tol)
    return out



def dict2absdeltas(mf):
    """Given a hopping dictionary, extract the absolute value of
    the three deltas, uu,dd and ud. This function should be used for
    qualitative checks"""
    for key in mf: n = mf[key].shape[0]//4 # number of sites
    uu = np.zeros(n)
    dd = np.zeros(n)
    ud = np.zeros(n)
    for key in mf: # loop over keys
        m = mf[key]
        (uut,ddt,udt) = extract_pairing(m) # extract the three matrices
        uu += np.sum(np.abs(uut),axis=0)
        dd += np.sum(np.abs(ddt),axis=0)
        ud += np.sum(np.abs(udt),axis=0)
    return (uu,dd,ud)






def check_nambu():
    n = 20
    m00 = np.random.random((n,n)) + 1j*np.random.random((n,n))
    m01 = np.random.random((n,n)) + 1j*np.random.random((n,n))
    m10 = np.random.random((n,n)) + 1j*np.random.random((n,n))
    m = build_nambu_matrix(m00,c12=m01,c21=m10) 
    d01 = get_eh_sector(m,i=0,j=1)
    d10 = get_eh_sector(m,i=1,j=0)
    print(np.max(np.abs(d01-m01)))
    print(np.max(np.abs(d10-m10)))




def turn_nambu(self):
  """Turn a Hamiltonian an Nambu Hamiltonian"""
  nambu = build_eh
  if self.check_mode("spinful_nambu"): return # do nothing
  if not self.check_mode("spinful"): raise # error
  def f(m): return nambu(m,is_sparse=self.is_sparse)
  self.modify_hamiltonian_matrices(f) # modify all the matrices
  self.has_eh = True


from .sctk.extract import get_anomalous_hamiltonian
from .sctk.dvector import average_hamiltonian_dvector




