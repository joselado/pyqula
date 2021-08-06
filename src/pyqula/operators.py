# library to create operators
from __future__ import division
import numpy as np
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csc_matrix
from scipy.sparse import bmat,diags
from scipy.sparse import identity
from .superconductivity import build_eh
from scipy.sparse import issparse
import scipy.linalg as lg
#from bandstructure import braket_wAw
from . import current
from . import algebra
from . import topology
from . import superconductivity
from .algebra import braket_wAw

import numbers

isnumber = algebra.isnumber


class Operator():
    def __init__(self,m,linear=True):
        """Initialization"""
        self.linear = linear
        self.matrix = None
        if algebra.ismatrix(m):
            self.m = lambda v,k=None: m@v # create dummy function
            self.matrix = m
        elif type(m)==Operator: 
            self.m = m.m
            self.linear = m.linear
        elif isinstance(m, numbers.Number): 
            self.m = lambda v,k=None: m*v
        elif callable(m): 
            self.m = m # as function
        else: 
            print("Unrecognised type",type(m))
            raise
    def __mul__(self,a):
        """Define the multiply method"""
        if type(a)==Operator:
            out = Operator(self)
            if self.matrix is not None and a.matrix is not None:
                out.matrix = self.matrix@a.matrix
                out.m = lambda v,k=None: out.matrix@v # create dummy function
            else: out.m = lambda v,k=None: self.m(a.m(v,k=k),k=k)
            out.linear = self.linear and a.linear
            return out
        elif algebra.ismatrix(a): # matrix type
            if self.matrix is not None: # return a matrix
                return self.matrix@a # multiply matrices
            else:
                return self*Operator(a) # convert to operator
        else:
            return self*Operator(a) # convert to operator
    def __rmul__(self,a):
        return Operator(a)*self
    def __add__(self,a):
        """Define the add method"""
        if type(a)==Operator:
            out = Operator(self)
            out.m = lambda v,k=None: self.m(v,k=k) + a.m(v,k=k)
            if self.matrix is not None and a.matrix is not None:
                out.matrix = self.matrix + a.matrix
            out.linear = self.linear and a.linear
            return out
        else:
            return self + Operator(a) # convert to operator
    def __sub__(self,a):
        """Substraction method"""
        return self + (-a)
    def __neg__(self):
        """Negative operator"""
        out = Operator(self)
        out.m = lambda v,k=None: -self.m(v,k=k)
        return out
    def __call__(self,v,k=None):
        """Define the call method"""
        return self.m(v,k=k) 
    def get_matrix(self):
        """Return matrix if possible"""
        if self.matrix is not None: return self.matrix
    def braket(self,w,**kwargs):
        """Compute an expectation value"""
        wi = self(w,**kwargs) # apply the operator
        out =  algebra.braket_ww(w,wi)
        if np.abs(out.imag)<1e-8: return out.real
        else: return out


def object2operator(a):
    if a is None: return None
    else: return Operator(a)



def index(h,n=[0]):
  """Return a projector onto a site"""
  num = len(h.geometry.r)
  val = [1. for i in n]
  m = csc((val,(n,n)),shape=(num,num),dtype=np.complex)
  return h.spinless2full(m) # return matrix



def rfunction2operator(h,f):
    """Given a function that takes a position, return the operator"""
    n = len(h.geometry.r)
    val = [f(ri) for ri in h.geometry.r]
    inds = range(n)
    m = csc((val,(inds,inds)),shape=(n,n),dtype=np.complex)
    return h.spinless2full(m) # return matrix


def density2operator(h,d):
    """Given a function that takes a position, return the operator"""
    n = len(h.geometry.r)
    if len(d)!=n: raise
    inds = range(n)
    m = csc((d,(inds,inds)),shape=(n,n),dtype=np.complex)
    return h.spinless2full(m) # return matrix




def operator2list(operator):
  """Convert an input operator in a list of operators"""
  if operator is None: # no operator given on input
    operator = [] # empty list
  elif not isinstance(operator,list): # if it is not a list
    operator = [operator] # convert to list
  return operator


#
#def get_surface(h,cut = 0.5,which="both"):
#  """Return an operator which is non-zero in the upper surface"""
#  zmax = np.max(h.geometry.r[:,2]) # maximum z
#  zmin = np.min(h.geometry.r[:,2]) # maximum z
#  dind = 1 # index to which divide the positions
#  n = len(h.geometry.r) # number of elments of the hamiltonian
#  data = [] # epmty list
#  for i in range(n): # loop over elements
#    z = h.geometry.z[i]
#    if which=="upper": # only the upper surface
#      if np.abs(z-zmax) < cut:  data.append(1.)
#      else: data.append(0.)
#    elif which=="lower": # only the upper surface
#      if np.abs(z-zmin) < cut:  data.append(1.)
#      else: data.append(0.)
#    elif which=="both": # only the upper surface
#      if np.abs(z-zmax) < cut:  data.append(1.)
#      elif np.abs(z-zmin) < cut:  data.append(1.)
#      else: data.append(0.)
#    else: raise
#  row, col = range(n),range(n)
#  m = csc((data,(row,col)),shape=(n,n),dtype=np.complex)
#  m = h.spinless2full(m)
#  return m # return the operator
#


def interface1d(h,cut = 3.):
  dind = 1 # index to which divide the positions
  if h.has_spin:  dind *= 2 # duplicate for spin
  if h.has_eh:  dind *= 2  # duplicate for eh
  n = h.intra.shape[0] # number of elments of the hamiltonian
  data = [] # epmty list
  for i in range(n): # loop over elements
    y = h.geometry.y[i//dind]
    if np.abs(y)<cut: data.append(1.) # if it belongs to the interface
    else:  data.append(0.)  # otherwise
  row, col = range(n),range(n)
  m = csc((data,(row,col)),shape=(n,n),dtype=np.complex)
  return m # return the operator



def get_interface(h,fun=None):
  """Return an operator that projects onte the interface"""
  dind = 1 # index to which divide the positions
  if h.has_spin:  dind *= 2 # duplicate for spin
  if h.has_eh:  dind *= 2  # duplicate for eh
  iden = csc(np.matrix(np.identity(dind,dtype=np.complex))) # identity matrix
  r = h.geometry.r # positions
  out = [[None for ri in r] for rj in r] # initialize
  if fun is None: # no input function
    cut = 2.0 # cutoff
    if h.dimensionality==1: index = 1
    elif h.dimensionality==2: index = 2
    else: raise
    def fun(ri): # define the function
      if np.abs(ri[index])<cut: return 1.0
      else: return 0.0
  for i in range(len(r)): # loop over positions
    out[i][i] = fun(r[i])*iden 
  return bmat(out) # return matrix



def get_pairing(h,ptype="s"):
  """Return an operator that calculates the expectation value of the
  s-wave pairing"""
  if not h.has_eh: raise # only for e-h systems
  if ptype=="s": op = superconductivity.spair
  elif ptype=="deltax": op = superconductivity.deltax
  elif ptype=="deltay": op = superconductivity.deltay
  elif ptype=="deltaz": op = superconductivity.deltaz
  else: raise
  r = h.geometry.r
  out = [[None for ri in r] for rj in r]
  for i in range(len(r)): # loop over positions
    out[i][i] = op
  return bmat(out) # return matrix



def get_electron(h):
  """Operator to project on the electron sector"""
  if not h.has_eh:
      return np.identity(h.intra.shape[0])
  elif h.check_mode("spinful_nambu"): # only for e-h systems
      op = superconductivity.proje
      r = h.geometry.r
      out = [[None for ri in r] for rj in r]
      for i in range(len(r)): # loop over positions
        out[i][i] = op
      return bmat(out)
  elif h.check_mode("spinless_nambu"):
      from .sctk import spinless
      return spinless.proje(h.intra.shape[0])
  else: raise


def get_hole(h):
  """Operator to project on the electron sector"""
  if not h.has_eh: raise # only for e-h systems
  op = superconductivity.projh
  r = h.geometry.r
  out = [[None for ri in r] for rj in r]
  for i in range(len(r)): # loop over positions
    out[i][i] = op
  return bmat(out)


def get_bulk(h,fac=0.2):
    """Return the bulk operator"""
    r = h.geometry.r # positions
    g = h.geometry
    g.center() # center the geometry
    out = np.array([1. for ir in r]) # initialize
    if h.dimensionality==0:
        dr = r[:,0]**2 + r[:,1]**2 # radii
        dr = dr - np.min(dr)
        dr = dr/np.max(dr) # to interval 0,1
        out[fac>dr] = 0.0 # set to zero
    elif h.dimensionality==1:
        dr = r[:,1] # y positions
        dr = dr - np.min(dr)
        dr = dr/np.max(dr) # to interval 0,1
        out[fac>dr] = 0.0 # set to zero
        out[(1.-fac)<dr] = 0.0 # set to zero
    elif h.dimensionality==2:
        dr = r[:,2] # z positions
        dr = dr - np.min(dr)
        dr = dr/np.max(dr) # to interval 0,1
        out[fac>dr] = 0.0 # set to zero
        out[(1.-fac)<dr] = 0.0 # set to zero
    else: return NotImplemented
    from scipy.sparse import diags
    n = len(r) # number of sites
    out = diags([out],[0],shape=(n,n),dtype=np.complex) # create matrix
    m = h.spinless2full(out) # return this matrix
    return m@m # return the square


def get_surface(self,**kwargs):
    m = get_bulk(self,**kwargs)
    return identity(m.shape[0]) - m 


def bulk1d(h,p = 0.5):
    return get_bulk(h,fac=1.-p/2.)

def get_xposition(h):  return get_position(h,mode="x")
def get_yposition(h):  return get_position(h,mode="y")
def get_zposition(h):  return get_position(h,mode="z")




def get_position(h,mode="z"):
  dind = 1
  if h.has_spin:  dind *= 2 # duplicate for spin
  if h.has_eh:  dind *= 2  # duplicate for eh
  n = h.intra.shape[0] # number of elments of the hamiltonian
  if len(h.geometry.z)!=n//dind: raise # dimensions do not match
  data = [] # epmty list
  if mode=="x": pos = h.geometry.x
  elif mode=="y": pos = h.geometry.y
  elif mode=="z":  pos = h.geometry.z
  else: raise
  for i in range(n): # loop over elements
    z = pos[i//dind]
    data.append(z)
  row, col = range(n),range(n)
  m = csc((data,(row,col)),shape=(n,n),dtype=np.complex)
  return m # return the operator



from .rotate_spin import sx,sy,sz # import pauli matrices
 

def get_si(h,i=1):
  """Return a certain Pauli matrix for the full Hamiltonian"""
  if not h.has_spin: return None # no spin
  if i==1: si = sx # sx matrix
  elif i==2: si = sy # sy matrix
  elif i==3: si = sz # sz matrix
  else: raise # unknown pauli matrix
  if h.has_eh: ndim = h.intra.shape[0]//4 # half the dimension
  else: ndim = h.intra.shape[0]//2 # dimension
  if h.has_spin: # spinful system
    op = [[None for i in range(ndim)] for j in range(ndim)] # initialize
    for i in range(ndim): op[i][i] = si # store matrix
    op = bmat(op) # create matrix
  if h.has_eh: op = build_eh(op,is_sparse=True) # add electron and hole parts 
  return op

# define the functions for the three spin components
get_sx = lambda h: get_si(h,i=1) # sx matrix
get_sy = lambda h: get_si(h,i=2) # sy matrix
get_sz = lambda h: get_si(h,i=3) # sz matrix






def get_z(h):
  """Operator for the calculation of z expectation value"""
  if h.intra.shape[0]==len(h.geometry.z): # if as many positions as entries
    op = np.zeros(h.intra.shape,dtype=np.complex) # initialize matrix
    for i in range(len(h.geometry.z)):
      op[i,i] = h.geometry.z[i]
    return op
  raise
  if h.has_eh: raise
  if not h.has_spin: raise
  if h.has_spin:
    op = np.zeros(h.intra.shape,dtype=np.complex) # initialize matrix
    for i in range(len(op)//2):   
      op[2*i,2*i+1] = -1j
      op[2*i+1,2*i] = 1j
  



def get_rop(h,fun):
  """Operator for the calculation of a position expectation value"""
  rep = 1 # repetitions 
  if h.has_spin: rep *= 2
  if h.has_eh: rep *= 2
  data = []
  for ri in h.geometry.r: 
    for i in range(rep): data.append(fun(ri)) # store
  n = h.intra.shape[0]
  row = range(n)
  col = range(n)
  m = csc((data,(row,col)),shape=(n,n),dtype=np.complex)
  return m




def get_sublattice(h,mode="both"):
  """Sublattice operator"""
  if not h.geometry.has_sublattice: raise
  rep = 1 # repetitions 
  if h.has_spin: rep *= 2
  if h.has_eh: rep *= 2
  data = []
  for s in h.geometry.sublattice: 
    for i in range(rep): 
      if mode=="both": data.append(s) # store
      elif mode=="A": data.append((s+1.)/2.) # store
      elif mode=="B": data.append((-s+1.)/2.) # store
      else: raise
  n = h.intra.shape[0]
  row = range(n)
  col = range(n)
  m = csc((data,(row,col)),shape=(n,n),dtype=np.complex)
  return m


def get_velocity(h):
  """Return the velocity operator"""
  if h.dimensionality==1:
    vk = current.current_operator(h)
    def f(w,k=[0.,0.,0.]):
        return vk(k)@w
    return f
  elif h.dimensionality==2:
    def f(w,k=[0.,0.,0.]):
      vx = current.derivative(h,k,order=[0,1])
      vy = current.derivative(h,k,order=[1,0])
      R = np.array(h.geometry.get_k2K())
#      R = algebra.inv(R) # not sure if this is ok
      v = [braket_wAw(w,vx),braket_wAw(w,vy),0]
      v = np.array(v).real
      return R@v # return the scalar product
    return Operator(f)
  else: raise



get_current = get_velocity

def get_spin_current(h):
  vk = current.current_operator(h)
  sz = get_sz(h)
  def f(w,k=[0.,0.,0.]):
    return braket_wAw(w,vk(k)).real*braket_wAw(w,sz).real
  return f





def get_valley(h,delta=None,**kwargs):
  """Return a callable that calculates the valley expectation value
  using the modified Haldane coupling"""
  if h.dimensionality==0: projector = True # zero dimensional
  ho = h.copy() # copy Hamiltonian
  ho.turn_multicell()
  ho.clean() # set to zero
  ho.add_modified_haldane(1.0/4.5) # add modified Haldane coupling
  hkgen = ho.get_hk_gen() # get generator for the hk Hamiltonian
  def sharpen(m):
    """Sharpen the eigenvalues of a matrix"""
#    return m
    if delta is None: return m # do nothing
    if issparse(m): return m # temporal workaround
    (es,vs) = algebra.eigh(m) # diagonalize
    es = es/(np.abs(es)+delta) # renormalize the valley eigenvalues
    vs = np.matrix(vs) # convert
    m0 = np.matrix(np.diag(es)) # build new hamiltonian
    return vs@m0@vs.H # return renormalized operator
  def fun(m=None,k=None):
      if h.dimensionality>0 and k is None: raise # requires a kpoint
      hk = hkgen(k) # evaluate Hamiltonian
      hk = sharpen(hk) # sharpen the valley
      if m is None: return hk # just return the valley operator
      else: return hk@m # return the projector
  if h.dimensionality==0: return fun() # return a matrix
  return fun # return function



def get_inplane_valley(h):
  """Returns an operator that computes the absolute value
  of the intervalley mixing"""
  ho = h.copy() # copy Hamiltonian
  ho.clean() # set to zero
  ho.add_modified_haldane(1.0/4.5) # add modified Haldane coupling
  hkgen = ho.get_hk_gen() # get generator for the hk Hamiltonian
  hkgen0 = h.get_hk_gen() # get generator for the hk Hamiltonian
  def fun(w,k=None):
#    return abs(np.sum(w*w))
    if h.dimensionality>0 and k is None: raise # requires a kpoint
    hk = hkgen(k) # evaluate Hamiltonian
    hk0 = hkgen0(k) # evaluate Hamiltonian
    A = hk*hk0 - hk0*hk # commutator
    A = -A*A
    return abs(braket_wAw(w,A)) # return the braket
  return fun # return function





def tofunction(A):
    """Transform this object into a callable function"""
    if A is None: return lambda x,k=0.0: 1.0
    return Operator(A) # use operator
#    if A is None: return lambda x,k=0.0: 1.0 # no input
#    if callable(A): return A # if it is a function
#    else: return lambda x,k=0.0: braket_wAw(x,A).real # if it is a matrix


def ipr(w,k=None):
    """IPR operator"""
    return np.sum(np.abs(w)**4)*w # return a vector


def get_envelop(h,sites=[],d=0.3):
    """
    Return a list of operators that project on the different
    sites
    """
    # get a first neighbor Hamiltonian
    h0 = h.geometry.get_hamiltonian(has_spin=h.has_spin,is_sparse=True)
    m = h0.get_hk_gen()([0.,0.,0.]) # evaluate Hamiltonian at Gamma
    out = [] # output list
    for s in sites: # loop over sites
      c = m.getcol(s) # get column
      c = np.array(c.todense()) # transform into a dense matrix
      c = c.reshape(m.shape[0]) # 1D vector
      c = c*d # renormalize all the hoppings
      c[s] = 1.0 # set same atom to 1
      c = c/np.sum(c) # normalize the whole vector
      c = diags([c],[0],dtype=np.complex) # create matrix
      out.append(c) # store matrix
    return out # return matrices


def get_sigma_minus(h):
    """
    Return the sublattice Pauli matrix \sigma_-
    """
    def fun(r1,r2):
        i1 = h.geometry.get_index(r1,replicas=True)
        if not h.geometry.sublattice[i1]==1: return 0.0
        i2 = h.geometry.get_index(r2,replicas=True)
        dr = r1-r2 # distance
        if 0.9<dr.dot(dr)<1.1: return 1.0 # get first neighbor
        return 0.0
    h0 = h.geometry.get_hamiltonian(has_spin=h.has_spin,fun=fun) # FN coupling
    hk = h0.get_hk_gen() # get generator
    return hk # return function





def get_valley_taux(h):
    """
    Return the tau x valley operator
    """
    g = h.geometry
    h0 = g.get_hamiltonian(has_spin=False) # FN coupling
    h0.turn_multicell() # multicell Hamiltonian
    h1 = h0.copy() # new Hmailtonian
    h1.clean()
    hs = h1.copy() # for sublattice
    hs.add_onsite(0.5) ; hs.add_sublattice_imbalance(0.5) 
    h1.add_kekule(1.0) # add kekule term 
    hops = hs.get_multihopping() # Multihopping for sublattice
    hop0 = h0.get_multihopping() # MultiHopping object
    hop1 = h1.get_multihopping() # MultiHopping object
    # kill hoppings that start in the A sublattice
    hop0 = hop0*hops
   # hop1 = hop0*hops
    # now define the valley mixing
    hop2 = 1j*hop0*hop1 
    hop2 = hop1
    hop2 = hop2 + hop2.get_dagger() # make it Hermitian
    hop2 = hop2.get_dict() # get the dictionary
    h2 = h0.copy() # dummy Hamiltonian
    h2.clean()
    h2.intra = hop2[(0,0,0)] # onsite matrix
    del hop2[(0,0,0)] # onsite matrix
    h2.hoppings = hop2 # store the rest of the hoppings
    hk = h2.get_hk_gen() # get the generating function
    return Operator(lambda m=None,k=None: m@hk(k)) # return operator








def get_operator(op,k=[0.,0.,0.],h=None):
    """Get a function that acts as an operator"""
    return Operator(op)


def get_berry(h,**kwargs):
    """Return Berry operator"""
    return topology.berry_operator(h,**kwargs)

def get_valley_berry(h,**kwargs):
    """Return Valley Berry operator"""
    return get_operator_berry(h,"valley",**kwargs)


def get_operator_berry(h,name,**kwargs):
    """Return Valley Berry operator"""
    op = h.get_operator(name,return_matrix=True)
    return topology.berry_operator(h,operator=op,**kwargs)



def get_sz_berry(h,**kwargs):
    """Return Valley Berry operator"""
    return get_operator_berry(h,"sz",**kwargs)


def get_matrix_operator(h,name,k=None,**kwargs):
    """Return a function that takes a matrix as input and returns another
    matrix"""
    if name=="valley":
        op = get_valley(h,projector=True) # valley operator
        return op
    elif name in ["valley_spin","spin_valley","valley_sz","sz_valley"]:
        op = get_valley(h,projector=True) # valley operator
        sz = h.get_operator("sz")
        return lambda m,k=None: op(m,k=k)@sz # return operator
    else:
        op = h.get_operator(name) # assume that it is a matrix
        return lambda m,k=None: op@m


def bool_layer_array(g,n=0):
    """Return the lowest layer array"""
    fac = []
    z0 = sorted(np.unique(g.z).tolist())[n]
    fac = g.z*0. # initialize
    fac[np.abs(g.z-z0)<1e-3] = 1.0
#    for z in g.z:
#        if abs(z-z0)<1e-3: fac.append(1)
#        else: fac.append(0)
#    fac = np.array(fac)
    return fac


bottom_layer = lambda g: bool_layer_array(g,n=0)
top_layer = lambda g: bool_layer_array(g,n=1)

def get_valley_layer(self,n=0,**kwargs):
    """Get the valley operator for a specific layer"""
    ht = self.copy() # create a dummy
    fac = bool_layer_array(self.geometry,n=n) # create array
    ht.geometry.sublattice = self.geometry.sublattice * fac
    return get_valley(ht,**kwargs) # return the valley operator

operator_list = ["None","Sx","Sy","Sz","valley","sublattice","Berry","valleyberry","IPR","electron","hole","Bulk","Surface","xposition","yposition","zposition"]

def get_layer(self,n=0):
   fac = bool_layer_array(self.geometry,n=n)
   inds = range(len(fac)) # sites
   d = len(fac)
   m = csc_matrix((fac,(inds,inds)),shape=(d,d),dtype=np.complex) # matrix
   return self.spinless2full(m)




def get_up(self):
    """Return up sector"""
    op = get_sz(self)
    return (op@op + op)/2.


def get_dn(self):
    """Return up sector"""
    op = get_sz(self)
    return (op@op - op)/2.


def get_potential(self,**kwargs):
    """Return the operator associated to a potential"""
    h = self.copy()
    h.clean() # clean the Hamiltonian
    from . import potentials
    f = potentials.commensurate_potential(h.geometry,amplitude=1.0,**kwargs)
    h.add_onsite(f)
    return Operator(h.intra) # return the operator


