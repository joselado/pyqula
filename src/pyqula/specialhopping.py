import numpy as np
from scipy.sparse import csc_matrix
from numba import jit

#try:
#    from . import specialhoppingf90
##    raise
#    use_fortran=True
#except:
#    use_fortran=False
use_fortran=False


def obj2callable(a):
    if callable(a): return a # input is a function
    else: return lambda x: a # input is a number



def twisted(cutoff=5.0,ti=0.3,lambi=8.0,t=1.0,
        lamb=12.0,dl=3.0,lambz=10.0,b=0.0,phi=0.0):
    """Hopping for twisted bilayer graphene"""
    ti = obj2callable(ti) # convert to callable
    cutoff2 = cutoff**2 # cutoff in distance
    def fun(r1,r2):
        rr = (r1-r2) # distance
        rm  = (r1+r2)/2. # average location
        rr = rr.dot(rr) # distance
        if rr>cutoff2: return 0.0 # too far
        if rr<0.001: return 0.0 # same atom
        dx = r1[0]-r2[0]
        dy = r1[1]-r2[1]
        dz = r1[2]-r2[2]
        r = np.sqrt(rr)
        if (r-1.0)<-0.1: raise
        out = -t*(dx*dx + dy*dy)/rr*np.exp(-lamb*(r-1.0))*np.exp(-lambz*dz*dz)
        # interlayer hopping
        out += -ti(rm)*(dz*dz)/rr*np.exp(-lambi*(r-dl))
        #### fix for magnetic field
  #      cphi = np.cos(phi*np.pi)
  #      sphi = np.sin(phi*np.pi)
  #      r = (r1+r2)/2.
  #      dr = r1-r2
  #      p = 2*r[2]*(dr[0]*sphi - dr[1]*cphi)
  #      out *= np.exp(1j*b*p)
        #####
        return out
    return fun


def twisted_matrix(cutoff=5.0,ti=0.3,lambi=8.0,mint=1e-5,
        t=1.0,lamb=12.0,dl=3.0,lambz=10.0,**kwargs):
  """Function capable of returning the hopping matrix
  for twisted bilayer graphene"""
#  if use_fortran:
#    from . import specialhoppingf90
#    def funhop(r1,r2):
#      """Function that returns a hopping matrix"""
#      nr = len(r1) # 
#      nmax = len(r1)*int(10*cutoff**2) # maximum number of hoppings
#      (ii,jj,ts,nout) = specialhoppingf90.twistedhopping(r1,r2,nmax,
#                                  cutoff,ti,lamb,lambi,lambz,mint,dl)
#      if nout>nmax: raise # sanity check
#      ts = ts[0:nout]
#      ii = ii[0:nout]
#      jj = jj[0:nout]
#      out = csc_matrix((ts,(ii-1,jj-1)),shape=(nr,nr),dtype=np.complex) # matrix
#      return out
#    return funhop # return function
  if True:
      if callable(ti): # workaround for callable interlayer hopping
          tij = twisted(cutoff=cutoff,ti=ti,t=t,
                          lambi=lambi,lamb=lamb,dl=dl,lambz=lambz)
          return entry2matrix(tij)
      else: # conventional JIT function
          return twisted_matrix_python(cutoff=cutoff,ti=ti,mint=mint,
                                  t=t,
                                  lambi=lambi,lamb=lamb,lambz=lambz,
                                  dl=dl,**kwargs)


def multilayer(ti=0.3,dz=3.0):
    """Return hopping for a multilayer"""
    def fhop(ri,rj):
      """Function to compute the hopping"""
      dr = ri-rj ; dr2 = dr.dot(dr) # distance
      if abs(1.0-dr2)<0.01: return 1.0 # first neighbors
      # interlayer hopping (distance between the layers is 3)
      if abs(dz**2-dr2)<0.01 and abs(dz-abs(dr[2]))<0.01: return ti
      return 0.0 # else
    return fhop

def entry2matrix(f):
    def fout(rs1,rs2):
        return np.array([[f(r1,r2) for r1 in rs1] for r2 in rs2]).T
    return fout


def phase_C3_matrix(*args,**kwargs):
    f = phase_C3(*args,**kwargs)
    return entry2matrix(f) # return the matrix


def phase_C3(g,phi=0.5,t=1.0,d=1.0):
    """Create a function that computes hoppings that alternate
    between +\phi and -\phi every 60 degrees"""
    if len(g.r)==1:
      g = g.get_supercell(2) # create a supercell
    ds = g.get_connections()
    i = 0 # first site
    j = ds[i][0] # connected site
    dr = g.r[i] - g.r[j] # distance between sites
    z = dr[0] + 1j*dr[1] # comple vector
    z0 = np.exp(1j*np.pi*2./3.) # rotation
    zs = [z,z*z0,z*z0**2] # the three vectors
    def fun(r1,r2):
        """Function to compute hoppings"""
        dr = r1-r2
        if (d-0.01)<dr.dot(dr)<(d+0.01): # first neighbors
            zi = dr[0]+1j*dr[1]
            for zj in zs: # one of the three directions
                dd = np.abs(zi/zj-1.0)
                if dd<1e-2: 
                    return t*np.exp(1j*phi*np.pi)
            else: return np.exp(-1j*phi*np.pi)
        return 0.0
    return fun


def neighbor_hopping_matrix(g,vs):
    """Return a hopping matrix for the N first neighbors"""
    ds = g.neighbor_distances(n=len(vs)) # get the different distances
    ds = ds[0:len(vs)] # take only these
    return distance_hopping_matrix(vs,ds)




def distance_hopping_matrix(vs,ds):
    """Return a hopping that to the 1-th neighbor is vs"""
    vs = np.array(vs)
    ds = np.array(ds)
    def mgenerator(r1,r2):
        r1 = np.array(r1)
        r2 = np.array(r2)
        n = len(r1)
        out = np.zeros((n,n),dtype=np.complex) # output
        return distance_hopping_matrix_jit(r1,r2,vs,ds*ds,out) 
    return mgenerator

@jit(nopython=True)
def distance_hopping_matrix_jit(r1,r2,vs,ds2,out):
    """Return a hopping that to the 1-th neighbor is vs"""
    n = len(r1) # number of sites
    nn = len(ds2) # number of neighbors
    for i in range(n):
      for j in range(n):
          dr = r1[i] - r2[j] # difference
          dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
          for k in range(nn):
              if np.abs(ds2[k]-dr2)<1e-4: out[i,j] = vs[k]
    return out



def strained_hopping(g,t=1.0,dt=0.0,f=None,**kwargs):
    """Return first neighbor hoppings with strain"""
    if f is None: # no function provided
        from . import potentials
        f = potentials.commensurate_potential(g,average=t,amplitude=dt,
                **kwargs)
    def fout(r1,r2):
        dr = r1-r2
        dr2 = dr.dot(dr)
        if .9<dr2<1.1: return f((r1+r2)/2.)
        else: return 0.0
    return fout


def strained_hopping_matrix(*args,**kwargs):
    f = strained_hopping(*args,**kwargs)
    return entry2matrix(f) # return the matrix


from .algebra import isnumber

class HoppingGenerator():
    """Class for a Hopping generator"""
    def __init__(self,m):
        if type(m)==HoppingGenerator: self.f = m.f # redefine
        elif callable(m): self.f = m # define callable function
        else: raise
    def __call__(self,rs1,rs2):
        """Call method"""
        return self.f(rs1,rs2)
    def __mul__(self,m):
        if isnumber(m): # number input 
            fout = lambda rs1,rs2: np.array(m*np.array(self(rs1,rs2)))
        else: # anything else
            m2 = HoppingGenerator(m) # transform to HoppingGenerator
            fout = lambda rs1,rs2: np.array(self(rs1,rs2))*np.array(m2(rs1,rs2))
        return HoppingGenerator(fout) # return new object
    def __rmul__(self,m): return self*m # commutative 
    def __radd__(self,m): return self + m # commutative 
    def __add__(self,m):  
        m2 = HoppingGenerator(m) # transform to HoppingGenerator
        fout = lambda rs1,rs2: np.array(self(rs1,rs2)) + np.array(m2(rs1,rs2))
        return HoppingGenerator(fout) # return new object
    def __neg__(self): return (-1)*self # minus
    def __sub__(self,a): return self + (-1)*a # minus
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    def apply(self,f):
        """Given a certain function of the position, 
        apply it to all the hoppings"""
        def fm(rs1,rs2): # define a new function
            m = self.f(rs1,rs2) # get the matrix
            p = np.array([[f((r1+r2)/2.) for r1 in rs1] for r2 in rs2])
            return m*p # return product
        return HoppingGenerator(fm) # new generator




def twisted_matrix_python(cutoff=10,**kwargs):
  """Function returning the hopping of a twisted matrix"""
  def tij(rs1,rs2): # function to return
      nr = len(rs1) # length
      nmax = nr*int(10*cutoff**2) # maximum number of hoppings
      data = np.zeros(nmax,dtype=np.complex) # data
      ii = np.zeros(nmax,dtype=int) # index
      jj = np.zeros(nmax,dtype=int) # index
      ii,jj,data,nk = twisted_matrix_jit(np.array(rs1),np.array(rs2),
                         ii,jj,data,cutoff=cutoff,**kwargs) # call function
      if nk>nmax: raise # sanity check
      ii = ii[0:nk] # only nonzero
      jj = jj[0:nk] # only nonzero
      data = data[0:nk] # only nonzero
      out = csc_matrix((data,(ii,jj)),shape=(nr,nr),dtype=np.complex) # matrix
      out.eliminate_zeros()
      return out
  return tij
    


@jit(nopython=True)
def twisted_matrix_jit(rs1,rs2,ii,jj,data,cutoff=5.0,ti=0.3,lambi=8.0,
        mint = 1e-5,t=1.0,
        lamb=12.0,dl=3.0,lambz=10.0):
  """Hopping for twisted bilayer graphene, returning the indexes of a 
  sparse matrix"""
  cutoff2 = cutoff**2 # cutoff in distance
  ik = 0 # counter
  for i1 in range(len(rs1)):
    for i2 in range(len(rs2)):
      r1 = rs1[i1]
      r2 = rs2[i2]
      dr = r1-r2 # distance
      rr = dr[0]**2 + dr[1]**2 + dr[2]**2 # distance
      if rr>cutoff2: continue  # too far
      if rr<0.001: continue # same atom
      dx = r1[0]-r2[0]
      dy = r1[1]-r2[1]
      dz = r1[2]-r2[2]
      r = np.sqrt(rr)
  #    if r2>100.0: return 0.0 # too far
      if (r-1.0)<-0.1:
        raise
      out = -t*(dx*dx + dy*dy)/rr*np.exp(-lamb*(r-1.0))*np.exp(-lambz*dz*dz)
      out += -ti*(dz*dz)/rr*np.exp(-lambi*(r-dl))
      #### fix for magnetic field
#      cphi = np.cos(phi*np.pi)
#      sphi = np.sin(phi*np.pi)
#      r = (r1+r2)/2.
#      dr = r1-r2
#      p = 2*r[2]*(dr[0]*sphi - dr[1]*cphi)
#      out *= np.exp(1j*b*p)
      #####
      if np.abs(out)>mint:
        ii[ik] = i1 # store
        jj[ik] = i2 # store
        data[ik] = out # store
        ik += 1 # increase counter
  return ii,jj,data,ik





def NNG(g,ts):
    """Hopping generator for nearest neighbor hoppings
    - g: geometry
    - ts: hoppings for each neighbor
    """
    fm = neighbor_hopping_matrix(g,ts) # get the function
    return HoppingGenerator(fm) # return a generator 

def ILG(g,ti,**kwargs):
    """Generator for an interlayer hopping using an exponential
    parametrization
    - g: geometry
    - ti: interlayer hopping
    - **kwargs: optional arguments for twisted hopping
    """
    from .potentials import Potential
    from . import algebra
    if callable(ti): ti = Potential(ti) # transform to potential
    elif algebra.isnumber(ti): pass
    else: raise # not implemented
    fm = twisted_matrix(t=0.,ti=-1*ti,**kwargs) # interlayer hopping generator
    # return a generator
    return HoppingGenerator(lambda *args: algebra.todense(fm(*args))) 
