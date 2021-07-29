import numpy as np
from scipy.sparse import csc_matrix
from numba import jit

try:
    from . import specialhoppingf90
    use_fortran=True
except:
    use_fortran=False
#    print("FORTRAN not working in specialhopping")



def twisted(cutoff=5.0,ti=0.3,lambi=8.0,
        lamb=12.0,dl=3.0,lambz=10.0,b=0.0,phi=0.0):
  """Hopping for twisted bilayer graphene"""
  cutoff2 = cutoff**2 # cutoff in distance
  def fun(r1,r2):
    rr = (r1-r2) # distance
    rr = rr.dot(rr) # distance
    if rr>cutoff2: return 0.0 # too far
    if rr<0.001: return 0.0 # same atom
    dx = r1[0]-r2[0]
    dy = r1[1]-r2[1]
    dz = r1[2]-r2[2]
    r = np.sqrt(rr)
#    if r2>100.0: return 0.0 # too far
    if (r-1.0)<-0.1: 
      raise
    out = -(dx*dx + dy*dy)/rr*np.exp(-lamb*(r-1.0))*np.exp(-lambz*dz*dz)
    out += -ti*(dz*dz)/rr*np.exp(-lambi*(r-dl))
    #### fix for magnetic field
    cphi = np.cos(phi*np.pi)
    sphi = np.sin(phi*np.pi)
    r = (r1+r2)/2.
    dr = r1-r2
    p = 2*r[2]*(dr[0]*sphi - dr[1]*cphi)
    out *= np.exp(1j*b*p)
    #####
    return out
  return fun


def twisted_matrix(cutoff=5.0,ti=0.3,lambi=8.0,
        lamb=12.0,dl=3.0,lambz=10.0,**kwargs):
  """Function capable of returning the hopping matrix
  for twisted bilayer graphene"""
  if use_fortran:
    from . import specialhoppingf90
    def funhop(r1,r2):
      """Function that returns a hopping matrix"""
      nr = len(r1) # 
      nmax = len(r1)*int(10*cutoff**2) # maximum number of hoppings
      (ii,jj,ts,nout) = specialhoppingf90.twistedhopping(r1,r2,nmax,
                                  cutoff,ti,lamb,lambi,lambz,1e-5,dl)
      if nout>nmax: raise # sanity check
      ts = ts[0:nout]
      ii = ii[0:nout]
      jj = jj[0:nout]
      out = csc_matrix((ts,(ii-1,jj-1)),shape=(nr,nr),dtype=np.complex) # matrix
      return out
  else:
    print("Using Python function in twisted")
    def funhop(r1,r2):
      fh = twisted(cutoff=cutoff,ti=ti,lambi=lambi,lamb=lamb,dl=dl,**kwargs)
      m = np.array([[fh(r1i,r2j) for r1i in r1] for r2j in r2],dtype=np.complex)
      m = csc_matrix(m,dtype=np.complex).T
      m.eliminate_zeros()
      return m
#      raise
  return funhop # function




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
    """Create a fucntion that computes hoppings that alternate
    between +\phi and -\phi every 60 degrees"""
    if len(g.r)==1:
      g = g.supercell(2) # create a supercell
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



class HoppingGenerator():
    """Class for a Hopping generator"""
    def __init__(self,m):
        if type(m)==HoppingGenerator: self.f = m.f # redefine
        elif callable(m): self.f = m # define callable function
    def __call__(self,rs1,rs2):
        """Call method"""
        return self.f(rs1,rs2)
    def __mul__(self,m):
        m2 = HoppingGenerator(m) # transform to HoppingGenerator
        fout = lambda rs1,rs2: np.array(self(rs1,rs2))*np.array(m2(rs1,rs2))
        return HoppingGenerator(fout) # return new object
    def __rmul__(self,m): return self*m # commutative 





