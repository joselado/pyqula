from __future__ import print_function, division
import numpy as np
from .algebra import isnumber

class Potential():
    def __init__(self,f,g=None):
        if type(f)==Potential: self.f = f.f # store function
        elif callable(f): self.f = f # store function
        elif isnumber(f): self.f = lambda r: f # store function
        else: 
            print("Unrecognized potential",f)
            raise
        self.g = g # geometry
    def __add__(self,a):
        a = Potential(a)
        g = lambda r: self.f(r) + a.f(r)
        return Potential(g,g=self.g)
    def redefine(self,f):
        g = lambda r: f(self.f(r))
        return Potential(g,g=self.g)
    def __mul__(self,a):
        a = Potential(a)
        g = lambda r: self.f(r)*a.f(r)
        return Potential(g,g=self.g)
    def __rmul__(self,a): return self*a
    def __neg__(self): return (-1)*self
    def __sub__(self,a): return self + (-1)*a
    def __rsub__(self,a): return -self + a
    def __radd__(self,a): return self + a
    def __truediv__(self,a): return self*(1./a)
    def __call__(self,r):
        return self.f(r)
    def normalize(self):
        return Potential(enforce_minmax(self,[0.,1.],g=self.g),g=self.g)
    def set_average(self,average):
        out = enforce_average(self,average,g=self.g)
        return Potential(out,g=self.g)



def cnpot(n=4,k=None,v=1.0,angle=0.):
  """Returns a function that generates a potential
  with C_n symmetry"""
  if k is None: raise
  if n==0: return lambda r: v
  if n%2==0: f = np.cos # even 
  if n%2==1: f = np.sin # even 
  def fun(r):
    """Function with the potential"""
    x0,y0 = r[0],r[1]
    def Rk(k0,angle):
      x0,y0 = k0[0],k0[1] # components
      x = np.cos(angle)*x0 + np.sin(angle)*y0
      y = np.cos(angle)*y0 - np.sin(angle)*x0
      return np.array([x,y,k0[2]]) # return rotated vector
    acu = 0. # result
    for i in range(n):
      ki = Rk(k,np.pi*2*i/n)
      acu += f(2.*np.pi*ki.dot(r)) 
    return v*acu/n
  return Potential(fun)




def aahf1d(n0=0,beta=0.0000001,k=None,b=None,v=1.0,normalize=False):
  """Return the generalized AAHF potential"""
  tau = (1.+np.sqrt(5))/2.
  if b is None: b = 1/tau # default field
  if k is None: k = 3*np.pi*b # default phase
  if beta==0.0: beta=0.000001 # just in case
  def fun(r):
    """Function"""
    ns = r[0] # first coordinate
    ys = np.tanh(beta*(np.cos(2.*np.pi*b*ns+k)-np.cos(np.pi*b)))
    ys /= np.tanh(beta)
    return v*ys
  return fun # return function



def commensurate_potential(g,k=1,amplitude=1.0,n=None,
        average=0.0,minmax=None,**kwargs):
    """Return a potential that is commensurate with
    the lattice"""
    if g.dimensionality==2:
      a12 = g.a2.dot(g.a1)/(np.sqrt(g.a1.dot(g.a1))*np.sqrt(g.a1.dot(g.a1)))
      if n is None:
        if 0.49<abs(a12)<0.51: n = 6
        elif abs(a12)<0.01: n = 4
        else: n = 3
      f = cnpot(n=n,k=k*g.b1,**kwargs)
    elif g.dimensionality==1: 
      f = cnpot(n=1,k=k*g.b1,**kwargs)
    else: raise
    f = enforce_amplitude(f,amplitude,g=g) # enforce the amplitude
    f = enforce_average(f,average,g=g) # enforce average
    if minmax is not None: f = enforce_minmax(f,minmax,g=g) # enforce minmax
    f = Potential(f)
    f.g = g
    return f


def impurity(r0,v=0.):
    """Create the potential for an impurity"""
    def f(r):
        dr = r-r0
        if dr.dot(dr)<1e-3: return v
        else: return 0.
    return Potential(f) # return the potential



def fibonacci(n,n0=0):
    """Generate the Fibonacci sequence"""
    a1 = [0] # first
    a2 = [0,1] # first
    while True:
      a3 = a1 + a2
      a1 = a2
      a2 = a3
      if len(a3)>n+n0: break
    out = [a3[i] for i in range(n0,n0+n)] # output
    return np.array(out)




def thue_morse(n,n0=0):
  """Generate the Thue-Morse sequence"""
  out = []
  for i in range(n0,n0+n): # loop over integers
    ib = bin(i).split("b")[1] # in binary
    acu = 0 # start
    for iib in ib: # loop
      if iib=="1": acu += 1
    acu = acu%2 # modulus
    out.append(acu) # output
  return np.array(out)



def tbgAA(g):
    """Return a function that yields +1 for AA and -1 otherwise"""
    from . import geometry
    from .specialhopping import twisted_matrix
    h = g.get_hamiltonian(mgenerator=twisted_matrix(ti=0.3),
            has_spin=False,is_sparse=True)
    h.set_filling(0.5,nk=1)
    ks = [[.5,0.,0.],[0.,.5,0.],[.5,.5,0.]]
    (x,y,d) = h.get_ldos(e=0.0,delta=0.01,ks=ks,mode="arpack",
            write=False,silent=True,nrep=1)
    d = d - np.mean(d) # average in zero
    d = d - np.min(d)
    d = d/np.max(d) # set maximum in 1
    d = 2*(d-0.5) # between -1 and +1
    funr = geometry.get_fractional_function(g,center=True)
    rf = np.array([funr(ri) for ri in g.r])
    fint = interpolate2d(rf[:,0:2],d) # interpolation
    return Potential(lambda ri: fint(funr(ri))[0])



def interpolate2d(r,v):
    """Return a function that does 2d interpolation"""
    from scipy.interpolate import interp2d
    x,y = r[:,0],r[:,1] # data
    grid_x, grid_y = np.mgrid[np.min(x):np.max(x):100j,np.min(y):np.max(y):100j]
    from scipy.interpolate import griddata
    grid_z = griddata(r,v, (grid_x, grid_y), method='nearest')
    f = interp2d(grid_x, grid_y, grid_z, kind='linear')
    return lambda ri: f(ri[0],ri[1])




def enforce_average(f,a,g=None):
    """Normalize the average value of a function for the geometry"""
    if g is None: raise
    m = np.mean([f(ri) for ri in g.r]) # average value
    def fout(r):
        return f(r) + a - m # return this value
    return fout # return new function


def enforce_amplitude(f,a,g=None):
    """Normalize the average value of a function for the geometry"""
    if g is None: raise
    vs = [f(ri) for ri in g.r] 
    minv = np.min(vs)
    maxv = np.max(vs)
    dv = maxv-minv # amplitude
    def fout(r):
        return f(r)*a/dv # return this value
    return fout # return new function



def enforce_minmax(f,a,g=None):
    """Enforce the minimum, without changing the maximum"""
    if g is None: raise
    vs = [f(ri) for ri in g.r] 
    minv = np.min(vs)
    maxv = np.max(vs)
    dv = maxv-minv # amplitude
    fout = lambda r: (f(r)-minv)/dv*a[1] + (1.-(f(r)-minv)/dv)*a[0] # return this value
    return fout # return new function



def array2potential(x,y,v):
    """Given an initial xyz array, return a function
    that interpolates over them"""
    from .interpolation import interpolator2d
    if len(v)!=len(x): raise
    return Potential(interpolator2d(x,y,v))


def object2potential(V,r=None):
    """Transform a generic object into a callable potential"""
    # this should be finished
    return V


from .potentialtk.profiles import radial_decay


def delta(r0,v0=1.0):
    """Delta function potential"""
    def f(r):
        dr = r - r0
        dr2 = dr.dot(dr)
        if dr2<1e-4: return v0
        else: return v0*0.0
    return Potential(f) # return potential



def edge_potential(g):
    """Return potential for the edge sites"""
    cs = g.get_connections() # get the connections
    nn = np.max([len(c) for c in cs]) # maximum number of connections
    v = [len(c)<nn for c in cs] # check if the atom is an edge 
    fout = 0 # output function
    for i in range(len(v)): # loop over locations
        if v[i]: fout = fout + delta(g.r[i],v0=1.0)
    return Potential(fout) # return the potential



def stacking_potential(g,**kwargs):
    """Given a certain geometry, return a function with that gives the
    stacking"""
    # this function should be polished
    from .crystalfield import cf_potential
#    g1 = g.copy() ; g1.r[:,2] = 0.0
    v = cf_potential(g,vc=2.0,mode="stacking",**kwargs)
    v = v - np.min(v) # shift to zero
    v = v/np.max(v) # normalize
    v = v - 0.5 # between -0.5 and 0.5
    from .geometry import array2function
    fv = array2function(g,v)
    return Potential(fv) # return the potential






