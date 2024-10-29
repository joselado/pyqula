import numpy as np


from .momenttoprofile import generate_profile

# functions for calculation of local DOS

def get_ldos(m_in,i=0,scale=10.,x=None,
        npol=None,ne=5000,kernel="jackson",**kwargs):
  """Return two arrays with energies and local DOS"""
  if npol is None: npol = ne
  mus = moments_local_dos(m_in/scale,
          i=i,n=npol,**kwargs) # get coefficients
  xs = np.linspace(-1.0,1.0,ne,endpoint=True)*0.99 # energies
  ys = generate_profile(mus,xs,kernel=kernel)
  xs,ys = scale*xs,ys/scale # rescale data
  if x is not None:
    from scipy.interpolate import interp1d
    f = interp1d(xs,ys,bounds_error=False,fill_value=(ys[0],ys[-1]))
    return x,f(x)
  else: return xs,ys




from .kpmnumba import kpm_moments_v

def moments_local_dos(m,i=0,n=200,**kwargs):
  """ Calculates local DOS using the KPM"""
  nd = m.shape[0] # length of the matrix
  mus = np.array([0.0j for j in range(2*n)])
  v = index2vector(i,nd) # generate vector
# get the chebychev moments
  return kpm_moments_v(v,m,n=n,**kwargs)





def index2vector(i,size):
    """Transform an index into a vector"""
    v = np.zeros(size,dtype=np.complex128) # initialize
    if is_iterable(i): # assume it is a collection of indexes 
        for ii in i: v[ii] = np.random.random()-0.5 # assign
        v = v/np.sqrt(np.sum(np.abs(v)**2)) # normalize
    else: # assume that it is an integer
        v[i] = 1.0
    return v


from collections.abc import Iterable
def is_iterable(e): return isinstance(e,Iterable)

