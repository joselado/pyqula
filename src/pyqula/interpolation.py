
from __future__ import print_function
from scipy.interpolate import interp1d
import numpy as np

# perform interpolation a matrix 
def intermatrix(fin,xs=np.linspace(-5.0,5.0,20)):
  """Return a function capable to interpolate the
  input function between values in the interval. Values
  outside the intervale will be returned as zero"""
  m = fin(0) # call once
  ar = np.zeros((len(xs),m.shape[0],m.shape[1])) # empty array, real part
  ai = np.zeros((len(xs),m.shape[0],m.shape[1])) # empty array, imag part
  for i in range(len(xs)): # loop
    m = fin(xs[i]) # call the function
    ar[i,:,:] = m.real # real part
    ai[i,:,:] = m.imag # imaginary part
  zero = np.matrix(np.zeros(m.shape,dtype=np.complex)) # zero matrix
  fr = interp1d(xs, ar, axis=0,kind=3,fill_value=zero)
  fi = interp1d(xs, ai, axis=0,kind=3,fill_value=zero)
  def fout(e): # output function
    return np.matrix(fr(e) + 1j*fi(e)) # return 
  return fout # return function


def interpolator2d(x,y,z):
  """Return a 2D interpolator"""
  x = np.array(x)
  y = np.array(y)
  from scipy.interpolate import interp2d
  ny = (y.max()-y.min())/abs(y[1]-y[0])
  ny = int(round(ny)) + 1
  nx = len(z)/ny
  nx = int(nx)
  ny = int(ny)
  x0 = np.linspace(min(x),max(x),nx)
  y0 = np.linspace(min(y),max(y),ny)
  xx, yy = np.meshgrid(x0, y0)
  Z = np.array(z).reshape(nx,ny) # makes a (Zy,Zx) matrix out of z
  T = Z.T
  f = interp2d(x0, y0, T, kind='linear')
  return f



def interpolator2d(x,y,z,mode=None):
    from scipy.interpolate import griddata
    from scipy.interpolate import NearestNDInterpolator
    def f(p):
        return griddata((x,y), z,p[0:2], method='nearest')
    if mode is None: return f
    elif mode=="periodic":
        def f0(k):
            """Define a periodic function"""
            k = k[:,0:2]%1.
            return f(k)
        return f0
    else: raise




def periodic_grid2mesh(ds,qs):
    """Given a certain data in a (periodic) grid, return in the input mesh"""
    nx = ds.shape[0]
    ny = ds.shape[1]
    grid_kx, grid_ky = np.mgrid[0:1:nx*1j, 0:1:ny*1j] # kx and ky
    ksg = np.array([grid_kx, grid_ky]).reshape((2,nx*ny)).T # create points
    return interpolator2d(ksg[:,0],ksg[:,1],ds.reshape((nx*ny)),mode="periodic")(qs[:,0:2])



from .interpolatetk.atomicinterpolation import atomic_interpolation


