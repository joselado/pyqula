
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
  zero = np.matrix(np.zeros(m.shape,dtype=np.complex128)) # zero matrix
  fr = interp1d(xs, ar, axis=0,kind=3,fill_value=zero)
  fi = interp1d(xs, ai, axis=0,kind=3,fill_value=zero)
  def fout(e): # output function
    return np.matrix(fr(e) + 1j*fi(e)) # return 
  return fout # return function


def interpolator2d(x,y,z,mode=None):
    from scipy.interpolate import griddata
    from scipy.interpolate import NearestNDInterpolator
    def f(p):
        return griddata((x,y), z,p[:,0:2], method='nearest')
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
    fo = interpolator2d(ksg[:,0],ksg[:,1],ds.reshape((nx*ny)),mode="periodic")
    out = fo(qs[:,0:2])
    return out



from .interpolatetk.atomicinterpolation import atomic_interpolation



def points2grid(x,y,z,n=100,interpolation="cubic"):
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[np.min(x):np.max(x):n*1j, np.min(y):np.max(y):n*1j]
    grid_z = griddata(np.array([x,y]).T, z, (grid_x, grid_y), method=interpolation)
    return grid_x,grid_y,grid_z



