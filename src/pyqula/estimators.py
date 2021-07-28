from __future__ import print_function
import scipy.linalg as lg
import numpy as np

# library to get function that estimate the importance of each point
# in reciprocal space

def kmesh_density(h,nk=20,delta=0.01):
  """Function to compute a k-mesh density districution"""
  if h.dimensionality !=2: raise
  hkgen = h.get_hk_gen() # get generator
  out = [] # empty list
  xout = []
  yout = []
  for x in np.linspace(-0.1,1.1,nk,endpoint=True):
    for y in np.linspace(-0.1,1.1,nk,endpoint=True):
      hk = hkgen([x,y])
      es = lg.eigvalsh(hk) # eigenvalues
      e = np.min(es[es>0.]) - np.max(es[es<0.]) # gap
      out.append(e) # append gap
      xout.append(x)
      yout.append(y)
  # now create the function that interpolates
  import interpolation
  f = interpolation.interpolator2d(xout,yout,out) # interpolator function
  # now define a function that returns a probability density
  def fout(k):
    return delta/(delta**2 + f(k[0],k[1])) # output
  return fout # return function



def weighted_mesh(h,n=10000,delta=0.1):
  """Return a set of random k-points and their relative weight
  using the gap as a probability measure"""
  kx = np.random.random(n) # kx coordinate
  ky = np.random.random(n) # ky coordinate





