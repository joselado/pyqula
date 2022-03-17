import numpy as np

def jackson_kernel(mus):
  """ Modify coeficient using the Jackson Kernel"""
  mo = mus.copy() # copy array
  n = len(mo)
  pn = np.pi/(n+1.) # factor
  for i in range(n):
    fac = ((n-i+1)*np.cos(pn*i)+np.sin(pn*i)/np.tan(pn))/(n+1)
    mo[i] *= fac
  return mo



def lorentz_kernel(mus):
  """ Modify coeficient using the Jackson Kernel"""
  mo = mus.copy() # copy array
  n = len(mo)
  pn = np.pi/(n+1.) # factor
  lamb = 3.
  for i in range(n):
    fac = np.sinh(lamb*(1.-i/n))/np.sinh(lamb)
    mo[i] *= fac
  return mo






def fejer_kernel(mus):
  """Default kernel"""
  n = len(mus)
  mo = mus.copy()
  for i in range(len(mus)):
    mo[i] *= (1.-float(i)/n)
  return mo


