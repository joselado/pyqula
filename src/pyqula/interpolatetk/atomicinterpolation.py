#!/mnt/extradrive/apps/anaconda3/bin/python

import numpy as np
import numpy 
from numba import jit



@jit(nopython=True)
def wave_jit(x,y,centers,heights,facx,facy,z):
  for i in range(len(centers)):
    c = centers[i]
    h = heights[i]
    r = facx*(x-c[0])**2+facy*(y-c[1])**2 # renormalized gaussian       
    z += h*np.exp(-(r))
  return z




def compute_interpolation(centers=[[0.,0.,0.]], heights=[10.0],name="",
        nx=600,ny=600,smooth=.3,smooth_nxny=False,xcut=None,ycut=None,
        dx=None,dy=None,**kwargs):
    """ Creates a set of gaussians at that position and with a height"""
    if len(heights)<len(centers): raise
    if smooth_nxny:
        if xcut is not None:
            facx = nx/float(xcut)
        else:
            facx = nx/np.max(np.abs(np.array(centers)[:,0]))
        if ycut is not None:
            facy = ny/float(ycut)
        else:
            facy = ny/np.max(np.abs(np.array(centers)[:,1]))
        facx = facx/smooth
        facy = facy/smooth
    else:
      fac = 1./smooth # factor to mix the gaussians
      facx = fac
      facy = fac
    def wave(x,y):
        z = x*0.0 + y*0.0
        return wave_jit(x,y,np.array(centers),np.array(heights),facx,facy,z)
    xs = [c[0] for c in centers] # get x coordinates
    ys = [c[1] for c in centers] # get y coordinates
    if dx is None: dx = (np.max(xs) - np.min(xs))/10
    else: dx = float(dx)
    if dy is None: dy = (np.max(ys) - np.min(ys))/10
    else: dy = float(dy)
    x = np.linspace(min(xs)-dx,max(xs)+dx,nx)
    y = np.linspace(min(ys)-dy,max(ys)+dy,ny)
    xout = []
    yout = []
    zout = []
    for ix in x:
      z = wave(ix,y)
      for (iy,iz) in zip(y,z):
          xout.append(ix)
          zout.append(iz)
          yout.append(iy)
    return xout,yout,zout



def atomic_interpolation(x,y,z,xcut=None,ycut=None,**kwargs):
    col = 2
    m = np.array([x,y,z])
    if xcut is not None: # given value
        xc = float(xcut)
        m = m[:,np.abs(m[0,:])<xc]
    if ycut is not None: # given value
        yc = float(ycut)
        m = m[:,np.abs(m[1,:])<yc]
    centers = [[m[0,i],m[1,i],0.] for i in range(len(m[0]))]
    heights = m[col,:]
    return compute_interpolation(centers=centers,heights=heights,**kwargs)
  


