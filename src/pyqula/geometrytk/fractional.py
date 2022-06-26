import numpy as np
import scipy.linalg as lg

def get_fractional_function(g,center=False):
    """Get fractional coordinates"""
  #  if g.dimensionality<2: raise # stop 
    dim = g.dimensionality # dimensionality
    if dim==0: return lambda x: x
    elif dim==1: # one dimensional
      R = np.array([g.a1,[0.,1.,0.],[0.,0.,1.]]).T # transformation matrix
      if np.max(np.abs(g.a1[1:2]))>1e-6: raise
    elif dim==2: # two dimensional
      R = np.array([g.a1,g.a2,[0.,0.,1.]]).T # transformation matrix
      if np.abs(g.a1[2])>1e-6 or np.abs(g.a2[2])>1e-6: raise
    elif dim==3:
      R = np.array([g.a1,g.a2,g.a3]).T # transformation matrix
    else: raise
    g.has_fractional = True # has fractional coordinates
    L = lg.inv(R) # inverse matrix
    def f(r):
        if center: 
            a = np.array(L)@np.array(r).real
            return (a - np.round(a)).real
        else: return (L@np.array(r)).real  # transform
    return f




def get_fractional(g,center=False):
    dim = g.dimensionality # dimensionality
    if dim==0: return
    f = get_fractional_function(g,center=center)
    store = [f(r) for r in g.r] # empty list
    store = np.array(store) # convert to array
    # if you remove the shift the Berry Green formalism does not work
    if dim>0: g.frac_x = store[:,0]
    if dim>1: g.frac_y = store[:,1]
    if dim>2: g.frac_z = store[:,2]
    g.frac_r = store




def fractional2real(self):
    """Write real coordinates using the fractional ones"""
    if self.dimensionality==0: raise
    elif self.dimensionality==1: # 1D
      self.x = self.frac_x*self.a1[0]
    elif self.dimensionality==2: # 2D
      self.x = self.frac_x*self.a1[0] +  self.frac_y*self.a2[0]
      self.y = self.frac_x*self.a1[1] +  self.frac_y*self.a2[1]
    elif self.dimensionality==3: # 3D
      self.x = self.frac_x*self.a1[0] +  self.frac_y*self.a2[0] + self.frac_z*self.a3[0]
      self.y = self.frac_x*self.a1[1] +  self.frac_y*self.a2[1] + self.frac_z*self.a3[1]
      self.z = self.frac_x*self.a1[2] +  self.frac_y*self.a2[2] + self.frac_z*self.a3[2]
    else: raise
    self.xyz2r() # update xyz
    self.center() # center




