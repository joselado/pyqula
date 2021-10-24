import numpy as np


def get_k2K_generator(self,toreal=False):
      """
      Function to turn a reciprocal lattice vector to natural units
      """
      R = get_k2K(self) # get matrix
      if toreal: R = np.conjugate(R).T # transform to real coordinates
      def fun(k0):
        if len(k0)==3: k = k0 # do nothing
        elif len(k0)==2: k = np.array([k0[0],k0[1],0.]) # convert to three comp
        r = np.array(k) # real space vectors
        out = np.array(R)@r # change of basis
        if len(k0)==2: return np.array([out[0],out[1]]) # return two
        return out
      return fun




def get_k2K(g):
  """Return a matrix that converts vectors
  in the reciprocal space into natural units, useful for drawing
  2D/3D quantities"""
  if g.dimensionality == 2:
    (ux,uy,uz) = (g.a1,g.a2,np.array([0.,0.,1]))
  elif g.dimensionality == 3:
    (ux,uy,uz) = (g.a1,g.a2,g.a3)
  else: raise
  ux = ux/np.sqrt(ux.dot(ux))
  uy = uy/np.sqrt(uy.dot(uy))
  uz = uz/np.sqrt(uz.dot(uz))
  a2kn = np.matrix([ux,uy,uz]) # matrix for the change of basis
  r2a = np.matrix([ux,uy,uz]).T.I # from real space to lattice vectors
  R = a2kn@r2a@a2kn.T # rotation matrix
  return R


def unitary(v): return v/np.sqrt(v.dot(v))

