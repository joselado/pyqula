import numpy as np
from .. import algebra
from .sharpen import get_sharpen


def valley_coupling(g):
  """Return the modified Haldane coupling whose mass is one at the Dirac
  points, so that it is tau_z with eigenvalues +-1 there.

  The second-neighbor hopping of the Haldane term is the coupling times
  the z component of the cross product of the two bonds it goes through,
  sqrt(3)/2 times the square of the in-plane bond length, and the mass is
  3 sqrt(3) times that hopping. For a flat lattice of bond length 1 this
  is 4.5 times the coupling, and the coupling returned is exactly 1/4.5.
  In a buckled honeycomb the in-plane bond is shorter, and 1/4.5 gave a
  tau_z of eigenvalues +-8/9 for geometry.buckled_honeycomb_lattice, so
  the coupling is divided by the square of the in-plane bond length. With
  no first neighbors, or with in-plane bond lengths that are not all the
  same, the coupling stays 1/4.5."""
  from scipy.spatial import cKDTree
  rs = np.asarray(g.r)
  shifts = [np.zeros(3)]
  for a in [g.a1,g.a2,g.a3][:g.dimensionality]: # neighboring replicas
      shifts = [s+n*np.asarray(a) for s in shifts for n in (-1,0,1)]
  r2 = np.concatenate([rs+s for s in shifts])
  near = cKDTree(rs).query_ball_tree(cKDTree(r2),np.sqrt(1.01))
  ii = np.repeat(np.arange(len(rs)),[len(x) for x in near])
  jj = np.array([j for x in near for j in x],dtype=int)
  dr = rs[ii] - r2[jj]
  dr = dr[np.sum(dr*dr,axis=1)>0.99] # the first-neighbor bonds
  if len(dr)==0: return 1.0/4.5
  a2 = dr[:,0]**2 + dr[:,1]**2 # squared in-plane bond lengths
  if np.max(a2)-np.min(a2)>1e-6: return 1.0/4.5 # not one bond length
  if abs(np.mean(a2)-1.)<1e-9: return 1.0/4.5 # flat lattice
  return 1.0/(4.5*np.mean(a2))


def get_valley(h,delta=None,**kwargs):
  """Return a callable that calculates the valley expectation value
  using the modified Haldane coupling"""
  if h.dimensionality==0: projector = True # zero dimensional
  ho = h.copy() # copy Hamiltonian
  ho.turn_multicell()
  ho.clean() # set to zero
  ho.add_modified_haldane(valley_coupling(h.geometry)) # tau_z
  hkgen = ho.get_hk_gen() # get generator for the hk Hamiltonian
  sharpen = get_sharpen(delta=delta) # renormalize eigenvalues to +-1
  def fun(m=None,k=None):
      if h.dimensionality>0 and k is None: # requires a kpoint
        raise ValueError("the valley operator of a periodic Hamiltonian has "
                "to be evaluated at a k-point, and none was given")
      hk = hkgen(k) # evaluate Hamiltonian
      hk = sharpen(hk) # sharpen the valley
      if m is None: return hk # just return the valley operator
      else: return hk@m # return the projector
  if h.dimensionality==0: return fun() # return a matrix
  return fun # return function

