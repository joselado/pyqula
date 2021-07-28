import os
import numpy as np
from . import sculpt
from . import hamiltonians
from . import geometry



def generate_mullen_island(nx,ny,pristine=False):
  """Generate a Mullen island"""
  g = geometry.honeycomb_zigzag_ribbon(ny)
  g = g.supercell(2*nx+1+2)
  g.dimensionality = 0
  minx = np.min(g.x)
  
  def finite(x,y):
    if x<minx+.1: return False
    return True
  
  g = sculpt.intersec(g,finite)
  
  g = sculpt.remove_unibonded(g)
  
  
  g.center()
  
  # now add the hexagon
  
  dy = max(g.y) + 1.9
  
  r1 = np.array([-1.,0.+dy,0.])
  r2 = np.array([1.,0.+dy,0.])
  r3 = np.array([.5,np.sqrt(3.0)/2.+dy,0.])
  r4 = np.array([-.5,np.sqrt(3.0)/2.+dy,0.])
  r5 = np.array([.5,-np.sqrt(3.0)/2.+dy,0.])
  r6 = np.array([-.5,-np.sqrt(3.0)/2.+dy,0.])

  if not pristine:  
    g.r = [ri for ri in g.r] + [r1,r2,r3,r4,r5,r6]
    g.r = [ri for ri in g.r] + [-r1,-r2,-r3,-r4,-r5,-r6]
  #g.r =  [r1,r2,r3,r4,r5,r6]
  g.r = np.array(g.r)
  g.r2xyz()
  
  g.write()
  
  h = g.get_hamiltonian() # initialize hamiltonian
  
  def funh(ri,rj):
    dr = ri-rj
    if .1<dr.dot(dr)<1.3: return 1.0
    else: return 0.0
  
  
  h = hamiltonians.generate_parametric_hopping(h,funh)
  return h


def generate_mullen_ribbon(nx,ny,single=False):
  """Generate a Mullen island"""
  g = geometry.honeycomb_zigzag_ribbon(ny)
  g = g.supercell(2*nx+2)
  
  
  # now add the hexagon
  
  dy = max(g.y) + 1.9
  dx = np.sqrt(3.)/4 +0.
  xx = np.array([dx,0.,0.]) 
  g.r = [ri+xx for ri in g.r] # displace whole
  r1 = np.array([-1.,0.+dy,0.])
  r2 = np.array([1.,0.+dy,0.])
  r3 = np.array([.5,np.sqrt(3.0)/2.+dy,0.])
  r4 = np.array([-.5,np.sqrt(3.0)/2.+dy,0.])
  r5 = np.array([.5,-np.sqrt(3.0)/2.+dy,0.])
  r6 = np.array([-.5,-np.sqrt(3.0)/2.+dy,0.])
  
  g.r = [ri for ri in g.r] + [r1,r2,r3,r4,r5,r6] # upper
  if not single: g.r = [ri for ri in g.r] + [-r1,-r2,-r3,-r4,-r5,-r6] # lower
  #g.r =  [r1,r2,r3,r4,r5,r6]
  g.r = np.array(g.r)
  g.r2xyz()
  
  g.write()
  
  h = g.get_hamiltonian() # initialize hamiltonian
  
  def funh(ri,rj):
    dr = ri-rj
    if .1<dr.dot(dr)<1.3: return 1.0
    else: return 0.0
  
  
  h = hamiltonians.generate_parametric_hopping(h,funh)
  return h


def generate_mullen_surface(nx,ny):
  """Generate a Mullen surface, return bulk hamiltonian and surface cell"""
  g = geometry.honeycomb_lattice_zigzag_cell()
  g = g.supercell([2*nx+2+1,1]) # two dimensional
  gb = g.copy() # copy geometry
  g.dimensionality = 1 # reduce dimensionality
  g = sculpt.remove_unibonded(g)
  gb = sculpt.remove_unibonded(gb)
  g.center()
  gb.center()
  
  # now add the hexagon
  
  dy = max(g.y) + 1.9
  dx = 0.
  xx = np.array([dx,0.,0.]) 
  g.r = [ri+xx for ri in g.r] # displace whole
  r1 = np.array([-1.,0.+dy,0.])
  r2 = np.array([1.,0.+dy,0.])
  r3 = np.array([.5,np.sqrt(3.0)/2.+dy,0.])
  r4 = np.array([-.5,np.sqrt(3.0)/2.+dy,0.])
  r5 = np.array([.5,-np.sqrt(3.0)/2.+dy,0.])
  r6 = np.array([-.5,-np.sqrt(3.0)/2.+dy,0.])
  g.r = [ri for ri in g.r] + [r1,r2,r3,r4,r5,r6] # upper
  g.r = np.array(g.r)
  g.r2xyz()
  
  g.write()
  
  h = g.get_hamiltonian() # initialize hamiltonian

  def funh(ri,rj):
    dr = ri-rj
    if .1<dr.dot(dr)<1.3: return 1.0
    else: return 0.0
  
  
  h = hamiltonians.generate_parametric_hopping(h,funh)
  h.remove_spin() # remove spin degree of freedom 
  
  hb = gb.get_hamiltonian() # bulk hamiltonian
  hb.remove_spin() # remove spin
 
  return (h,hb) 





def generate_pentagon_island(nx,ny,pentagons=[0]):
  """Generate a Mullen island"""
  g = geometry.honeycomb_zigzag_ribbon(ny)
  g = g.supercell(2*nx+1)
  g.dimensionality = 0
  minx = np.min(g.x)
  
  def finite(x,y):
    if x<minx+.1: return False
    return True
  
  g = sculpt.intersec(g,finite)
  
  g = sculpt.remove_unibonded(g)
  
  
  g.center()
  
  # now add the hexagon
  
  dy = max(g.y) + 1.9

  ym = max(g.y) # position of zigzag atoms
  xz = [] # list of zigzag atoms

  for ir in g.r: # loop over atoms
    if (ir[1]-ym)<0.1: xz.append(ir[0]) # store upper zigzag atom
  for ip in pentagons: # loop over pentagons 
    # generate positions
    r1 = np.array([-1.+dx,0.+dy,0.])
    r2 = np.array([1.+dx,0.+dy,0.])
    r3 = np.array([.5+dx,np.sqrt(3.0)/2.+dy,0.])
    r4 = np.array([-.5+dx,np.sqrt(3.0)/2.+dy,0.])
    r5 = np.array([.5+dx,-np.sqrt(3.0)/2.+dy,0.])
    r6 = np.array([-.5+dx,-np.sqrt(3.0)/2.+dy,0.])
  
    # add the extra hexagons
    g.r = [ri for ri in g.r] + [r1,r2,r3,r4,r5,r6]
    g.r = [ri for ri in g.r] + [-r1,-r2,-r3,-r4,-r5,-r6]
  #g.r =  [r1,r2,r3,r4,r5,r6]
  g.r = np.array(g.r)
  g.r2xyz()
  
  g.write()
  
  h = g.get_hamiltonian() # initialize hamiltonian
  
  def funh(ri,rj):
    dr = ri-rj
    if .1<dr.dot(dr)<1.3: return 1.0
    else: return 0.0
  
  
  h = hamiltonians.generate_parametric_hopping(h,funh)
  return h







