# library for creating different kinds of junctions

import numpy as np
from . import geometry  # library with different geometries 
from . import hamiltonians  # hamiltonians library

class junction():
  def __init__(self):
    self.number = 3  



def generator_super_squid_square(width=1,inner_radius=4,arm_length=8,
          arm_width=2,delta=0.0,mag_field=0.0,fill=False):
  """ Creates a function that returns a superconducting 
     squid made of square lattice,
     central part is a normal system"""
  g = geometry.squid_square(width=width,inner_radius=inner_radius,
                              arm_length=arm_length, 
                              arm_width = arm_width,fill=fill)  # create the geometry
  g.dimensionality = 0 # set 0 dimensional
  g.celldis = None # no distance to neighboring cell
  h = hamiltonians.hamiltonian(g) # create hamiltonian
  nt = width + inner_radius+1. # half side of the big square
  def mag_field_f(x1,y1,x2,y2):
    """ Return magnetic field only in the center,
       using atan x gauga """
  #  return mag_field*(x1-x2)*(y1+y2)/2.0
  #  if (x1 or x2) < -float(nt): return 0.0
  #  if (x1 or x2) > float(nt): return 0.0
  #  else: return mag_field*(x1-x2)*(y1+y2)/2.0
    bb = (np.arctan(x1/nt)+np.arctan(x2/nt))/2.0  # mean x value
    bb = bb*(y2-y1)*mag_field
    return bb
  h.get_simple_tb(mag_field=mag_field_f)  # create simple tight binding
  def delta_f(x,y):
    """ Defines left and right arms arm """
    if x < -float(nt): return delta
    if x > float(nt): return delta
    else: return 0.0
  def h_generator(phi):
    """ Creates the function to return"""
    def phi_f(x,y):
      """ Defines left and right arms arm """
      if x < -float(nt): return phi
      if x > float(nt): return -phi
      else: return 0.0
    # add superconducting parameter
    h.remove_spin() # remove spin degree of freedom
    h.add_swave_electron_hole_pairing(delta=delta_f,phi=phi_f)
    h.set_finite_system() # transform to finite system
    return h # return hamiltonian
  return h_generator



def super_normal_super(g,f_sc,f_n,replicas = [1,1,1],phi=0.0,delta=0.1):
  """Creates a junction superconductor-normal-superconductor,
          - g is the geometry of the unit cell of the system
          - f_sc is the function which generates the superconductor
            which takes three arguments, geometry, SC order and phase
          - f_n generates the hamiltonian of the central part,
            result has to have electron-hole dimension
          - replicas are the replicas of the SC-normal_SC of the cell
          - phi is the infinitesimal phase difference
          - delta is the superconducting parameter """
  h_left = f_sc(g,delta=delta,phi=phi) # create hamiltonian of the left part
  h_right = f_sc(g,delta=delta,phi=-phi) # create hamiltonian of the right part
  h_central = f_n(g)  # central part
  from scipy.sparse import coo_matrix as coo
  from scipy.sparse import bmat
  from hamiltonians import hamiltonian
  hj = hamiltonian(g) # create hamiltonian of the junction
  tc = replicas[0] + replicas[1] + replicas[2] # total number of cells
  intra = [[None for i in range(tc)] for j in range(tc)]  # create intracell


  # transfor to coo matrix
  h_left.intra = coo(h_left.intra)
  h_left.inter = coo(h_left.inter)
  h_central.intra = coo(h_central.intra)
  h_central.inter = coo(h_central.inter)
  h_right.intra = coo(h_right.intra)
  h_right.inter = coo(h_right.inter)


  # intracell contributions
  for i in range(replicas[0]): 
    intra[i][i] = h_left.intra # intracell of the left lead
  init = replicas[0] # inital index
  for i in range(replicas[1]): 
    intra[init+i][init+i] = h_central.intra # intracell of the left lead
  init = replicas[0]+replicas[1] # inital index
  for i in range(replicas[2]): 
    intra[init+i][init+i] = h_right.intra # intracell of the left lead

  # intercell contributions
  for i in range(replicas[0]-1): 
    intra[i][i+1] = h_left.inter # intercell of the left lead
    intra[i+1][i] = h_left.inter.H # intercell of the left lead
  init = replicas[0] # inital index
  for i in range(replicas[1]-1): 
    intra[init+i][init+i+1] = h_central.inter # intercell of the central lead
    intra[init+i+1][init+i] = h_central.inter.H # intercell of the central lead
  init = replicas[0]+replicas[1] # inital index
  for i in range(replicas[2]-1): 
    intra[init+i][init+i+1] = h_right.inter # intercell of the right lead
    intra[init+i+1][init+i] = h_right.inter.H # intercell of the right lead

  # coupling between SC and central part, take couapling of the central
  il = replicas[0]
  intra[il-1][il] = h_central.inter # intracell of the left lead
  intra[il][il-1] = h_central.inter.H # intracell of the left lead
  il = replicas[0]+replicas[1]
  intra[il-1][il] = h_central.inter # intracell of the left lead
  intra[il][il-1] = h_central.inter.H # intracell of the left lead
  # create matrix
  intra = bmat(intra).todense()
  hj.intra = intra # add to hamiltonian junction
  hj = hj.finite_system() # convert in finite system
  return hj

