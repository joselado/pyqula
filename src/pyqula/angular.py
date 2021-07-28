from __future__ import print_function
import numpy as np


def ylm2xyz_l1():
  """Return the matrix that converts the cartesian into spherical harmonics"""
  m = np.matrix([[0.0j for i in range(3)] for j in range(3)])
  s2 = np.sqrt(2.)
  m[1,2] = 1. # pz
  m[0,0] = 1./s2 # dxz
  m[2,0] = -1./s2 # dxz
  m[0,1] = 1j/s2 # dyz
  m[2,1] = 1j/s2 # dyz
  return m


def ylm2xyz_l2():
  """Return the matrix that converts the cartesian into spherical harmonics"""
  m = np.matrix([[0.0j for i in range(5)] for j in range(5)])
  s2 = np.sqrt(2.)
  m[2,0] = 1. # dz2
  m[1,1] = 1./s2 # dxz
  m[3,1] = -1./s2 # dxz
  m[1,2] = 1j/s2 # dyz
  m[3,2] = 1j/s2 # dyz
  m[0,3] = 1j/s2 # dxy
  m[4,3] = -1j/s2 # dxy
  m[0,4] = 1./s2 # dx2y2
  m[4,4] = 1./s2 # dx2y2
  return m # return change of bassi matrix


# names of orbitals
dorbs = ["dz2","dxz","dyz","dxy","dx2-y2"]
porbs = ["px","py","pz"]
sorbs = ["s"]

def names_soc_orbitals(specie):
  """Names of the orbitals, order coherent with angular change of basis"""
  name = get_element(specie) # get the true name of the atom
  datoms = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
  datoms += ["Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd"]
  datoms += ["Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"]
  patoms = ["B","C","N","O","F","P"]
  patoms += ["Al","Si","P","S","Cl"]
  patoms += ["Ga","Ge","As","Se","Br"]
  patoms += ["In","Sn","An","Te","I"]
  patoms += ["Tl","Pb","Bi","Po","At"]
  if name in datoms: return dorbs # for dorbitals
  elif name in patoms: return porbs # for dorbitals
  else: raise



def get_element(name):
  """Gets the true name of an element, assumes that there might be a number"""
  out = "" # initialize
  if name[-1] in [str(i) for i in range(10)]:
    for i in range(len(name)-1): # except the last one
      out += name[i] # add to the string
    return get_element(out) # recall the function, just in case two digits
  return name




def angular_momentum(orbs):
  """Return the angular momentum operator in a certain basis
  of cartesian operators"""
  for o in orbs:
    if o in sorbs: # check if they are s orbitals
      l=0
      orbnames = sorbs
      break
  for o in orbs:
    if o in porbs: # check if they are s orbitals
      l=1
      orbnames = porbs
      break
  for o in orbs:
    if o in dorbs: # check if they are s orbitals
      l=2
      orbnames = dorbs
      break
  nm = 2*l + 1 # number of components
  zero = np.matrix([[0.0j for i in range(nm)] for j in range(nm)])
  # initialize matrices
  lz = zero.copy()
  lx = zero.copy()
  ly = zero.copy()
  lm = zero.copy()
  lp = zero.copy()
  # create l+ and l- and lz
  for m in range(-l,l): # loop over m components
    val = np.sqrt((l-m)*(l+m+1)) # value of the cupling
    im = m + l
    lp[(im+1),im] = val # up channel
  for m in range(-l,l+1): # loop over m components
    im = m + l
    lz[im,im] = m # value of lz, up channel
  lm = lp.H # adjoint
  lx = (lp + lm) /2.
  ly = -1j*(lp - lm) /2.
  # at this point lx,ly,lz are written in the basis of spherical harmonics
  if l==2: R = ylm2xyz_l2() # get change of basis matrix
  if l==1: R = ylm2xyz_l1() # get change of basis matrix
  lx = R.H * lx * R # change to cartesian orbitals
  ly = R.H * ly * R # change to cartesian orbitals
  lz = R.H * lz * R # change to cartesian orbitals
  
  # at this point the angular operators are written in the cartesian basis
  # now you must project onto the input manifold
  proj = np.matrix([[0.0j for i in range(nm)] for j in range(len(orbs))])
  for i in range(len(orbs)): # loop over input cartesian orbitals
    for j in range(len(orbnames)): # loop over full cartesian orbitals
      if orbs[i] == orbnames[j]: # if same orbital
        proj[i,j] = 1. # identity
#        print(orbs[i],orbnames[j],i,j) 
        break
#      print(orbs,orbnames,i,j)
#      raise # raise error if this point is reached
#  print(proj)
#  raise
  proj = proj.H
  # now project
  lx = proj.H * lx * proj # change to cartesian orbitals
  ly = proj.H * ly * proj # change to cartesian orbitals
  lz = proj.H * lz * proj # change to cartesian orbitals
  return (lx,ly,lz) # return the three matrices




