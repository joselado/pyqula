from __future__ import print_function,division
import numpy as np
from . import geometry
from . import sculpt

def twisted_bilayer(m0,**kwargs):
    """Geomtry of a twisted bilayer"""
    return twisted_multilayer(m0=m0,rot=[1,0],**kwargs)

#def twisted_bilayer(m0=3,rotate=True,shift=[0.,0.],center="AB/BA",
#  sublattice=True,r=1,g=None,dz=1.5):
#  """Return the geometry for twisted bilayer graphene"""
#  if g is None: g = geometry.honeycomb_lattice()
#  g.has_sublattice = False
#  if sublattice: # trace the sublattice using a dirty trick
#    g.z[0] += 0.001
#    g.z[1] -= 0.001
#    g.xyz2r()
#  else: pass
#  g = geometry.non_orthogonal_supercell(g,m=[[-1,0,0],[0,1,0],[0,0,1]])
##  m0 = 3
##  r = 1
#  theta = np.arccos((3.*m0**2+3*m0*r+r**2/2.)/(3.*m0**2+3*m0*r+r**2))
#  print("Theta",theta*180.0/np.pi)
#  g.angle = theta # terrible workaround
#  g.data["angle"] = theta # store the angle of the geometry
#  nsuper = [[m0,m0+r,0],[-m0-r,2*m0+r,0],[0,0,1]]
#  g = geometry.non_orthogonal_supercell(g,m=nsuper,
#           reducef=lambda x: 3*np.sqrt(x))
#  g1 = g.copy()
#  g1.shift([1.,0.,0.]) 
#  g.z -= dz
#  g.xyz2r() # update
#  if rotate: # rotate one of the layers
#    g1 = g1.rotate(theta*180/np.pi)
#    g1s = g1.supercell(2) # supercell
#    g1s.z += dz
#    g1s.x += shift[0] # shift upper layer
#    g1s.y += shift[1] # shift upper layer
#    g1s.xyz2r() # update
#    g1.a1 = g.a1
#    g1.a2 = g.a2
#    rs = sculpt.retain_unit_cell(g1s.r,g.a1,g.a2,g.a3,dim=2) # get positions
#  else: # do not rotate
#    rs = np.array(g1.r) # same coordinates
#    rs[:,2] = 1.0 # set as one
#  g1.r = np.concatenate([rs,g.r])
# 
#  g1.r2xyz() # update
#  g1.real2fractional() # update fractional coordinates 
#  if center=="AB/BA": pass # do nothing 
#  elif center=="AA": 
#    g1.frac_x = (g1.frac_x)%1 # to the unit cell
#    g1.fractional2real()
#  elif center=="AB": 
#    g1.frac_x = (g1.frac_x)%1 # to the unit cell
#    g1.frac_y = (g1.frac_y)%1 # to the unit cell
#    g1.frac_y += 0.5  # to the unit cell
#    g1.frac_x = (g1.frac_x)%1 # to the unit cell
#    g1.frac_y = (g1.frac_y)%1 # to the unit cell
#    g1.frac_x -= 1./3.
#    g1.frac_y -= 1./3.
#    g1.frac_x = (g1.frac_x)%1 # to the unit cell
#    g1.frac_y = (g1.frac_y)%1 # to the unit cell
#    g1.fractional2real()
#  else: raise
#  if sublattice: # recover the sublattice
#    g1.has_sublattice = True # say that it has
#    sl = []
#    for r in g1.r: # loop over positions
#      if np.abs(r[2]-1.5)<0.01: # upper layer
#        if r[2]-1.5>0.0: sl.append(-1.) # A sublattice
#        else: sl.append(1.) # B sublattice
#      elif np.abs(r[2]+1.5)<0.01: # lower layer
#        if r[2]+1.5>0.0: sl.append(-1.) # A sublattice
#        else: sl.append(1.) # B sublattice
#    g1.z = np.round(g1.z,2) # delete the small shift
#    g1.xyz2r() # update coordinates
#    g1.sublattice = np.array(sl) # store sublattice
#  g1 = sculpt.rotate_a2b(g1,g1.a1,np.array([1.,0.,0.])) # rotate
#  g1.get_fractional() # get fractional coordinates 
#  return g1


def twisted_supercell(g,m0=3,r=1):
  """Return the supercell for a twisted system"""
  g.has_sublattice = False # no sublattice
  g = geometry.non_orthogonal_supercell(g,m=[[-1,0,0],[0,1,0],[0,0,1]])
  theta = np.arccos((3.*m0**2+3*m0*r+r**2/2.)/(3.*m0**2+3*m0*r+r**2))
  print("Theta",theta*180.0/np.pi)
  nsuper = [[m0,m0+r,0],[-m0-r,2*m0+r,0],[0,0,1]]
  g = geometry.non_orthogonal_supercell(g,m=nsuper,
           reducef=lambda x: 3*np.sqrt(x))
  return g




def twisted_multilayer(m0=3,rotate=True,shift=None,
  sublattice=True,r=1,rot=[1,1,0,0],g=None,dz=3.0):
  """Return the geometry for twisted multilayer graphene"""
  if g is None: g = geometry.honeycomb_lattice() # default is honeycomb
#  g.has_sublattice = False # no sublattice
  g = geometry.non_orthogonal_supercell(g,m=[[-1,0,0],[0,1,0],[0,0,1]])
  g0 = g.copy() # copy geometry
  theta = np.arccos((3.*m0**2+3*m0*r+r**2/2.)/(3.*m0**2+3*m0*r+r**2))
  if shift is None:
      shift = [[0.,0.] for r in rot] # initialize
  print("Theta",theta*180.0/np.pi)
  nsuper = [[m0,m0+r,0],[-m0-r,2*m0+r,0],[0,0,1]]
  g = geometry.non_orthogonal_supercell(g,m=nsuper,
           reducef=lambda x: 3*np.sqrt(x))
  if rotate: # rotate one of the layers
    gs = [] # empty list with geometries
    ii = 0
    for i in rot: # loop
        if i!=0 and i!=1: raise # nope
        dr = shift[ii][0]*g0.a1 + shift[ii][1]*g0.a2 # shift geometry
        dr = dr + np.array([0.,0.,dz*(ii-len(rot)/2.+.5)]) # shift layer
        gs.append(rotate_layer(g,i*theta,dr=dr))
      #  gs.append(g.copy())
        ii += 1 # increase counter
  else: raise
#  g.r = np.concatenate([g1.r,g.r,g2.r]).copy()
  g.r = np.concatenate([gi.r for gi in gs]).copy() # all the positions
  if g.has_sublattice:
    g.sublattice = np.concatenate([gi.sublattice for gi in gs]).copy() 
#  g.r = np.concatenate([g2.r,g.r]).copy()
 # g.r = g1.r
  g.r2xyz() # update
  g.real2fractional() # update fractional coordinates 
  g = sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.])) # rotate
  g.get_fractional() # get fractional coordinates 
  return g









def twisted_trilayer(m0=3):
  return twisted_multilayer(m0=m0,shift=[-1.,0.,1.])


def generalized_twisted_multilayer(m0=3,rotate=True,shift=[0.,0.],
  sublattice=True,r=1,rot=[1,1,0,0],gf=None,dz=3.0):
  """Return the geometry for twisted bilayer graphene"""
  theta = np.arccos((3.*m0**2+3*m0*r+r**2/2.)/(3.*m0**2+3*m0*r+r**2))
  if rotate: # rotate one of the layers
    gs = [] # empty list with geometries
    ii = 0
    for i in rot: # loop
        g = gf[ii] # get the geometry
        g = twisted_supercell(g,m0=m0,r=r) # get the twisted supercell
        print(i)
        if i!=0 and i!=1: raise # nope
        gs.append(rotate_layer(g,i*theta,dr=[0.,0.,dz*(ii-len(rot)/2.+.5)]))
        ii += 1 # increase counter
  else: raise
#  g.r = np.concatenate([g1.r,g.r,g2.r]).copy()
  g.r = np.concatenate([gi.r for gi in gs]).copy() # all the positions
#  g.r = np.concatenate([g2.r,g.r]).copy()
 # g.r = g1.r
  g.r2xyz() # update
  g.real2fractional() # update fractional coordinates 
  g = sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.])) # rotate
  g.get_fractional() # get fractional coordinates 
  return g






def rotate_layer(g,theta,dr=[0.,0.,0.]):
    """Rotate one of the layers, retainign the same unit cell"""
    g1 = g.copy() # copy geometry
    g1 = g1.rotate(theta*180/np.pi)
    g1s = g1.supercell(2) # supercell
#    g1.a1 = g.a1.copy()
#    g1.a2 = g.a2.copy()
    inuc = sculpt.sites_in_unit_cell(g1s.r,g.a1,g.a2,g.a3,dim=2)
    g1.r = g1s.r[inuc==1.]
    if g1.has_sublattice:
        g1.sublattice = g1s.sublattice[inuc==1.]
#    g1.r = sculpt.retain_unit_cell(g1s.r,g.a1,g.a2,g.a3,dim=2) # get positions
    g1.r2xyz() # update
    g1.x += dr[0] # shift upper layer
    g1.y += dr[1] # shift upper layer
    g1.z += dr[2] # shift upper layer
    g1.xyz2r() # update
    return g1 # return the geometry

def multilayer_graphene(**kwargs):
    """Return a multilayer graphene geometry"""
    g = geometry.honeycomb_lattice() # get a honeycomb lattice
    dr = g.r[0] - g.r[1] # distance
    return multilayer(g,dr=dr,dz=[0.,0.,3.],**kwargs)


def multilayer(g,l=[0],dr=[1.,0.,0.],dz=[0.,0.,1.]):
    """Return a multilayer geometry"""
    dr = np.array(dr) # to array
    dz = np.array(dz) # to array
    ss = [] # list for the sublattice
    rs = [] # list for the positions
    ii = 0 # start
    for il in l: # loop over layers
        if g.has_sublattice:
            for (ri,si) in zip(g.r,g.sublattice): # loop over positions
                rs.append(ri + dr*il + ii*dz) # add position
                ss.append(si*(-1)**(il+ii)) # add sublattice
        else:
            for ri in g.r: # loop over positions
                rs.append(ri + dr*il + ii*dz) # add position
        ii += 1
    go = g.copy()
    go.r = np.array(rs)
    if g.has_sublattice: go.sublattice = np.array(ss)
    go.r2xyz()
    go.center()
    go = sculpt.rotate_a2b(go,go.a1,np.array([1.,0.,0.]))
    return go



def mismatched_lattice(n1=5,n2=4,g=None):
    if g is None: g = geometry.honeycomb_lattice()
    g1 = g.supercell(n1) # graphene
    g2 = g.supercell(n2) # BN
    g1.z += 1.5 # shift
    g2.z -= 1.5 # shift
    g1.xyz2r() # update
    g2.xyz2r() # update
    
    gn = g1.copy() # new geometry
    if g.has_sublattice: 
        gn.sublattice = np.concatenate([g1.sublattice,g2.sublattice])
    scale = n1/n2 # scale
    g2.x *= scale # scale
    g2.y *= scale # scale
    g2.xyz2r() # update
    gn.r = np.concatenate([g1.r,g2.r]) # concatenate
    gn.r2xyz() # update
    gn.get_fractional() # update fractional coordinates
    return gn # get geometry



def twisted_multimultilayer(m0=3,
  r=1,rot=[1,0],
  g=None,dz=3.0):
  """Return the geometry for twisted multimultilayer graphene"""
  if g is None: # nothing provided, assume twisted bilayer
    g = [] # empty list
    for i in rot:
      gi = geometry.honeycomb_lattice() # default is honeycomb
      g.append(gi) # store geometry
  g0 = [] # empt list
  for gi in g:
      gi = geometry.non_orthogonal_supercell(gi,m=[[-1,0,0],[0,1,0],[0,0,1]])
      g0.append(gi) # store
  g = g0 # overwrite
  # define the rotation angle
  theta = np.arccos((3.*m0**2+3*m0*r+r**2/2.)/(3.*m0**2+3*m0*r+r**2))
  print("Theta",theta*180.0/np.pi)
  # supercell used
  nsuper = [[m0,m0+r,0],[-m0-r,2*m0+r,0],[0,0,1]]
  gs = [] # empty list
  if len(rot)!=len(g): raise # inconsistency
  zshift= 0.0 # initial shift
  for (irot,gi) in zip(rot,g): # loop
    dzi = np.max(gi.z)-np.min(gi.z) # width of this layer
    gi = geometry.non_orthogonal_supercell(gi,m=nsuper,
           reducef=lambda x: 3*np.sqrt(x))
    gi.r[:,2] -= np.min(gi.r[:,2]) # lowest layer in zero
    gi.r2xyz() # update
    if irot!=0 and irot!=1: raise # nope
    gs.append(rotate_layer(gi,irot*theta,dr=[0.,0.,zshift]))
    zshift = zshift + dzi + dz
  # now create the final geometry object
  del g # remove that list
  g = gs[0].copy() # overwrite g
  g.r = np.concatenate([gi.r for gi in gs]).copy() # all the positions
  g.sublattice = np.concatenate([gi.sublattice for gi in gs]).copy() # all the positions
  g.r2xyz() # update
  g.real2fractional() # update fractional coordinates 
  g = sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.])) # rotate
  g.get_fractional() # get fractional coordinates 
  g.data["angle_degrees"] = theta*180.0/np.pi # store the angle of the geometry
  return g



def parse_twisted_multimultilayer(name,n=3):
    """Given a name for a multilayer, return the geometry"""
    # name is two element list
    # second element is the list with aligned stacking
    # thirs element determines whether if they are rotated
    gs = [] # empty list
    for ni in name[0]: # loop over names of the layers
            if ni in ["","A"]: gi = multilayer_graphene([0]) # normal honeycomb
            elif ni=="AA": gi = multilayer_graphene([0,0]) 
            elif ni=="AB": gi = multilayer_graphene([0,1]) 
            elif ni=="BA": gi = multilayer_graphene([1,0]) 
            elif ni=="ABC": gi = multilayer_graphene([0,1,2]) 
            elif ni=="ABAB": gi = multilayer_graphene([0,1,0,1]) 
            elif ni=="ABBA": gi = multilayer_graphene([0,1,1,0]) 
            else: raise
            gs.append(gi) # store
    rot = name[1] # rotations
    return twisted_multimultilayer(rot=rot,g=gs,m0=n)



def multilayer_codes(n=3):
    """Return all the codes for aligned multilayers
    up to n layers"""
    out = []
    import itertools 
    for i in range(n): # loop over layer numbers
        o = list(itertools.product('ABC', repeat=i))
        o = ["A"+"".join(io) for io in o]
        out += o
    return out





