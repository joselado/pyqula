from __future__ import print_function,division
import numpy as np
import scipy.linalg as lg
from . import sculpt
from numba import jit



def non_orthogonal_supercell(gin,m,ncheck=2,mode="fill",reducef=lambda x: x):
  """Generate a non orthogonal supercell based on a tranformation
  matrix of the unit vectors, pretty much as VESTA does"""
  # workaround
  g = gin.copy()
  if g.dimensionality==0: return
  if g.dimensionality==1:
    g.a2 = np.array([0.,np.max(np.abs(gin.y))*2.+1.,0.])
    g.a3 = np.array([0.,0.,np.max(np.abs(gin.z))*2.+1.])
  if g.dimensionality==2:
    dz = np.max(np.abs(gin.z))*2.+1.
    g.a3 = np.array([0.,0.,dz])
  a1,a2,a3 = g.a1,g.a2,g.a3 # cell vectors
  go = g.copy() # output unit cell
  # new cell vectors
  go.a1 = m[0][0]*a1 + m[0][1]*a2 + m[0][2]*a3
  go.a2 = m[1][0]*a1 + m[1][1]*a2 + m[1][2]*a3
  go.a3 = m[2][0]*a1 + m[2][1]*a2 + m[2][2]*a3
  # calculate old and new volume
  vold = a1.dot(np.cross(a2,a3))  
  vnew = go.a1.dot(np.cross(go.a2,go.a3))  
  if abs(vnew)<0.0001: 
    print("No volume",vnew,"\n",a1,"\n",a2,"\n",a3)
    raise
  c = vnew/vold
  c = int(round(abs(c)))
  # now create replicas until there as c times as many atoms in the
  # unit cell
  if mode=="fill": # look for atoms until everything is filled
    rs = []
    k2K = go.get_k2K().I # matrix transformation
    R = np.matrix([go.a1,go.a2,go.a3]).T # transformation matrix
    L = R.I # inverse matrix
    d0 = -np.random.random()*0.1 # accuracy
    d0 = -0.122132112 # some random number
    d1 = 1.0 + d0 # accuracy
    from .geometry import neighbor_cells
# get as many cells as necessary
    cneigh = reducef(c) # cells to generate given the volume increase c
    cneigh = int(round(cneigh)) # integer
    inds = neighbor_cells(cneigh,dim=g.dimensionality) 
    sl = [] # list for the sublattice
    for (i,j,k) in inds: # loop
          ii = 0 # start count
          for ir in range(len(g.r)): # loop over positions
            ri = g.r[ir] # get the position
            store = False
            rj = ri + i*g.a1 + j*g.a2 + k*g.a3 # new position
            rn = L*np.matrix(rj).T  # transform
            rn = np.array([rn.T[0,ii] for ii in range(3)]) # convert to array
            n1,n2,n3 = rn[0],rn[1],rn[2]
            if g.dimensionality==3 and d0<n1<d1 and d0<n2<d1 and d0<n3<d1:
                store = True
            if g.dimensionality==2 and d0<n1<d1 and d0<n2<d1: 
                store = True
            if g.dimensionality==1 and d0<n1<d1: 
                store = True
            if store:
                rs.append(rj)
                if g.has_sublattice: sl.append(g.sublattice[ir])
#          if len(rs)==len(g.r)*c:
#            print("All the atoms found")
#            break
#    print(rs)
    go.r = np.array(rs) # store
    if go.has_sublattice: go.sublattice = sl # store sublattice
    if len(rs)!=len(g.r)*c: 
      print("Not all the atoms have been found")
      print("New atoms",len(rs))
      print("Expected atoms",len(g.r)*c)
      print("Volume of the cell increase",c)
      raise
  elif mode=="brute":
    if g.dimensionality==1:
      rs3 = replicate3d(g.r,g.a1,g.a2,g.a3,c,1,1) # new positions
    if g.dimensionality==2:
      rs3 = replicate3d(g.r,g.a1,g.a2,g.a3,c,c,1) # new positions
    if g.dimensionality==3:
      rs3 = replicate3d(g.r,g.a1,g.a2,g.a3,c,c,c) # new positions
    while True: # infinite loop, stop when scf reached
      rs1 = np.array(rs3) # store the first iteration
#      print(rs1)
      for i in range(-ncheck,ncheck+1):
        for j in range(-ncheck,ncheck+1):
          for k in range(-ncheck,ncheck+1):
            if i==0 and j==0 and k==0: continue
            rs2 = [ri + i*go.a1 + j*go.a2 + k*go.a3 for ri in rs1] # shift by this vector
            rs1 = return_unique(rs1,rs2) # return the unique positions
#            print(len(rs1),i,j,k)
      if len(rs1)==len(rs3): break
      rs3 = np.array(rs1) 
    go.r = np.array(rs1) # store new positions
    if go.has_sublattice: go.get_sublattice()
  go.r2xyz() # update coordinates
  go.center()
  go.get_fractional()
  return go # return new geometry
  

# from numba import jit


def replicate3d(rs,a1,a2,a3,n1,n2,n3):
  ro = np.zeros((n1*n2*n3*nc,3)) # allocate output array
  return replicate3d_jit(rs,a1,a2,a3,n1,n2,n3,ro) # compute

@jit(nopython=True)
def replicate3d_jit(rs,a1,a2,a3,n1,n2,n3,ro):
  """Function to make a three dimensional supercell"""
  nc = len(rs)
  ik = 0
  for i in range(n1):
    for j in range(n2):
      for l in range(n3):
        for k in range(nc):
          ro[ik] = a1*i + a2*j + a3*l + rs[k] # store position
          ik += 1 # increase counter
  return ro # return positions


#@jit(nopython=True)
#@jit
def return_unique(rs1,rs2):
  """Return only those positions in rs1 that do not appear in rs2"""
  rout = []
  for ri in rs1:
    drs = [(ri-rj).dot(ri-rj) for rj in rs2] # distances
    if np.array(drs).min() > 0.001: rout.append(ri) # store this position
  return np.array(rout)




#def request(g,nat,ntries=20):
#  """Request a unit cell with as many atoms"""


def target_angle_volume(g,angle=None,n=5,volume=None,same_length=False):
    """Return a supercell, targetting a certain new angle between vectors"""
    if g.dimensionality!=2: raise # only for 2d
    a1 = g.a1
    a2 = g.a2
    def getm(): # get the matrix
      out = [] # empty list
      vs = [] # volumes
      for i in range(-n,n+1):
        for j in range(-n,n+1):
          for k in range(-n,n+1):
            for l in range(-n,n+1):
                store = False
                a1n = i*a1 + j*a2
                a2n = k*a1 + l*a2
                v = lg.norm(np.cross(a1n,a2n)) # new volume
                v /= lg.norm(np.cross(a1,a2)) # new volume
                if abs(v)<1e-6: continue # zero volume
                u1 = a1n/np.sqrt(a1n.dot(a1n)) # normalize
                u2 = a2n/np.sqrt(a2n.dot(a2n)) # normalize
                if angle is not None: # check if it has the desired angle
                    diff = u1.dot(u2)-np.cos(angle*np.pi) # difference
                    if abs(diff)>1e-6: continue # next try 
                if same_length: # check if they must have the same length
                    diff = a1n.dot(a1n) - a2n.dot(a2n)
                    if abs(diff)>1e-6: continue # next try 
                if volume is not None: # target such volume
                    if abs(v-volume)>1e-6: continue
                    else: return [[i,j,0],[k,l,0],[0,0,1]]
                out.append([[i,j,0],[k,l,0],[0,0,1]]) # orthogonal, return
                vs.append(v) # volume
      if len(out)==0: return None # nothng found
      vs = np.array(vs) # as array
      return [o for (v,o) in sorted(zip(vs,out))][0]
    out = getm() # get rotation matrix
    if out is None: raise
    g = g.supercell(out)
    g = sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.])) # set in the x direction
    return g


target_angle = target_angle_volume

