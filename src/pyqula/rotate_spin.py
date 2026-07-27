import numpy as np
import scipy.linalg as lg
from . import algebra


from scipy.sparse import csc_matrix,bmat
from .spin import sx,sy,sz


def rotation_matrix(m,vectors):
    """ Rotates a matrix, to align its components with the direction
    of the magnetism """
    if not len(m)==2*len(vectors): # stop if they don't have
                                    # compatible dimensions
       raise
    # pauli matrices
    n = len(m)//2 # number of sites
    R = [[None for i in range(n)] for j in range(n)] # rotation matrix
    from scipy.linalg import expm  # exponenciate matrix
    for (i,v) in zip(range(n),vectors): # loop over sites
        vv = np.sqrt(v.dot(v)) # norm of v
        if vv>1e-8: # if nonzero scale
            u = v/vv
            uxy = np.sqrt(u[0]**2 + u[1]**2) # component in xy plane
            phi = np.arctan2(u[1],u[0]) # phi axis
            theta = np.arctan2(uxy,u[2]) # angle with respect to z axis
            r1 =  phi*sz/2.0 # rotate along z
            r2 =  theta*sy/2.0 # rotate along y
            # a factor 2 is taken out due to 1/2 of S
            rot = algebra.expm(1j*r1) @ algebra.expm(-1j*r2)   
        else: # if zero vector, no rotation
            rot = np.identity(2) # just no rotation
        R[i][i] = rot  # save term
    R = algebra.bmat(R)  # convert to full sparse matrix
    return algebra.todense(R)



def align_magnetism(m,vectors):
  """ Align matrix with the magnetic moments"""
  R = rotation_matrix(m,vectors) # return the rotation matrix
  Rh = np.conjugate(R).T
  mout = Rh @ m @ R  # rotate matrix
  return algebra.todense(mout) # return dense matrix





def build_rotation_matrix(n,vector = np.array([0.,0.,1.]),angle = 0.0):
  """ Build the full n-site block-diagonal spin rotation matrix for a
  global spin rotation by `angle` about `vector` (same convention as
  global_spin_rotation, which uses this internally). Split out so that a
  caller applying the *same* fixed rotation to many matrices (e.g. every
  iteration of an SCF loop) can build R once and reuse it, instead of
  recomputing the matrix exponential on every call. """
  u = np.array(vector) # rotation direction
  u = u/np.sqrt(u.dot(u)) # normalize rotation direction
  rot = (u[0]*sx + u[1]*sy + u[2]*sz)/2. # rotation
  # a factor 2 is taken out due to 1/2 of S
  # a factor 2 is added to have BZ in the interval 0,1
  rot = algebra.todense(rot)
  rot = lg.expm(2.*np.pi*1j*rot*angle/2.0)
  # same rotation at every site, so this is just a repeated block-diagonal;
  # np.kron avoids scipy.sparse.bmat, which mishandles the n=1 (single
  # site per cell) case (raises "blocks must be 2-D")
  return np.kron(np.eye(n),rot) # full rotation matrix


def global_spin_rotation(m,vector = np.array([0.,0.,1.]),angle = 0.0,
                             spiral = False,atoms = None):
  """ Rotates a matrix along a certain qvector """
  n = m.shape[0]//2 # number of sites
  if atoms is not None: raise # per-atom rotation not implemented
  R = build_rotation_matrix(n,vector=vector,angle=angle)
  if spiral:  # for spin spiral
    mout = R @ m  # rotate matrix
  else:  # normal global rotation
    mout = R @ m @ algebra.dagger(R)  # rotate matrix
  return mout # return dense matrix




def spiralhopping(m,ri,rj,svector = np.array([0.,0.,1.]),
        qvector=[1.,0.,0.]): 
  """ Rotates a hopping matrix to create a spin spiral
  antsaz
      - ri and rj must be coordinates in lattice constants
      - svector is the axis of the rotation
      - qvector is the vector of the spin spiral
  """
  from scipy.sparse import csc_matrix,bmat
  iden = csc_matrix([[1.,0.],[0.,1.]]) # identity matrix
  def getR(r):
      """Return a rotation matrix"""
      n = len(r) # number of sites 
      R = [[None for i in range(n)] for j in range(n)] # rotation matrix
      u = np.array(svector) # rotation direction
      u = u/np.sqrt(u.dot(u)) # normalize rotation direction
      for i in range(n): # loop over sites
         rot = u[0]*sx + u[1]*sy + u[2]*sz 
         angle = np.array(qvector).dot(np.array(r[i])) # angle of rotation
         # a factor 2 is taken out due to 1/2 of S
         # a factor 2 is added to have BZ in the interval 0,1
         R[i][i] = algebra.expm(2.*np.pi*1j*rot*angle/2.0)
      return algebra.bmat(R)  # convert to full sparse matrix
  Roti = getR(ri) # get the first rotation matrix
  Rotj = getR(rj) # get the second rotation matrix
#  print(Roti@Rotj.H)
#  print(ri,rj)
  return Rotj @ m @ algebra.dagger(Roti) # return the rotated matrix


def hamiltonian_spin_rotation(self,vector=np.array([0.,0.,1.]),angle=0.):
    """ Perform a global spin rotation.

    Also correct for BdG (Nambu, has_eh=True) Hamiltonians: pyqula's Nambu
    convention (sctk/reorder.py's block2nambu) groups each site's electron
    pair and hole pair as consecutive (up,down)-like 2-blocks, so
    global_spin_rotation's n=m.shape[0]//2, kron(eye(n),rot) construction
    already applies the same rotation to every one of those blocks
    (electron pair and hole pair alike) with no changes needed -- verified
    numerically (eigenvalue-preserving, and matches rotating the physical
    exchange/pairing directly) against Hamiltonians with both an exchange
    field and s-wave pairing present. """
    if not self.has_spin: raise # no spin in the Hamiltonian
    gsr = global_spin_rotation # rename method
    self.intra = gsr(self.intra,vector=vector,angle=angle)
    if self.is_multicell: # multicell hamiltonian
      for i in range(len(self.hopping)): # loop 
        self.hopping[i].m = gsr(self.hopping[i].m,vector=vector,angle=angle)
    else:
      if self.dimensionality==0: pass
      elif self.dimensionality==1:
        self.inter = gsr(self.inter,vector=vector,angle=angle)
      elif self.dimensionality==2:
        self.tx = gsr(self.tx,vector=vector,angle=angle)
        self.ty = gsr(self.ty,vector=vector,angle=angle)
        self.txy = gsr(self.txy,vector=vector,angle=angle)
        self.txmy = gsr(self.txmy,vector=vector,angle=angle)
      else: raise



def generate_spin_spiral(self,vector=np.array([0.,0.,1.]),
                            qspiral=[1.,0.,0.],fractional=True,
                            **kwargs):
    """
    Generate a spin spiral antsaz in the Hamiltonian
    """
    if not self.has_spin: raise # no spin
    qspiral = np.array(qspiral) # to array
    if qspiral.dot(qspiral)<1e-7: qspiral = np.array([0.,0.,0.])
    self.geometry.get_fractional()
    def tmprot(m,vec): # function used to rotate
      """Function to rotate one matrix"""
      if fractional: # fractional coordinates provided
        # rotate fractional coordinates
        ri = self.geometry.frac_r # positions of the first cell
        rj = self.geometry.frac_r + np.array(vec) # positions of the next cell
        return spiralhopping(m,ri,rj,svector=vector,
                qvector = qspiral)
      else:
        # only rotate between supercells
        angleq = qspiral.dot(np.array(vec)) # angle of the rotation
        return global_spin_rotation(m,vector=vector,
              angle=angleq,spiral=True,atoms=None)
    self.intra = tmprot(self.intra,[0.,0.,0.]) # rotate intra matrix
    # now rotate every matrix
    if self.is_multicell: # multicell Hamiltonian
      a1,a2,a3 = self.geometry.a1, self.geometry.a2,self.geometry.a3
      for i in range(len(self.hopping)): # loop
        ar = self.hopping[i].dir # direction
#        direc = a1*ar[0] + a2*ar[1] + a3*ar[2]
        self.hopping[i].m = tmprot(self.hopping[i].m,ar) # rotate matrix
    else:
      if self.dimensionality==0: pass
      elif self.dimensionality==1:
        self.inter = tmprot(self.inter,[1.,0.,0.])
      elif self.dimensionality==2:
        a1,a2 = self.geometry.a1,self.geometry.a2
        self.tx = tmprot(self.tx,[1.,0.,0.])
        self.ty = tmprot(self.ty,[0.,1.,0.])
        self.txy = tmprot(self.txy,[1.,1.,0.])
        self.txmy = tmprot(self.txmy,[1.,-1.,0.])
      else: raise
