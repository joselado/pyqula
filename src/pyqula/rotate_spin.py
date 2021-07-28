import numpy as np
import scipy.linalg as lg



from scipy.sparse import csc_matrix,bmat
sx = csc_matrix([[0.,1.],[1.,0.]])
sy = csc_matrix([[0.,-1j],[1j,0.]])
sz = csc_matrix([[1.,0.],[0.,-1.]])



def rotation_matrix(m,vectors):
  """ Rotates a matrix, to align its components with the direction
  of the magnetism """
  if not len(m)==2*len(vectors): # stop if they don't have
                                  # compatible dimensions
     raise
  # pauli matrices
  n = len(m)/2 # number of sites
  R = [[None for i in range(n)] for j in range(n)] # rotation matrix
  from scipy.linalg import expm  # exponenciate matrix
  for (i,v) in zip(range(n),vectors): # loop over sites
    vv = np.sqrt(v.dot(v)) # norm of v
    if vv>0.000001: # if nonzero scale
      u = v/vv
    else: # if zero put to zero
      u = np.array([0.,0.,0.,])
#    rot = u[0]*sx + u[1]*sy + u[2]*sz 
    uxy = np.sqrt(u[0]**2 + u[1]**2) # component in xy plane
    phi = np.arctan2(u[1],u[0])
    theta = np.arctan2(uxy,u[2])
    r1 =  phi*sz/2.0 # rotate along z
    r2 =  theta*sy/2.0 # rotate along y
    # a factor 2 is taken out due to 1/2 of S
    rot = expm(1j*r2) * expm(1j*r1)   
    R[i][i] = rot  # save term
  R = bmat(R)  # convert to full sparse matrix
  return R.todense()



def align_magnetism(m,vectors):
  """ Align matrix with the magnetic moments"""
  R = rotation_matrix(m,vectors) # return the rotation matrix
  mout = R * csc_matrix(m) * R.H  # rotate matrix
  return mout.todense() # return dense matrix





def global_spin_rotation(m,vector = np.array([0.,0.,1.]),angle = 0.0,
                             spiral = False,atoms = None):
  """ Rotates a matrix along a certain qvector """
  # pauli matrices
  from scipy.sparse import csc_matrix,bmat
  iden = csc_matrix([[1.,0.],[0.,1.]])
  n = m.shape[0]//2 # number of sites
  if atoms==None: 
    atoms = range(n) # all the atoms
  else: 
    raise
  R = [[None for i in range(n)] for j in range(n)] # rotation matrix
  for i in range(n): # loop over sites
    u = np.array(vector) # rotation direction
    u = u/np.sqrt(u.dot(u)) # normalize rotation direction
    rot = (u[0]*sx + u[1]*sy + u[2]*sz)/2. # rotation
    # a factor 2 is taken out due to 1/2 of S
    # a factor 2 is added to have BZ in the interval 0,1
    rot = lg.expm(2.*np.pi*1j*rot*angle/2.0)
#    if i in atoms:
    R[i][i] = rot  # save term
#    else:
#      R[i][i] = iden  # save term
  R = bmat(R)  # convert to full sparse matrix
  if spiral:  # for spin spiral
    mout = R @ m  # rotate matrix
  else:  # normal global rotation
    mout = R @ m @ R.H  # rotate matrix
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
         R[i][i] = lg.expm(2.*np.pi*1j*rot*angle/2.0)
      return bmat(R)  # convert to full sparse matrix
  Roti = getR(ri) # get the first rotation matrix
  Rotj = getR(rj) # get the second rotation matrix
#  print(Roti@Rotj.H)
#  print(ri,rj)
  return Rotj @ m @ Roti.H # return the rotated matrix


def hamiltonian_spin_rotation(self,vector=np.array([0.,0.,1.]),angle=0.):
    """ Perform a global spin rotation """
    if not self.has_spin: raise # no spin in the Hamiltonian
    gsr = global_spin_rotation # rename method
    if self.has_eh: raise
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
                            qspiral=[1.,0.,0.],fractional=True):
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
