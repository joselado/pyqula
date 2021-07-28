import numpy as np
from . import classicalspinf90
from . import neighbor
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix

zero = np.matrix(np.zeros((3,3)))  # real matrix
iden = np.matrix(np.identity(3))  # real matrix
zzm = zero.copy() ; zzm[2,2] = 1.0 # matrix for ZZ interaction


class SpinModel(): # class for a spin Hamiltonian
  def __init__(self,g): # geometry
    self.geometry = g # store geometry
    self.nspin = len(g.r) # number of sites
    self.theta = np.zeros(len(g.r)) # theta angles
    self.phi = np.zeros(len(g.r)) # phi angles
    self.b = np.zeros((len(g.r),3)) # magnetic field
    self.j = np.array([zero]) # empty list
    self.pairs = np.array([[0,0]]) # empty list
  def add_heisenberg(self,fun=None):
    h = self.geometry.get_hamiltonian(has_spin=False,fun=fun)
    m = coo_matrix(h.get_hk_gen()([0.,0.,0.])) # get onsite matrix
    (pairs,js) = add_heisenberg(self.geometry.r)
    pairs = np.array([m.row,m.col]).transpose() # convert to array
    self.pairs = pairs
    self.j = [ j.real*np.identity(3) for j in m.data] # store
  def add_field(self,v):
    """Add magnetic field"""
    self.b += np.array([v for i in range(self.nspin)])
  def energy(self,use_fortran=True):
    """ Calculate the energy"""
    if use_fortran:
      eout = classicalspinf90.energy(self.theta,self.phi,self.b,self.j,
               self.pairs)
    return eout
  def minimize_energy(self,theta0=None,phi0=None,tries=10,calle=None):
    """Minimize the energy of the spin model"""
    thetas = [None for i in range(tries)]
    phis = [None for i in range(tries)]
    es = [None for i in range(tries)]
    for i in range(tries): # loop over tries
      theta,phi = minimize_energy(self,theta0=theta0,phi0=phi0,calle=calle)
      self.theta = theta.copy()
      self.phi = phi.copy()
      e = self.energy() # calculate energy
      thetas[i] = theta.copy()
      phis[i] = phi.copy()
      es[i] = e
    imin = es.index(min(es)) # minimum
    print("Minimum energy",es[imin])
    print("Minimum energy per spin",es[imin]/len(self.phi))
    self.theta = thetas[imin].copy()
    self.phi = phis[imin].copy()
    self.update_magnetization()
    return self.theta,self.phi
  def write(self,label=""):
    """Write in file"""
    self.geometry.write() # write the geometry
    write_magnetization(self) # write magnetization
  def add_tensor(self,fun):
    """Add a tensor interaction"""
    pairs,js = add_tensor(self,fun)
    self.pairs = np.concatenate([self.pairs,pairs])
    self.j = np.concatenate([self.j,js])
  def add_tensor_2d(self,fun,ncells=1,vspiral=[0.,0.]):
    pairs,js = add_tensor_2d(self,fun,ncells=ncells,vspiral=vspiral)
    self.pairs = np.concatenate([self.pairs,pairs])
    self.j = np.concatenate([self.j,js])
  def get_jacobian(self):
    """Return a function that calculated the Jacobian"""
#    return None
    return get_jacobian(self.b,self.j,self.pairs)
  def load_magnetism(self,name="MAGNETIZATION.OUT"):
    """Read the magnetization from a file"""
    m = np.genfromtxt(name).transpose()
    r = np.sqrt(m[1]**2+m[2]**2) # in-plane radius
    self.theta = np.arctan2(r,m[3]) # theta angle
    self.phi = np.arctan2(m[2],m[1]) # theta angle
  def regroup(self):
    """Regroups the terms in the Hamiltonian"""
    print(len(self.pairs))
    self.pairs,self.j = regroup(self.pairs,self.j) 
    print(len(self.pairs))
  def update_magnetization(self):
    """Update the magnetization"""
    mx = np.sin(self.theta)*np.cos(self.phi)
    my = np.sin(self.theta)*np.sin(self.phi)
    mz = np.cos(self.theta)
    self.magnetization = np.array([mx,my,mz]).transpose()
  def setup_matrix(self):
    """Creates the matrix that allows to calculate the energy""" 




def add_heisenberg(r):
  """Return pairs and js for a Heisenberg interaction"""
  pairs = neighbor.find_first_neighbor(r,r)
  js = np.array([iden for p in pairs]) # array
  return (pairs,js) # return pairs 


def energy(thetas,phis,bs,js,indsjs):
  """Calculate the energy"""
  eout = classicalspinf90.energy(thetas,phis,bs,js,indsjs)
  return eout



def get_jacobian(bs,js,indsjs):
  """Return the Jacobian"""
  def jacobian(thetaphi):
    thetas = thetaphi[0:len(thetaphi)//2]
    phis = thetaphi[len(thetaphi)//2:len(thetaphi)]
    jac = classicalspinf90.jacobian(thetas,phis,bs,js,indsjs)
    return jac
  return jacobian


def minimize_energy(sm,theta0=None,phi0=None,tol=1e-5,calle=None):
  """Return the minimum energy and gound state theta/phi"""
  if theta0 is None: theta0 = np.random.random(sm.nspin)*np.pi
  if phi0 is None: phi0 = np.random.random(sm.nspin)*np.pi*2
  from scipy.optimize import minimize
  thetaphi0 = np.concatenate([theta0,phi0]) # initial guess
  if calle is None: # no added function to the total energy
    def fun(thetaphi): # function to optimize
      sm.theta = thetaphi[0:sm.nspin]
      sm.phi = thetaphi[sm.nspin:sm.nspin*2]
      return sm.energy() # return energy
    result = minimize(fun,thetaphi0,tol=tol,jac=sm.get_jacobian())
  else: # a function is provided that corrects the energy
    def fun2(thetaphi): # function yielding the energy
      sm.theta = thetaphi[0:sm.nspin]
      sm.phi = thetaphi[sm.nspin:sm.nspin*2]
      sm.update_magnetization() # update the magnetization
      return sm.energy() + calle(sm) # return total energy
    result = minimize(fun2,thetaphi0,tol=tol,method="Powell") # minimize this function
  theta = result.x[0:sm.nspin]
  phi = result.x[sm.nspin:sm.nspin*2]
  return theta,phi

def write_magnetization(self):
  theta = self.theta
  phi = self.phi
  mx = np.sin(theta)*np.cos(phi)
  my = np.sin(theta)*np.sin(phi)
  mz = np.cos(theta)
  fo = open("MAGNETISM.OUT","w")
  for i in range(len(mx)):
    fo.write(str(self.geometry.x[i])+"   ")
    fo.write(str(self.geometry.y[i])+"   ")
    fo.write(str(self.geometry.z[i])+"   ")
    fo.write(str(mx[i])+"   ")
    fo.write(str(my[i])+"   ")
    fo.write(str(mz[i])+"\n")
  fo.close()



def add_tensor(sm,fun):
  """Add a tensor interaction"""
  pairs = []
  js = []
  for i1 in range(sm.nspin): # loop over positions
    r1 = sm.geometry.r[i1]
    for i2 in range(sm.nspin): # loop over positions
      r2 = sm.geometry.r[i2]
      m = fun(r1,r2) # call the function 
      if np.max(np.abs(m))>0.00001: # if non zero
        pairs.append((i1,i2)) # store 
        js.append(m) # store matrix
  return np.array(pairs),np.array(js)


def rotatez(angle):
  """In-plane rotation"""
  m = np.matrix(np.zeros((3,3),dtype=np.complex)) # zero matrix
  cp = np.cos(angle*np.pi)
  sp = np.sin(angle*np.pi)
  m[0,0] = cp
  m[1,1] = cp
  m[0,1] = sp
  m[1,0] = -sp
  m[2,2] = 1.0
  return m # return matrix





def add_tensor_2d(sm,fun,ncells=4,vspiral=[0.,0.]):
  """Add a tensor interaction, assuming 2d system"""
  pairs = []
  js = []
  for ia in range(-ncells,ncells+1): # loop over cells
    for ja in range(-ncells,ncells+1): # loop over cells
      for i1 in range(sm.nspin): # loop over positions
        r1 = sm.geometry.r[i1]
        for i2 in range(sm.nspin): # loop over positions
          r2 = sm.geometry.r[i2] + ia*sm.geometry.a1 + ja*sm.geometry.a2
          m = fun(r1,r2) # call the function 
          if np.max(np.abs(m))>0.00001: # if non zero
            angle = vspiral[0]*ia + vspiral[1]*ja
            R = rotatez(angle)
            pairs.append((i1,i2)) # store 
            js.append(m*R) # store matrix
  return np.array(pairs),np.array(js)








def generating_functions(name="Heisenberg",J=1.0,v=np.array([0.,0.,1.]),
       fdiff = lambda x,y: x-y,fc=None,fr=None):
  """Return fucntion that return matrices for spin itneractions"""
  if fc is None: # if no function provided
    def fc(d):
      if 0.9<d<1.1: return 1.0
      else: return 0.0
  if fr is None: # if no function provided
    def fr(r1,r2): # vector dependent rotation
      return iden # identity matrix
  if name=="Heisenberg":
    def fun(r1,r2):
      dr = fdiff(r2,r1)
      dr = np.sqrt(dr.dot(dr))
      return iden*J*fc(dr)*fr(r1,r2) # distance dependent coupling
    return fun
  elif name=="Linear":
    def fun(r1,r2):
      r12 = fdiff(r2,r1) # vector between them
      dr = np.sqrt(r12.dot(r12)) # distance
      if dr<0.1: return np.zeros((3,3))
      ur = r12/dr # unit vector
      m = np.matrix([[ur[i]*ur[j] for i in range(3)] for j in range(3)])
      return m*J*fr(r1,r2)/dr**3 # distance dependent coupling
    return fun
  elif name=="RKKYTI":
    """RKKY interaction in the surface of a TI, as derived in
    PRB 81 233405 (2010) """
    def fun(r1,r2):
      r12 = fdiff(r2,r1) # vector between them
      dr = np.sqrt(r12.dot(r12)) # distance
      if dr<0.1: return np.zeros((3,3)) # return 0
      ur = r12/dr # unit vector
      m = np.matrix([[ur[i]*ur[j] for i in range(3)] for j in range(3)])
      m = m*3/2 - np.identity(3) 
      return m*J*fr(r1,r2)/dr**3 # distance dependent coupling
    return fun
  elif name=="ZZ":
    def fun(r1,r2):
      dr = fdiff(r2,r1)
      dr = np.sqrt(dr.dot(dr))
      return zzm*J*fc(dr)*fr(r1,r2) # return identity
    return fun
  elif name=="XYZ":
    def fun(r1,r2):
      dr = fdiff(r2,r1)
      dr2 = np.sqrt(dr.dot(dr))
      if np.abs(fc(dr2))<0.00000001: return zero
      if callable(v): return J*fc(dr2)*np.diag(v(dr)) # return matrix
      else: return J*fc(dr)*np.diag(v)*fr(r1,r2) # return matrix
    return fun
  elif name=="DM":
    eps = get_lc()
    def fun(r1,r2):
      dr1 = fdiff(r2,r1)
      dr = np.sqrt(dr1.dot(dr1))
      if 0.9<dr<1.1: 
        m = np.zeros((3,3)) # intialize the matrix
        if callable(v): rm = np.cross(dr1,v(dr1)) # intermediate ion
        else: rm = np.cross(dr1,v) # intermediate ion
        rm = rm/np.sqrt(rm.dot(rm)) # unitary vector
        for i in range(3):
          for j in range(3): 
            for k in range(3): 
              m[i,j] += eps[i,j,k]*rm[k]
        return m*J*fr(r1,r2) # return identity
      else: return zero
    return fun



def generating_profiles(r,name="skyrmion",n=1.,cut=1.0):
  """Return thetas and phis"""
  rho = np.sqrt(r[:,0]**2 + r[:,1]**2) # in-plnae r
  if name=="skyrmion":
    theta = cut*rho/np.max(rho)*np.pi
    phi = n*np.arctan2(r[:,1],r[:,0])
    return theta,phi
  elif name=="spiral":
    phi = (r[:,0]/np.max(r[:,0])+1)*n*np.pi*2
    return phi*0.+np.pi/2,phi
  else: raise


def get_lc():
  """Return a Levi Civita"""
  e = np.zeros((3,3,3)) # start as zero
  e[0,1,2] = 1.0 
  e[0,2,1] = -1.0 
  e[1,0,2] = -1.0 
  e[1,2,0] = 1.0 
  e[2,0,1] = 1.0 
  e[2,1,0] = -1.0 
  return e





def regroup(ps,js):
  """Regroups the terms in the Hamiltonian"""
  outp = [] # output pairs
  outj = [] # output js
  dictj = dict() # create dictionary
  for (p,j) in zip(ps,js): # loop over inputs
    ip = (p[0],p[1])
    jp = (p[1],p[0])
    if ip in outp: # if the pair is present
      dictj[ip] += j # add contribution
    elif jp in outp: # if the alternative pair is present
      dictj[jp] += j.transpose() # add contribution
    else: # not present
      outp.append(ip) # store
      dictj[ip] = j # store this j
  outj = [dictj[p] for p in outp] # get all
  return np.array(outp),np.array(outj)


