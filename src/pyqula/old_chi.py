# library for response calculation

import numpy as np
import numpy.linalg as lg
from .chi_fortran import calculate_xychi
from .collinear_xychi import collinear_xychi

def chi0d(m,energies = [0.],t=0.0001):
  """ Calculate spin response"""
  e,ev = lg.eigh(m) # get eigenvalues
  ev = np.matrix(ev.transpose())
  n = np.shape(ev)
#  import check
#  check.check(ev,e)
  ct = calculate_xychi(ev,e,ev,e,energies,t)
#  return ct


def chi1d(h,energies = [0.],t=0.0001,delta=0.01,q=0.001,nk=1000,U=None,adaptive=True,ks=None):
  hkgen = h.get_hk_gen() # get the generating hamiltonian
  n = len(h.geometry.x) # initialice response
  m = np.zeros((len(energies),n*n),dtype=np.complex) # initialice
  if not adaptive: # full algorithm  
    if ks==None: ks = np.linspace(0.,1.,nk)  # create klist
    for k in ks:
#      print "Doing k=",k
      hk = hkgen(k) # fist point
      e1,ev1 = lg.eigh(hk) # get eigenvalues
      hk = hkgen(k+q) # second point
      e2,ev2 = lg.eigh(hk) # get eigenvalues
      ct = calculate_xychi(ev1.T,e1,ev2.T,e2,energies,t,delta) # contribution
      m += ct
    m = m/nk # normalice by the number of kpoints  
    ms = [m[i,:].reshape(n,n) for i in range(len(m))] # convert to matrices
  if adaptive: # adaptive algorithm 
    from .integration import integrate_matrix
    def get_chik(k): # function which returns a matrix
      """ Get response at a cetain energy"""
      hk = hkgen(k) # first point
      e1,ev1 = lg.eigh(hk) # get eigenvalues
      hk = hkgen(k+q) # second point
      e2,ev2 = lg.eigh(hk) # get eigenvalues
      ct = calculate_xychi(ev1.T,e1,ev2.T,e2,energies,t,delta) # contribution
      return ct # return response
    ms = []
    m = integrate_matrix(get_chik,xlim=[0.,1.],eps=.1,only_imag=False) # add energy
    ms = [m[i,:].reshape(n,n) for i in range(len(m))] # convert to matrices

  if U==None: # raw calculation
    return ms 
  else: # RPA calculation
    return rpachi(ms,U=U)





def chi2d(h,energies = [0.],t=0.0001,delta=0.01,q=np.array([0.001,0.0]),nk=20,U=None,ks=None,collinear=False,adaptive=False):
  hkgen = h.get_hk_gen() # get the generating hamiltonian
  n = len(h.geometry.x) # initialice response
  m = np.zeros((len(energies),n*n),dtype=np.complex) # initialice
  ##############################
  def get_chik(k):
    """Function to integrate"""
    hk = hkgen(k) # fist point
    if collinear: hk = np.matrix([[hk[2*i,2*j] for i in range(len(hk)/2)] for j in range(len(hk)/2)])  # up component
    e1,ev1 = lg.eigh(hk) # get eigenvalues
    hk = hkgen(k+q) # second point
    if collinear: hk = np.matrix([[hk[2*i+1,2*j+1] for i in range(len(hk)/2)] for j in range(len(hk)/2)])  # down component
    e2,ev2 = lg.eigh(hk) # get eigenvalues
    if collinear:
      ct = collinear_xychi(ev1.T,e1,ev2.T,e2,energies,t,delta) # contribution
    else:
      ct = calculate_xychi(ev1.T,e1,ev2.T,e2,energies,t,delta) # contribution
    return ct
  #########################
  if adaptive: # adaptive integration
    from .integration import integrate_matrix
    def get_chik_dim2(kx): # first integration
      def get_chik_dim1(ky): # redefine the function, second integration
        return get_chik(np.array([kx,ky]))
      ct = integrate_matrix(get_chik_dim1,xlim=[0.,1.],eps=delta/100.,only_imag=False) 
      return ct # return second integration
    m = integrate_matrix(get_chik_dim2,xlim=[0.,1.],eps=delta/100.,only_imag=False) # add energy
  else: # normal integration
    ks = np.linspace(0.,1.,nk)  # create klist
    for kx in ks:
      for ky in ks:
        k = np.array([kx,ky])
        ct = get_chik(k)
        m += ct
    m = m/(nk**2) # normalice by the number of kpoints  
  ms = [m[i,:].reshape(n,n) for i in range(len(m))] # convert to matrices
  if U==None: # raw calculation
    return ms 
  else: # RPA calculation
    return rpachi(ms,U=U)






def sumchi(ms):
  """ Sums all the elements of the matrix"""
  n = ms[0].shape[0]
  return np.array([np.sum(mi.reshape(n*n)) for mi in ms])


def rpachi(ms,U=0.0):
  """ Calculate the RPA spin response"""
  iden = np.matrix(np.identity(len(ms[0]),dtype=np.complex)) # identity
  msrpa = [] # list with matrices
  for m in ms:
    m = np.matrix(m) # convert to matrix
    m = (iden - U*m).I*m  # RPA formula
    msrpa.append(m) # store matrix
  return msrpa # return RPA




def collinear_chi1d(h,energies = [0.],t=0.0001,delta=0.01,q=0.001,nk=1000,U=None,adaptive=True,ks=None):
  collinear = True
  hkgen = h.get_hk_gen() # get the generating hamiltonian
  n = len(h.geometry.x) # initialice response
  m = np.zeros((len(energies),n*n),dtype=np.complex) # initialice
  if not adaptive: # full algorithm  
    if ks==None: ks = np.linspace(0.,1.,nk)  # create klist
    for k in ks:
#      print "Doing k=",k
      hk = hkgen(k) # fist point

      if collinear: hk = np.matrix([[hk[2*i,2*j] for i in range(len(hk)/2)] for j in range(len(hk)/2)])  # up component

      e1,ev1 = lg.eigh(hk) # get eigenvalues
      hk = hkgen(k+q) # second point

      if collinear: hk = np.matrix([[hk[2*i+1,2*j+1] for i in range(len(hk)/2)] for j in range(len(hk)/2)])  # down component

      e2,ev2 = lg.eigh(hk) # get eigenvalues
      ct = collinear_xychi(ev1.T,e1,ev2.T,e2,energies,t,delta) # contribution
      m += ct
    m = m/nk # normalice by the number of kpoints  
    ms = [m[i,:].reshape(n,n) for i in range(len(m))] # convert to matrices
  if adaptive: # adaptive algorithm 
    from .integration import integrate_matrix
    def get_chik(k): # function which returns a matrix
      """ Get response at a cetain energy"""
      hk = hkgen(k) # first point
      if collinear: hk = np.matrix([[hk[2*i,2*j] for i in range(len(hk)/2)] for j in range(len(hk)/2)])  # up component
      e1,ev1 = lg.eigh(hk) # get eigenvalues
      hk = hkgen(k+q) # second point
      if collinear: hk = np.matrix([[hk[2*i+1,2*j+1] for i in range(len(hk)/2)] for j in range(len(hk)/2)])  # down component
      e2,ev2 = lg.eigh(hk) # get eigenvalues
      ct = collinear_xychi(ev1.T,e1,ev2.T,e2,energies,t,delta) # contribution
      return ct # return response
    ms = []
    m = integrate_matrix(get_chik,xlim=[0.,1.],eps=.01,only_imag=False) # add energy
    ms = [m[i,:].reshape(n,n) for i in range(len(m))] # convert to matrices

  if U==None: # raw calculation
    return ms 
  else: # RPA calculation
    return rpachi(ms,U=U)













if __name__=="__main__":
  from . import geometry
  from . import hamiltonians
  g = geometry.chain()
  h = g.get_hamiltonian()
#  h.add_antiferromagnetism(.1)
  h.add_antiferromagnetism(.1)
  energies = np.linspace(-1.,1.,100)
  ms = chi1d(h,energies=energies)
  ms = sumchi(ms) # sum all the elements
  import pylab as py
#  b = h.plot_bands()
  py.plot(energies,ms.imag)
  py.plot(energies,ms.real)
  py.show()
