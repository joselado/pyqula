from __future__ import print_function
import numpy as np
from scipy.sparse import csc_matrix,bmat
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(method.__name__, te-ts)
        return result
    return timed


# routines to calculate DC supercurrent using keldysh formalism


def floquet_selfenergy(selfgen,e,omega,n=20,less=False,delta=0.01):
  """Generate a floquet selfenergy starting from
  a function that returns selfenergies"""
  out = [[None for i in range(2*n+1)] for i in range(2*n+1)] # output
  for i in range(-n,n+1): # loop
    ebar = e+i*omega # floquet energy
    term = selfgen(ebar) # call and store
    if less: # special propagator with the < symbol
#      bfac = 1./(np.exp(ebar/delta)+1) # boltzman factor
#      term *= bfac
      if ebar<0.: term *= 0.
      else: term = -(term-term.H)
    out[i+n][i+n] = term # store
  return bmat(out) # return dense matrix


def floquet_hamiltonian(ham,trl,omega,n=20):
  """Calculate the floquet Hamiltonian"""
  out = [[None for i in range(2*n+1)] for i in range(2*n+1)] # output
  iden = np.identity(ham.shape[0],dtype=np.complex) # identity matrix
  for i in range(-n,n+1): # loop
    out[i+n][i+n] = ham - iden*i*omega # diagonal
  for i in range(-n,n): # loop
    out[i+n][i+1+n] = trl # diagonal
    out[i+1+n][i+n] = trl.H # diagonal
#  for o in out[0]: print(o.shape)
  return bmat(out) # return dense matrix
  


def floquet_tauz(m,n=20):
  """Calculate the floquet Hamiltonian"""
  # hamgen is a function that return a time dependent hamiltonian
  out = [[None for i in range(2*n+1)] for i in range(2*n+1)] # output
  for i in range(-n,n+1): # loop
    out[i+n][i+n] = m # store
  return bmat(out) # return dense matrix
  










def current(data,voltage,n=3):
  """Apply the formula from San Jose NJP"""
  omega = voltage # frequency
  self_l = data.self_l  # self energy, callable
  self_r = data.self_r  # self energy, callable
  ham = data.ham  # Hamiltonian, matrix
  delta = data.delta  # Hamiltonian, matrix
  tfreq = data.tfreq # Hamiltonian, matrix
  tauz = data.tauz  # Hamiltonian, matrix
  def intfun(e,n=n): # function to integrate
    ftauz = floquet_tauz(tauz,n=n)
    fh = floquet_hamiltonian(ham,tfreq,omega,n=n) # floquet hamiltonian
# identity matrix
    iden = np.matrix(np.identity(fh.shape[0],dtype=np.complex)) 
    sr = floquet_selfenergy(self_r,e,omega,n=n,less=False) 
    sl = floquet_selfenergy(self_l,e,omega,n=n,less=False) 
    sR_less = floquet_selfenergy(self_r,e,omega,n=n,less=True) 
    sL_less = floquet_selfenergy(self_l,e,omega,n=n,less=True) 
    gr = (iden*(e+1j*delta) - fh - sr - sl).I # retarded green function
    gless = gr*(sL_less+sR_less)*gr.H # less green function
    val = (gr*sL_less + gless*sl.H)*ftauz
    curr = val.trace()[0,0] # add contribution
    return curr.real # return real part
  nglobal = n # set the global n 
  # function to decide the n to use afterwards
  def deciden(e,n=n,r0=-1000):
    r1 = intfun(e,n) # call the function
    dr = np.abs(r0-r1)/(np.abs(r0)+np.abs(r1)) # difference
    if dr<0.01: 
      return n
    else: return deciden(e,n=n+1,r0=r1) # recall
  nnew = deciden(voltage/2.,n=n) # update n
  def convfun(e,n=nnew,r0=-1000):
    """Converged function"""
    r1 = intfun(e,n) # call the function
    dr = np.abs(r0-r1)/(np.abs(r0)+np.abs(r1)) # difference
    if dr<0.01: 
      print("Converged with ",n,r1)
      return r1
    else: return convfun(e,n=n+1,r0=r1) # recall
  print(convfun(0.2,n=4))
#  exit()
  from scipy.integrate import quad
  xs = np.linspace(0.,omega,60) # interval
  ys = [convfun(x) for x in xs] # integrand
#  from parallel import pcall
#  ys = pcall(intfun,xs,cores=5) # return ys
  np.savetxt("TEST_"+str(voltage)+".OUT",np.matrix([xs,ys]).T)
  return np.trapz(ys,x=xs)
#  return np.sum(ys)/20
  curr,err = quad(convfun,0.0,omega,limit=10,epsrel=0.01)
  print(curr,err)
  return curr.real



def calculate_current(ht,v=0.01,delta=0.01):
  """Calculate the current using aheterostructure"""
  if ht.block_diagonal: raise # not implemented
  ###########################################################
  # time independent part of central Hamiltonian
  ham = [[None for i in range(3)] for j in range(3)]  
  ham[0][0] = ht.left_intra
  ham[1][1] = ht.central_intra
  ham[2][2] = ht.right_intra
  ham = bmat(ham).todense() # dense matrix
  # get the Nambu tauz
  from .hamiltonians import get_nambu_tauz
  from .hamiltonians import project_electrons as eproj
  from .hamiltonians import project_holes as hproj
  tauz = get_nambu_tauz(ham,has_eh=True) # Nambu tauz
  # now calculate the time dependent coupling (positive frequency)
  tfreq = [[None for i in range(3)] for j in range(3)]  
  tfreq[0][0] = ht.left_intra*0.0
  tfreq[1][1] = ht.central_intra*0.0
  tfreq[2][2] = ht.right_intra*0.0
  tfreq[0][1] = eproj(ht.left_coupling.H) # project on electrons
  tfreq[1][0] = hproj(ht.left_coupling) # project on holes
  tfreq[1][2] = eproj(ht.right_coupling) # project on electrons
  tfreq[2][1] = hproj(ht.right_coupling.H) # project on holes
  tfreq = bmat(tfreq)
  class Data(): pass
  data =  Data() # create data
  data.tauz = tauz
  data.ham = ham
  data.tfreq = tfreq
  def m2block(m,i=0):
    out = [[None for i in range(3)] for j in range(3)]  
    out[0][0] = np.zeros(ht.left_intra.shape)
    out[1][1] = np.zeros(ht.central_intra.shape)
    out[2][2] = np.zeros(ht.right_intra.shape)
    out[i][i] = m
    return bmat(out)
  lgen = lambda e: ht.get_selfenergy(e,lead=0,pristine=True)
  rgen = lambda e: ht.get_selfenergy(e,lead=1,pristine=True)
  data.self_l = lambda e: m2block(lgen(e),i=0)
  data.self_r = lambda e: m2block(rgen(e),i=2)
  data.delta = delta
  return current(data,v) # return current



