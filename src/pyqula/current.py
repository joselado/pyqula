import numpy as np
import scipy.sparse.linalg as lg
import scipy.linalg as lg

def braket_wAw(w,A):
  w = np.matrix(w) # convert to matrix
  return ((w.T).H*A*w.T)[0,0] # expectation value

def current_operator(h):
  """Get the current operator"""
  h = h.get_multicell()
  def fj(k0):
      return derivative(h,k0,order=[1])
  return fj



def gs_current(h,nk=400):
  weighted_current(h,nk=nk)




def fermi_current(h,nk=400,delta=0.5):
  def fun(e):
    return delta/(delta**2+e**2)*2/np.pi
  weighted_current(h,nk=nk,fun=fun)



def weighted_current(h,nk=400,fun=None):
  """Calculate the Ground state current"""
  if fun is None:
    delta = 0.01
    def fun(e): return (-np.tanh(e/delta) + 1.0)/2.0
  jgs = np.zeros(h.intra.shape[0]) # current array
  hkgen = h.get_hk_gen() # generator
  fj = current_operator(h) # current operator
  ks = np.linspace(0.0,1.0,nk,endpoint=False) # k-points
  for k in ks: # loop
    hk = hkgen(k) # Hamiltonian
    (es,ws) = lg.eigh(hk) # diagonalize
    ws = ws.transpose() # transpose
    jk = fj(k) # get the generator
    for (e,w) in zip(es,ws): # loop
      weight = fun(e) # weight
      print(weight)
      d = np.conjugate(w)*ket_Aw(jk,w) # current density
      jgs += d.real*weight # add contribution
#      jgs += (np.abs(w)**2*weight).real # add contribution
  jgs /= nk # normalize
  print("Total current",np.sum(jgs))
  np.savetxt("CURRENT1D.OUT",np.matrix([range(len(jgs)),jgs]).T)




def derivative(h,k,order=None):
  """Calculate the derivative of the Hamiltonian"""
  ## The order parameter is kind of weird now, this must be fixed ##
  h = h.get_multicell() # get multicell Hamiltonian
  if order is None:
#    order = [1 for i in range(h.dimensionality)] # order of the derivative
    order = [1,0,0] # default
  if h.dimensionality == 0: return None
  elif h.dimensionality == 1: # one dimensional
      mout = h.intra*0.0 # initialize
      for t in h.hopping: # loop over matrices
        phi = np.array(t.dir).dot(k) # phase
        pref = (t.dir[0]*1j)**order[0] # prefactor
        tk = pref*t.m * np.exp(1j*np.pi*2.*phi) # k hopping
        mout = mout + tk # add contribution
      return mout
  elif h.dimensionality == 2: # two dimensional
      k = np.array(k) # convert to array
      mout = h.intra*0.0 # initialize
      for t in h.hopping: # loop over matrices
        d = t.dir
        d = np.array(d) # vector director of hopping
        phi = d.dot(k) # phase
        pref1 = (d[0]*1j)**order[0] # prefactor
        pref2 = (d[1]*1j)**order[1] # prefactor
        pref = pref1*pref2 # total prefactor
        tk = pref*t.m * np.exp(1j*np.pi*2.*phi) # derivative of the first
        mout = mout + tk # add to the hamiltonian
      return mout
  else: raise # not implemented



