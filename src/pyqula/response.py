import numpy as np
from . import timing
from . import parallel



def magnetic_response_map(h,nk=20,nq=20,j=[0.1,0.,0.],r=[0,0,1],
          kp=None,qs=None):
  """Generate a magnetic susceptibility map"""
  h0 = h.copy() # copy Hamiltonian
# function with the energy
  energy_qvector = energy_qvector_generator(h,j=j,r=r,kp=kp,nk=nk) 
  f = open("SUSCEPTIBILITY.OUT","w")
  k2K = h.geometry.get_k2K_generator() # get the function
  # loop over qvectors
  if qs is None: # not provided
    xs = np.linspace(-2.0,2.0,nq) # kx
    ys = np.linspace(-2.0,2.0,nq) # ky
    qs = [] # initialize qvectors
    for x in xs:
      for y in ys:
        qs.append(np.array([x,y,0.])) # store
  else: pass # qs provided on input
  est = timing.Testimator(maxite=len(qs)) # initialize estimator
  def fun(q0):
    q = k2K(q0) # put in reciprocal coordinates
    return energy_qvector(q=q)
  es = parallel.pcall(fun,qs)
  for (e,q0) in zip(es,qs): # loop
#    est.iterate()
#    e = energy_qvector(q=q)  # energy
    f.write(str(q0[0])+"  "+str(q0[1])+"  "+str(e)+"\n")
    f.flush()
  f.close()
  print("writen SUSCEPTIBILITY.OUT")




def energy_qvector_generator(h,j=[1.0,0.0,0.0],r=[0,0,1],kp=None,nk=10):
  """Return a function that calculates the total energy
  in a specific qvector"""
  h0 = h.copy() # copy Hamiltonian
  def energy_qvector(random=False,q=[0.,0.,0.]):
    h = h0.copy()
    h.generate_spin_spiral(vector=r,qspiral=2.*q)
    h.add_zeeman(j)
    return h.total_energy(nk=nk,kp=kp)
  return energy_qvector


