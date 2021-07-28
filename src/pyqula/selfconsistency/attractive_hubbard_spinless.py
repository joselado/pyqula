# specialized routine to perform an SCF, taking as starting point an
# attractive local interaction in a spinless Hamiltonian

from ..sctk.spinless import onsite_delta_vev
from .. import inout
import numpy as np
import time
import os

mf_file = "MF.pkl" 

def attractive_hubbard(h0,mf=None,mix=0.9,g=0.0,nk=8,solver="plain",
        maxerror=1e-5,**kwargs):
    """Perform the SCF mean field"""
    if not h0.check_mode("spinless"): raise # sanity check
    h = h0.copy() # initial Hamiltonian
    if mf is None:
      try: dold = inout.load(mf_file) # load the file
      except: dold = np.random.random(h.intra.shape[0]) # random guess
    else: dold = mf # initial guess
    ii = 0
    os.system("rm -f STOP") # remove stop file
    def f(dold):
      """Function to minimize"""
#      print("Iteration #",ii) # Iteration
      h = h0.copy() # copy Hamiltonian
      if os.path.exists("STOP"): return dold
      h.add_swave(dold*g) # add the pairing to the Hamiltonian
      t0 = time.time()
      d = onsite_delta_vev(h,nk=nk,**kwargs) # compute the pairing
      t1 = time.time()
      print("Time in this iteration = ",t1-t0) # Difference
      diff = np.max(np.abs(d-dold)) # compute the difference
      print("Error = ",diff) # Difference
      print("Average Pairing = ",np.mean(np.abs(d))) # Pairing
      print("Maximum Pairing = ",np.max(np.abs(d))) # Pairing
      print()
#      ii += 1
      return d
    if solver=="plain":
      do_scf = True
      while do_scf:
        d = f(dold) # new vector
        dold = mix*d + (1-mix)*dold # redefine
        diff = np.max(np.abs(d-dold)) # compute the difference
        if diff<maxerror: 
          do_scf = False
    else:
        print("Solver used:",solver)
        import scipy.optimize as optimize 
        if solver=="newton": fsolver = optimize.newton_krylov
        elif solver=="anderson": fsolver = optimize.anderson
        elif solver=="broyden": fsolver = optimize.broyden2
        elif solver=="linear": fsolver = optimize.linearmixing
        else: raise
        def fsol(x): return x - f(x) # function to solve
        dold = fsolver(fsol,dold,f_tol=maxerror)
    h = h0.copy() # copy Hamiltonian
    h.add_swave(dold*g) # add the pairing to the Hamiltonian
    inout.save(dold,mf_file) # save the mean field
    scf = SCF() # create object
    scf.hamiltonian = h # store
    return scf # return SCF object



class SCF(): pass

