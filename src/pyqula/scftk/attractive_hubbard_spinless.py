# specialized routine to perform an SCF, taking as starting point an
# attractive local interaction in a spinless Hamiltonian

from ..sctk.spinless import onsite_delta_vev
from .. import inout
import numpy as np
import time
import os
from .. import filesystem as fs

mf_file = "MF.pkl" 

def attractive_hubbard(h0,mf=None,mix=0.9,g=0.0,nk=8,solver="plain",
        maxerror=1e-5,maxite=None,**kwargs):
    """Perform the SCF mean field"""
    if not h0.check_mode("spinless"): # sanity check
      raise ValueError("the spinless attractive Hubbard mean field needs a "
              "spinless Hamiltonian")
    h = h0.copy() # initial Hamiltonian
    if mf is None:
      try:
          dold = np.array(inout.load(mf_file)) # load the file
          # the cached file may well come from a different system, the same
          # check densitydensity.mf_matches_hamiltonian makes for its own
          # MF.pkl
          if dold.shape!=(h.intra.shape[0],):
              raise ValueError("cached MF.pkl shape does not match this Hamiltonian")
      except: dold = np.random.random(h.intra.shape[0]) # random guess
    else: dold = mf # initial guess
    ii = 0
    fs.rmfile("STOP") # remove stop file
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
    converged = True # assume convergence unless maxite says otherwise
    if solver=="plain":
      do_scf = True
      ite = 0 # start counter
      while do_scf:
        d = f(dold) # new vector
        # the residual is taken BEFORE mixing, as the three sibling SCF
        # loops in the package do: against the already-mixed vector it is
        # (1-mix) times the true one, so the effective tolerance was
        # maxerror/(1-mix), and at mix=1.0 it was identically zero and the
        # loop stopped after one iteration on the initial guess
        diff = np.max(np.abs(d-dold)) # compute the difference
        dold = mix*d + (1-mix)*dold # redefine
        if diff<maxerror: 
          do_scf = False
        elif maxite is not None and ite>=maxite: # too many iterations
          print("No convergence has been reached in",maxite,"iterations, stopping")
          converged = False # no convergence
          do_scf = False
        ite += 1 # increase number of iterations
    else:
        print("Solver used:",solver)
        import scipy.optimize as optimize 
        if solver=="newton": fsolver = optimize.newton_krylov
        elif solver=="anderson": fsolver = optimize.anderson
        elif solver=="broyden": fsolver = optimize.broyden2
        elif solver=="linear": fsolver = optimize.linearmixing
        else:
          raise ValueError("unknown solver; the accepted ones are 'plain', "
                  "'newton', 'anderson', 'broyden' and 'linear'")
        def fsol(x): return x - f(x) # function to solve
        dold = fsolver(fsol,dold,f_tol=maxerror)
    h = h0.copy() # copy Hamiltonian
    h.add_swave(dold*g) # add the pairing to the Hamiltonian
    inout.save(dold,mf_file) # save the mean field
    scf = SCF() # create object
    scf.hamiltonian = h # store
    scf.converged = converged # store whether the loop reached the tolerance
    return scf # return SCF object



class SCF(): pass

