# routines to update the mean field
from .. import meanfield
from .. import check
import numpy as np
from numba import jit

def sum_contributions(a,b,vav,vbv,g,inds):
    """Compute all the contributions to the mean field"""
    dirs = [] # empty list


def generate_index2vec(self):
    """Generate the index2vec variables"""
    if self.has_index2vec: return # already generated
    ds = [v.dir for v in self.interactions] # directions
    uniqued = []
    for d in ds:
        if not tuple(d) in uniqued: uniqued.append(tuple(d))
    inds = []
    for v in self.interactions: # loop
        i = uniqued.index(tuple(v.dir)) # get the index
        inds.append(i) # store
    self.interactions_dindex = np.array(inds) # store
    self.interactions_vecs = uniqued # store
    self.has_index2vec = True # generated


def update_mean_field(self,mixing=0.95):
    """Calculate the expectation values of the different operators"""
    generate_index2vec(self) # generate the array of not present yet
    accu = 0.0 # accumulator
    if self.correlator_mode == "multicorrelator":
      np.savetxt("VS_SCF.OUT",np.matrix([range(len(self.cij)),np.abs(self.cij)]).T)
    zero = self.hamiltonian.intra*0.
    # initialize
    storage = [zero.copy() for i in range(len(self.interactions_vecs))]
    for ii in range(len(self.interactions)): # loop over interactions
      v = self.interactions[ii] # get this one
      if v.contribution=="AB":
        tmp = v.a*v.vbv*v.g + v.b*v.vav*v.g # add contribution
        accu += np.abs(v.vbv) + np.abs(v.vav)
      elif v.contribution=="A":
        tmp = v.a*v.vbv*v.g
        accu += np.abs(v.vbv)
      else: raise
      # store in the dictionary
      jj = self.interactions_dindex[ii] # index for this one
      storage[jj] = storage[jj] + tmp # store
    ##################################
    ##################################
    # now put it in a dictionary
    mfnew = dict() # new mean field
    jj = 0
    for d in self.interactions_vecs: # get this one
      mfnew[tuple(d)] = storage[jj] # add contribution
      jj += 1
    accu /= len(self.interactions)*2
    if not self.silent: print("Average value of expectation values",accu)
    self.error = error_meanfield(self.mf,mfnew)/self.sites # get the error
    self.mf = mix_meanfield(self.mf,mfnew,mixing=mixing) # new mean field
    if self.hamiltonian.has_eh:
      print("################################")
      print("Enforcing electron-hole symmetry")
      print("################################")
      self.mf = meanfield.enforce_eh(self.hamiltonian,self.mf)
    if not self.silent: print("ERROR",self.error)
    check.check_dict(self.mf)




def error_meanfield(mf1,mf2):
  """Return the difference between two mean fields"""
  error = 0.0 # initialize
  for key in mf1:
    error += np.sum(np.abs(mf1[key]-mf2[key])) # sum error
  return error



def mix_meanfield(mf1,mf2,mixing=0.9):
  """Mix the two mean fields"""
  out = dict() # create dictionary
  for key in mf2:
    if key in mf1: # if present in the old one
      out[key] = mf1[key]*(1-mixing)+mf2[key]*mixing # sum error
    else:
      out[key] = mf2[key] # sum error
  return out


