import numpy as np
import scipy.linalg as lg
from numba import jit


def matrix_tails(m,discard=None):
    """Return the tails of a 1d Hamiltonian"""
    es,ws = lg.eigh(m)
    ws = np.abs(np.transpose(ws)) # wavefunctions
    if callable(discard): # check if the wave is accepted
      wout = []
      eout = []
      for (e,w) in zip(es,ws):
        if discard(w): 
          wout.append(w) # store wave
          eout.append(e) # store wave
      ws = np.array(wout) # store
      es = np.array(eout) # store
    ls = np.array([loclength([w]) for w in ws]) # localization length
    np.savetxt("TAILS.OUT",np.matrix([es,ls]).T)
    return (es,ls) # return data
    



def tails(vs):
  """Return the log of the tails, centered around the maximum"""
  return tails_python(vs) # python function



def logavgtails(vs):
    """Return the log of the average value of the tails"""
    out = tails(vs) # return all the tails
    ds = np.mean(out,axis=0) # average over waves
    ds = ds[ds>1e-10] # only sizaable elements
    return np.log(ds)



def loclength(vs):
    """Calculate the inverse localization length by fitting the
    decay of the wavefunctions"""
    out = logavgtails(vs) # get the logrho
    ns = np.array(range(len(out))) # length
    ps = np.polyfit(ns,out,1) # make a fit
    return -ps[0] # return the slope


def tails_python(vs):
    """Python implementation, Return the log of the tails, 
    centered around the maximum"""
    vs = np.array(vs)
    out = np.zeros((vs.shape[0],vs.shape[1]//2)) 
    return tails_python_jit(vs,out)



@jit(nopython=True)
def tails_python_jit(vs,out):
    """Python implementation, Return the log of the tails, 
    centered around the maximum"""
    ii = 0 # counter
    for v in vs: # loop over wavefunctions
        d = (v*np.conjugate(v)).real # density
        n = len(d) # size of the array
        nmax = np.argmax(v) # index with the maximum
        ds = np.zeros(n//2) # initialize logd
        for j in range(n//2): # loop over components
            ir = (nmax + j)%n # right index
            il = (nmax - j)%n # left index
            ds[j] = d[ir] + d[il] # average tails
        out[ii,:] = ds[:]
        ii += 1 # increase counter
    return out # return array


