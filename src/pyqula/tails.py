import numpy as np
import scipy.linalg as lg


# check whether if the fortran routines will be used
try:
  from . import tailsf90
  use_fortran = True
except:
  use_fortran=False
#  print("WARNING, Fortran functions not working in tails")



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
    



def tails(vs,use_fortran=use_fortran):
  """Return the log of the tails, centered around the maximum"""
  if use_fortran: return tailsf90.tails(vs) # fortran function
  else: return tails_python(vs) # python function



def logavgtails(vs,use_fortran=use_fortran):
  """Return the log of the average value of the tails"""
  out = tails(vs,use_fortran=use_fortran) # return all the tails
  ds = np.mean(out,axis=0) # average over waves
  ds = ds[ds>1e-10] # only sizaable elements
  return np.log(ds)



def loclength(vs,use_fortran=use_fortran):
  """Calculate the inverse localization length by fitting the
  decay of the wavefunctions"""
  out = logavgtails(vs,use_fortran=use_fortran) # get the logrho
  ns = np.array(range(len(out))) # length
  ps = np.polyfit(ns,out,1)
  return -ps[0]







def tails_python(vs,use_fortran=use_fortran):
  """Python implementation, Return the log of the tails, 
  centered around the maximum"""
  out = [] # empty list
  for v in vs: # loop over wavefunctions
    d = v*np.conjugate(v) # density
    n = len(d) # size of the array
    nmax = np.argmax(v) # index with the maximum
    ds = np.zeros(n//2) # initialize logd
    for j in range(n//2): # loop over components
      ir = (nmax + j)%n # right index
      il = (nmax - j)%n # left index
      ds[j] = d[ir] + d[il] # average tails
    out.append(ds) # store
  return out # return array


def test_tails(vs):
  """Perform a test of tails comparing Python and fortran"""
  import time
  t0 = time.perf_counter() # timing
  out1 = tails(vs,use_fortran=True) # fortran routine
  t1 = time.perf_counter() # timing
  out2 = tails(vs,use_fortran=False) # python routine
  t2 = time.perf_counter() # timing
  diff = np.mean(np.abs(np.array(out1)-np.array(out2))) # deviation
  avg = np.mean(np.abs(out1) + np.abs(out2))/2. # average
#  print("Deviation",diff)
#  print("Average",avg)
  print("Percentual error",diff/avg)
  print("Time with FORTRAN",t1-t0)
  print("Time with PYTHON",t2-t1)
